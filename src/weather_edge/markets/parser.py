"""Two-stage market parameter parser.

Stage 1: Regex fast path for clean/standard question formats.
Stage 2: Claude Haiku LLM fallback for creative/inconsistent phrasings.
"""

from __future__ import annotations

import json
import logging
import re
import calendar
from datetime import datetime, timedelta, timezone

import anthropic

from weather_edge.common.llm import ask_haiku
from weather_edge.markets.models import Comparison, MarketParams, MarketType
from weather_edge.weather.geocoding import geocode

logger = logging.getLogger(__name__)

# --- Temperature patterns ---
_TEMP_PATTERN = re.compile(
    r"(?:temperature|temp|high|low)"
    r".*?"
    r"(?:(?:reach|exceed|hit|above|over|surpass|top|go\s+above|break)\s+)?"
    r"(\d+(?:\.\d+)?)\s*°?\s*([FfCc])",
    re.IGNORECASE,
)

_TEMP_BELOW_PATTERN = re.compile(
    r"(?:temperature|temp|high|low)"
    r".*?"
    r"(?:below|under|drop\s+(?:below|under)|fall\s+(?:below|under)|dip\s+below)\s+"
    r"(\d+(?:\.\d+)?)\s*°?\s*([FfCc])",
    re.IGNORECASE,
)

_TEMP_BETWEEN_PATTERN = re.compile(
    r"(?:temperature|temp|high|low)?"
    r".*?"
    r"between\s+(\d+(?:\.\d+)?)\s*°?\s*([FfCc]?)"
    r"\s*(?:and|[-–])\s*"
    r"(\d+(?:\.\d+)?)\s*°?\s*([FfCc])",
    re.IGNORECASE,
)

# Polymarket bucket: "be 7°C on" → single-degree bucket (BETWEEN 6.5 and 7.5)
_TEMP_BUCKET_EXACT_PATTERN = re.compile(
    r"(?:highest|lowest|high|low)?\s*temperature.*?"
    r"\bbe\s+(-?\d+(?:\.\d+)?)\s*°\s*([FfCc])\b"
    r"(?!\s*or\b)",  # NOT followed by "or below"/"or higher"
    re.IGNORECASE,
)

# Polymarket bucket: "be 4°C or below" / "be 12°C or higher"
_TEMP_BUCKET_EDGE_PATTERN = re.compile(
    r"(?:highest|lowest|high|low)?\s*temperature.*?"
    r"\bbe\s+(-?\d+(?:\.\d+)?)\s*°\s*([FfCc])\s+or\s+(below|lower|higher|above)",
    re.IGNORECASE,
)

# Polymarket F-range bucket: "32-33°F", "38-39°F" → BETWEEN
_TEMP_RANGE_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*°\s*([FfCc])",
    re.IGNORECASE,
)

_TEMP_DEGREES_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*([FfCc])",
    re.IGNORECASE,
)

# --- Precipitation patterns ---
_PRECIP_PATTERN = re.compile(
    r"(?:rain(?:fall)?|precipitation|precip|snow(?:fall)?)"
    r".*?"
    r"(?:exceed|above|over|more\s+than|at\s+least)\s+"
    r"(\d+(?:\.\d+)?)\s*(?:inches?|in|mm|cm)",
    re.IGNORECASE,
)

# "between 5 and 6 inches of rain" / "precipitation between 3 and 4 inches"
_PRECIP_BETWEEN_PATTERN = re.compile(
    r"(?:rain(?:fall)?|precipitation|precip|snow(?:fall)?)?"
    r".*?"
    r"between\s+(\d+(?:\.\d+)?)\s*(?:and|[-–])\s*(\d+(?:\.\d+)?)\s*(?:inches?|in|mm|cm)"
    r"|"
    r"between\s+(\d+(?:\.\d+)?)\s*(?:and|[-–])\s*(\d+(?:\.\d+)?)\s*(?:inches?|in|mm|cm)"
    r".*?(?:of\s+)?(?:rain(?:fall)?|precipitation|precip|snow(?:fall)?)",
    re.IGNORECASE,
)

# Polymarket precipitation inch-mark: '3-4"' / '<3"' / '>8"'
_PRECIP_RANGE_INCH_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*\"",
)

_PRECIP_ABOVE_INCH_PATTERN = re.compile(
    r"(?:>|(?:more|greater)\s+than)\s*(\d+(?:\.\d+)?)\s*\"",
)

_PRECIP_BELOW_INCH_PATTERN = re.compile(
    r"(?:<|(?:less|fewer)\s+than)\s*(\d+(?:\.\d+)?)\s*\"",
)

# --- Hurricane patterns ---
_HURRICANE_PATTERN = re.compile(
    r"\b(?:hurricane|tropical\s+storm|cyclone|typhoon)\b",
    re.IGNORECASE,
)

# --- Location extraction ---
_LOCATION_PATTERN = re.compile(
    r"\bin\s+([A-Z][a-zA-Z\s,]+?)(?:\s+(?:on|by|be|before|during|this|next|in\s+\d|exceed|hit|reach|drop|fall|see|break|go|top|surpass|get|have|record|temperature|temp|rain|snow|precipitation)|\?|$)",
    re.IGNORECASE,
)

# --- Date extraction ---
_DATE_PATTERNS = [
    # "on July 4, 2025" / "on July 4th, 2025"
    re.compile(r"\bon\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4})", re.IGNORECASE),
    # "on February 16" / "on February 16th" (no year)
    re.compile(r"\bon\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?)\b", re.IGNORECASE),
    # "by January 15"
    re.compile(r"\bby\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4})", re.IGNORECASE),
    # "this week" / "this month"
    re.compile(r"(this\s+(?:week|weekend|month))", re.IGNORECASE),
    # "next Monday" etc
    re.compile(r"(next\s+\w+)", re.IGNORECASE),
    # "in February 2026" — month name with year
    re.compile(
        r"\bin\s+((?:january|february|march|april|may|june|july|august|september|october|november|december"
        r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{4})\b",
        re.IGNORECASE,
    ),
    # "in February" / "in March" — bare month name
    re.compile(
        r"\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december"
        r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
        re.IGNORECASE,
    ),
]

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_date_str(date_str: str) -> datetime | None:
    """Try to parse a date string into a datetime."""
    # Remove ordinal suffixes
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str.strip())
    # Try common formats
    for fmt in ["%B %d, %Y", "%B %d %Y", "%B %d"]:
        try:
            dt = datetime.strptime(cleaned, fmt)
            if dt.year == 1900:  # No year provided
                now = datetime.now(timezone.utc)
                dt = dt.replace(year=now.year, tzinfo=timezone.utc)
                if dt < now:
                    dt = dt.replace(year=now.year + 1)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _resolve_period(date_str: str) -> tuple[datetime, datetime] | None:
    """Convert a date string into a (period_start, period_end) range.

    Handles bare month names ("February"), "this week", "this month".
    Returns None when the string doesn't describe a period.
    """
    lower = date_str.strip().lower()
    now = datetime.now(timezone.utc)

    if lower in ("this week",):
        # Monday 00:00 → Sunday 23:59
        monday = now - timedelta(days=now.weekday())
        start = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        return start, end

    if lower in ("this month",):
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_day = calendar.monthrange(now.year, now.month)[1]
        end = start.replace(day=last_day, hour=23, minute=59, second=59)
        return start, end

    # Month + year: "february 2026"
    parts = lower.split()
    if len(parts) == 2 and parts[1].isdigit():
        month_num = _MONTH_NAMES.get(parts[0])
        if month_num is not None:
            year = int(parts[1])
            start = datetime(year, month_num, 1, tzinfo=timezone.utc)
            last_day = calendar.monthrange(year, month_num)[1]
            end = datetime(year, month_num, last_day, 23, 59, 59, tzinfo=timezone.utc)
            return start, end

    # Bare month name
    month_num = _MONTH_NAMES.get(lower)
    if month_num is not None:
        year = now.year
        # If the month is already past, assume next year
        if month_num < now.month:
            year += 1
        start = datetime(year, month_num, 1, tzinfo=timezone.utc)
        last_day = calendar.monthrange(year, month_num)[1]
        end = datetime(year, month_num, last_day, 23, 59, 59, tzinfo=timezone.utc)
        return start, end

    return None


async def parse_market(question: str, description: str = "") -> MarketParams | None:
    """Parse a weather market question into structured parameters.

    Two-stage: regex fast path, then LLM fallback.
    Returns None if parsing fails completely.
    """
    text = f"{question} {description}"

    # Stage 1: Try regex extraction
    params = _regex_parse(text)
    if params is not None and params.location:
        # Geocode the location
        latlon = await geocode(params.location)
        if latlon:
            params.lat_lon = latlon
        return params

    # Stage 2: LLM fallback
    params = await _llm_parse(question, description)
    if params is not None and params.location:
        latlon = await geocode(params.location)
        if latlon:
            params.lat_lon = latlon
    return params


def _regex_parse(text: str) -> MarketParams | None:
    """Try to extract market params using regex patterns."""
    # Determine market type
    market_type = MarketType.UNKNOWN
    threshold = None
    comparison = Comparison.ABOVE
    unit = "F"

    # Check hurricane first (simplest)
    if _HURRICANE_PATTERN.search(text):
        market_type = MarketType.HURRICANE

    # Check precipitation — BETWEEN first, then range/above/below inch-mark, then ABOVE
    threshold_upper = None
    precip_btwn = _PRECIP_BETWEEN_PATTERN.search(text)
    if precip_btwn:
        # Groups 1,2 for first alternative; 3,4 for second
        lo = precip_btwn.group(1) or precip_btwn.group(3)
        hi = precip_btwn.group(2) or precip_btwn.group(4)
        if lo and hi:
            market_type = MarketType.PRECIPITATION
            threshold = float(lo)
            threshold_upper = float(hi)
            comparison = Comparison.BETWEEN
            unit = "in" if "inch" in text.lower() or "in" in precip_btwn.group(0).lower() else "mm"

    # Polymarket inch-mark range: '3-4"' → BETWEEN
    if market_type != MarketType.PRECIPITATION:
        precip_range = _PRECIP_RANGE_INCH_PATTERN.search(text)
        if precip_range:
            market_type = MarketType.PRECIPITATION
            threshold = float(precip_range.group(1))
            threshold_upper = float(precip_range.group(2))
            comparison = Comparison.BETWEEN
            unit = "in"

    # Polymarket inch-mark above: '>8"'
    if market_type != MarketType.PRECIPITATION:
        precip_above = _PRECIP_ABOVE_INCH_PATTERN.search(text)
        if precip_above:
            market_type = MarketType.PRECIPITATION
            threshold = float(precip_above.group(1))
            comparison = Comparison.ABOVE
            unit = "in"

    # Polymarket inch-mark below: '<3"'
    if market_type != MarketType.PRECIPITATION:
        precip_below = _PRECIP_BELOW_INCH_PATTERN.search(text)
        if precip_below:
            market_type = MarketType.PRECIPITATION
            threshold = float(precip_below.group(1))
            comparison = Comparison.BELOW
            unit = "in"

    precip_match = _PRECIP_PATTERN.search(text)
    if precip_match and market_type != MarketType.PRECIPITATION:
        market_type = MarketType.PRECIPITATION
        threshold = float(precip_match.group(1))
        unit = "in" if "inch" in text.lower() or "in" in precip_match.group(0).lower() else "mm"

    # Polymarket bucket patterns — most specific, check first
    # "be 4°C or below" / "be 12°C or higher"
    if market_type != MarketType.PRECIPITATION:
        bucket_edge = _TEMP_BUCKET_EDGE_PATTERN.search(text)
        if bucket_edge:
            market_type = MarketType.TEMPERATURE
            threshold = float(bucket_edge.group(1))
            unit = bucket_edge.group(2).upper()
            direction = bucket_edge.group(3).lower()
            if direction in ("below", "lower"):
                comparison = Comparison.BELOW
            else:
                comparison = Comparison.ABOVE

    # "be 7°C on" — single-degree bucket → BETWEEN (val - 0.5, val + 0.5)
    if market_type != MarketType.PRECIPITATION and market_type != MarketType.TEMPERATURE:
        bucket_exact = _TEMP_BUCKET_EXACT_PATTERN.search(text)
        if bucket_exact:
            val = float(bucket_exact.group(1))
            market_type = MarketType.TEMPERATURE
            threshold = val - 0.5
            threshold_upper = val + 0.5
            unit = bucket_exact.group(2).upper()
            comparison = Comparison.BETWEEN

    # Polymarket F-range: "32-33°F" → BETWEEN
    if market_type != MarketType.PRECIPITATION and market_type != MarketType.TEMPERATURE:
        temp_range = _TEMP_RANGE_PATTERN.search(text)
        if temp_range:
            market_type = MarketType.TEMPERATURE
            threshold = float(temp_range.group(1))
            threshold_upper = float(temp_range.group(2))
            unit = temp_range.group(3).upper()
            comparison = Comparison.BETWEEN

    # Check temperature (between) — "between 45 and 50F"
    if market_type != MarketType.PRECIPITATION and market_type != MarketType.TEMPERATURE:
        temp_btwn = _TEMP_BETWEEN_PATTERN.search(text)
        if temp_btwn:
            market_type = MarketType.TEMPERATURE
            threshold = float(temp_btwn.group(1))
            threshold_upper = float(temp_btwn.group(3))
            # Unit from the upper bound (always present); lower may omit it
            unit = (temp_btwn.group(4) or temp_btwn.group(2)).upper()
            comparison = Comparison.BETWEEN

    # Check temperature (below)
    if market_type != MarketType.TEMPERATURE and market_type != MarketType.PRECIPITATION:
        temp_below_match = _TEMP_BELOW_PATTERN.search(text)
        if temp_below_match:
            market_type = MarketType.TEMPERATURE
            threshold = float(temp_below_match.group(1))
            unit = temp_below_match.group(2).upper()
            comparison = Comparison.BELOW

    # Check temperature (above) — only if not already matched
    if market_type not in (MarketType.TEMPERATURE, MarketType.PRECIPITATION):
        temp_match = _TEMP_PATTERN.search(text)
        if temp_match:
            market_type = MarketType.TEMPERATURE
            threshold = float(temp_match.group(1))
            unit = temp_match.group(2).upper()
            comparison = Comparison.ABOVE

    # If still unknown but we found degrees, try simple extraction
    if market_type == MarketType.UNKNOWN:
        deg_match = _TEMP_DEGREES_PATTERN.search(text)
        if deg_match:
            market_type = MarketType.TEMPERATURE
            threshold = float(deg_match.group(1))
            unit = deg_match.group(2).upper()
            # Guess direction from context
            if re.search(r"\b(below|under|drop|fall|dip|cold|low)\b", text, re.IGNORECASE):
                comparison = Comparison.BELOW

    if market_type == MarketType.UNKNOWN:
        return None

    # Extract location
    location = ""
    loc_match = _LOCATION_PATTERN.search(text)
    if loc_match:
        location = loc_match.group(1).strip().rstrip(",")

    # Extract date
    target_date = None
    target_date_str = ""
    for pat in _DATE_PATTERNS:
        date_match = pat.search(text)
        if date_match:
            target_date_str = date_match.group(1)
            target_date = _parse_date_str(target_date_str)
            break

    # Resolve period bounds for precipitation markets
    period_start = None
    period_end = None
    if market_type == MarketType.PRECIPITATION and target_date_str:
        period = _resolve_period(target_date_str)
        if period is not None:
            period_start, period_end = period
            # Use mid-point as target_date for lead-time calculation
            if target_date is None:
                target_date = period_start + (period_end - period_start) / 2

    # Detect daily aggregation for temperature markets
    daily_aggregation = None
    if market_type == MarketType.TEMPERATURE:
        if re.search(r"\bhighest\s+temperature\b", text, re.IGNORECASE):
            daily_aggregation = "max"
        elif re.search(r"\blowest\s+temperature\b", text, re.IGNORECASE):
            daily_aggregation = "min"

    return MarketParams(
        market_type=market_type,
        location=location,
        threshold=threshold,
        threshold_upper=threshold_upper,
        comparison=comparison,
        unit=unit,
        target_date=target_date,
        target_date_str=target_date_str,
        period_start=period_start,
        period_end=period_end,
        daily_aggregation=daily_aggregation,
    )


_PARSE_SYSTEM = """You extract structured weather parameters from prediction market questions.

Respond with ONLY a JSON object:
{
  "market_type": "temperature" | "precipitation" | "hurricane",
  "location": "city, state/country",
  "threshold": 100.0,
  "threshold_upper": null,
  "comparison": "above" | "below" | "between",
  "unit": "F" | "C" | "in" | "mm",
  "target_date": "YYYY-MM-DD" or null,
  "target_date_str": "original date text from question",
  "period_start": "YYYY-MM-DD" or null,
  "period_end": "YYYY-MM-DD" or null,
  "daily_aggregation": "max" | "min" | null
}

For precipitation markets that ask about a time period (e.g. "in February", "this month"),
set period_start/period_end to the first/last day of that period.
For BETWEEN comparisons set threshold to the lower bound and threshold_upper to the upper bound.
For temperature markets asking about "highest temperature", set daily_aggregation to "max".
For temperature markets asking about "lowest temperature", set daily_aggregation to "min".
Otherwise set daily_aggregation to null.

Examples:
- "Will the Big Apple see triple digits on Independence Day?" →
  {"market_type": "temperature", "location": "New York, NY", "threshold": 100.0, "threshold_upper": null, "comparison": "above", "unit": "F", "target_date": "2025-07-04", "target_date_str": "Independence Day", "period_start": null, "period_end": null, "daily_aggregation": null}
- "Will Seattle have between 5 and 6 inches of rain in February?" →
  {"market_type": "precipitation", "location": "Seattle, WA", "threshold": 5.0, "threshold_upper": 6.0, "comparison": "between", "unit": "in", "target_date": null, "target_date_str": "February", "period_start": "2026-02-01", "period_end": "2026-02-28", "daily_aggregation": null}
- "Will the highest temperature in Atlanta be 68°F or higher on February 18?" →
  {"market_type": "temperature", "location": "Atlanta, GA", "threshold": 68.0, "threshold_upper": null, "comparison": "above", "unit": "F", "target_date": "2026-02-18", "target_date_str": "February 18", "period_start": null, "period_end": null, "daily_aggregation": "max"}"""


async def _llm_parse(question: str, description: str) -> MarketParams | None:
    """Use Claude Haiku to parse an ambiguous market question."""
    user_prompt = f"Market question: {question}"
    if description:
        user_prompt += f"\nDescription: {description[:500]}"

    try:
        response = await ask_haiku(_PARSE_SYSTEM, user_prompt, max_tokens=300)
        json_match = re.search(r"\{[^}]*\}", response, re.DOTALL)
        if not json_match:
            logger.debug("LLM parser returned no JSON for: %s", question[:80])
            return None

        data = json.loads(json_match.group())

        try:
            market_type = MarketType(data.get("market_type", "unknown"))
        except ValueError:
            market_type = MarketType.UNKNOWN
        try:
            comparison = Comparison(data.get("comparison", "above"))
        except ValueError:
            comparison = Comparison.ABOVE

        target_date = None
        if data.get("target_date"):
            try:
                target_date = datetime.strptime(data["target_date"], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass

        # Extract period bounds
        period_start = None
        period_end = None
        if data.get("period_start"):
            try:
                period_start = datetime.strptime(data["period_start"], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass
        if data.get("period_end"):
            try:
                period_end = datetime.strptime(data["period_end"], "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59, tzinfo=timezone.utc
                )
            except ValueError:
                pass

        # If period is set but no target_date, use midpoint
        if period_start and period_end and target_date is None:
            target_date = period_start + (period_end - period_start) / 2

        # Extract daily_aggregation
        daily_aggregation = data.get("daily_aggregation")
        if daily_aggregation not in ("max", "min"):
            daily_aggregation = None

        return MarketParams(
            market_type=market_type,
            location=data.get("location") or "",
            threshold=float(data["threshold"]) if data.get("threshold") is not None else None,
            threshold_upper=(
                float(data["threshold_upper"])
                if data.get("threshold_upper") is not None
                else None
            ),
            comparison=comparison,
            unit=data.get("unit", "F"),
            target_date=target_date,
            target_date_str=data.get("target_date_str", ""),
            period_start=period_start,
            period_end=period_end,
            daily_aggregation=daily_aggregation,
        )
    except json.JSONDecodeError as exc:
        logger.warning("LLM parser returned invalid JSON: %s", exc)
        return None
    except anthropic.APIError as exc:
        logger.warning("LLM parser API error: %s", exc)
        return None
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("LLM parser data extraction error: %s", exc)
        return None
