"""Two-stage market parameter parser.

Stage 1: Regex fast path for clean/standard question formats.
Stage 2: Claude Haiku LLM fallback for creative/inconsistent phrasings.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

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

_TEMP_DEGREES_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*°?\s*([FfCc])",
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

# --- Hurricane patterns ---
_HURRICANE_PATTERN = re.compile(
    r"\b(?:hurricane|tropical\s+storm|cyclone|typhoon)\b",
    re.IGNORECASE,
)

# --- Location extraction ---
_LOCATION_PATTERN = re.compile(
    r"\bin\s+([A-Z][a-zA-Z\s,]+?)(?:\s+(?:on|by|before|during|this|next|in\s+\d|exceed|hit|reach|drop|fall|see|break|go|top|surpass|get|have|record|temperature|temp|rain|snow|precipitation)|\?|$)",
    re.IGNORECASE,
)

# --- Date extraction ---
_DATE_PATTERNS = [
    # "on July 4, 2025" / "on July 4th, 2025"
    re.compile(r"on\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4})", re.IGNORECASE),
    # "by January 15"
    re.compile(r"by\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{0,4})", re.IGNORECASE),
    # "this week" / "this month"
    re.compile(r"(this\s+(?:week|weekend|month))", re.IGNORECASE),
    # "next Monday" etc
    re.compile(r"(next\s+\w+)", re.IGNORECASE),
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
                dt = dt.replace(year=now.year)
                if dt < now:
                    dt = dt.replace(year=now.year + 1)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
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

    # Check precipitation
    precip_match = _PRECIP_PATTERN.search(text)
    if precip_match:
        market_type = MarketType.PRECIPITATION
        threshold = float(precip_match.group(1))
        unit = "in" if "inch" in text.lower() or "in" in precip_match.group(0).lower() else "mm"

    # Check temperature (below)
    temp_below_match = _TEMP_BELOW_PATTERN.search(text)
    if temp_below_match:
        market_type = MarketType.TEMPERATURE
        threshold = float(temp_below_match.group(1))
        unit = temp_below_match.group(2).upper()
        comparison = Comparison.BELOW

    # Check temperature (above) — only if not already matched below
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

    return MarketParams(
        market_type=market_type,
        location=location,
        threshold=threshold,
        comparison=comparison,
        unit=unit,
        target_date=target_date,
        target_date_str=target_date_str,
    )


_PARSE_SYSTEM = """You extract structured weather parameters from prediction market questions.

Respond with ONLY a JSON object:
{
  "market_type": "temperature" | "precipitation" | "hurricane",
  "location": "city, state/country",
  "threshold": 100.0,
  "comparison": "above" | "below" | "between",
  "unit": "F" | "C" | "in" | "mm",
  "target_date": "YYYY-MM-DD" or null,
  "target_date_str": "original date text from question"
}

Examples:
- "Will the Big Apple see triple digits on Independence Day?" →
  {"market_type": "temperature", "location": "New York, NY", "threshold": 100.0, "comparison": "above", "unit": "F", "target_date": "2025-07-04", "target_date_str": "Independence Day"}
- "Will Phoenix break 120F this summer?" →
  {"market_type": "temperature", "location": "Phoenix, AZ", "threshold": 120.0, "comparison": "above", "unit": "F", "target_date": null, "target_date_str": "this summer"}"""


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

        return MarketParams(
            market_type=market_type,
            location=data.get("location", ""),
            threshold=float(data["threshold"]) if data.get("threshold") is not None else None,
            comparison=comparison,
            unit=data.get("unit", "F"),
            target_date=target_date,
            target_date_str=data.get("target_date_str", ""),
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
