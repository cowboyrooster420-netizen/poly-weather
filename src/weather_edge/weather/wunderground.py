"""Weather Underground personal weather station history scraper.

Scrapes daily history from WU's history page. WU is a Next.js app —
observation data is embedded in a ``__NEXT_DATA__`` JSON blob in the
page source. Falls back to regex parsing of rendered HTML when the
Next.js structure isn't present or changes format.

Target URL pattern:
    https://www.wunderground.com/dashboard/pws/{station_id}/table/{YYYY-M-D}/{YYYY-M-D}/daily
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta

from weather_edge.common.http import HttpClient
from weather_edge.common.types import fahrenheit_to_celsius

logger = logging.getLogger(__name__)

_WU_BASE = "https://www.wunderground.com"


@dataclass
class WUDailyObs:
    """Daily high/low observation from a WU personal weather station."""

    station_id: str
    date: date
    high_temp_c: float
    low_temp_c: float


def _parse_next_data(html: str, station_id: str, target_date: date) -> WUDailyObs | None:
    """Extract daily high/low from __NEXT_DATA__ JSON blob."""
    # Match everything between the script tags — the JSON is deeply nested
    # so a non-greedy {.*?} would stop at the first }, producing invalid JSON.
    match = re.search(
        r'<script\s+id="__NEXT_DATA__"\s+type="application/json">\s*(.+?)\s*</script>',
        html,
        re.DOTALL,
    )
    if match is None:
        return None

    try:
        data = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        logger.warning("Failed to parse __NEXT_DATA__ JSON for %s", station_id)
        return None

    # Navigate the Next.js data structure to find daily observations.
    # The path may vary; try common locations.
    try:
        props = data.get("props", {}).get("pageProps", {})
        # Try the daily summary path
        daily = props.get("dailySummary") or props.get("summary") or {}

        # Look for temperature data in various possible structures
        if isinstance(daily, dict):
            high_f = daily.get("temperature", {}).get("high") or daily.get("tempHigh")
            low_f = daily.get("temperature", {}).get("low") or daily.get("tempLow")
            if high_f is not None and low_f is not None:
                return WUDailyObs(
                    station_id=station_id,
                    date=target_date,
                    high_temp_c=fahrenheit_to_celsius(float(high_f)),
                    low_temp_c=fahrenheit_to_celsius(float(low_f)),
                )

        # Try observations array path
        observations = (
            props.get("observations")
            or props.get("stationData", {}).get("observations")
            or []
        )
        if isinstance(observations, list) and observations:
            # Daily summary often has a single entry with high/low
            for obs in observations:
                if not isinstance(obs, dict):
                    continue
                # Try imperial units sub-object
                imperial = obs.get("imperial", obs)
                high_f = imperial.get("tempHigh") or imperial.get("tempMax")
                low_f = imperial.get("tempLow") or imperial.get("tempMin")
                if high_f is not None and low_f is not None:
                    return WUDailyObs(
                        station_id=station_id,
                        date=target_date,
                        high_temp_c=fahrenheit_to_celsius(float(high_f)),
                        low_temp_c=fahrenheit_to_celsius(float(low_f)),
                    )

    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("Could not extract temps from __NEXT_DATA__ for %s: %s", station_id, exc)

    return None


def _parse_html_fallback(html: str, station_id: str, target_date: date) -> WUDailyObs | None:
    """Fallback: regex-parse temperature values from rendered HTML.

    Uses multiple patterns in priority order, preferring more specific
    matches to avoid capturing unrelated page elements (dew point, ads, etc).
    """
    # Try specific WU table patterns first, then broader patterns.
    high_patterns = [
        # WU table cell: "High\n95 °F" or "High</span>95"
        r'(?:High\s*(?:Temp(?:erature)?)?)\s*(?:</\w+>)?\s*(-?\d+(?:\.\d+)?)\s*°?\s*F',
        # Fallback: "High: 95 °F" with limited gap (max 20 chars to avoid
        # matching across unrelated elements)
        r'High\s*(?:Temp(?:erature)?)?\s*[:=]?\s{0,5}(-?\d+(?:\.\d+)?)\s*°?\s*F',
    ]
    low_patterns = [
        r'(?:Low\s*(?:Temp(?:erature)?)?)\s*(?:</\w+>)?\s*(-?\d+(?:\.\d+)?)\s*°?\s*F',
        r'Low\s*(?:Temp(?:erature)?)?\s*[:=]?\s{0,5}(-?\d+(?:\.\d+)?)\s*°?\s*F',
    ]

    high_match = None
    for pat in high_patterns:
        high_match = re.search(pat, html, re.IGNORECASE)
        if high_match:
            break

    low_match = None
    for pat in low_patterns:
        low_match = re.search(pat, html, re.IGNORECASE)
        if low_match:
            break

    if high_match is None or low_match is None:
        return None

    try:
        high_f = float(high_match.group(1))
        low_f = float(low_match.group(1))
    except ValueError:
        return None

    return WUDailyObs(
        station_id=station_id,
        date=target_date,
        high_temp_c=fahrenheit_to_celsius(high_f),
        low_temp_c=fahrenheit_to_celsius(low_f),
    )


async def fetch_wu_daily(station_id: str, target_date: date) -> WUDailyObs | None:
    """Fetch a single day's high/low from Weather Underground.

    Returns None if the page can't be fetched or parsed.
    """
    # WU URL format uses non-zero-padded month and day
    date_str = f"{target_date.year}-{target_date.month}-{target_date.day}"
    url = f"{_WU_BASE}/dashboard/pws/{station_id}/table/{date_str}/{date_str}/daily"

    async with HttpClient() as client:
        try:
            resp = await client.get(url)
            html = resp.text
        except Exception as exc:
            logger.warning("Failed to fetch WU page for %s on %s: %s", station_id, target_date, exc)
            return None

    # Try __NEXT_DATA__ first, then HTML fallback
    result = _parse_next_data(html, station_id, target_date)
    if result is not None:
        return result

    result = _parse_html_fallback(html, station_id, target_date)
    if result is not None:
        logger.debug("Used HTML fallback for %s on %s", station_id, target_date)
        return result

    logger.warning("Could not parse WU data for %s on %s", station_id, target_date)
    return None


async def fetch_wu_history(
    station_id: str,
    start: date,
    end: date,
    *,
    rate_limit_delay: float = 0.5,
) -> list[WUDailyObs]:
    """Fetch daily observations for a date range, rate-limited.

    Args:
        station_id: WU station ID
        start: Start date (inclusive)
        end: End date (inclusive)
        rate_limit_delay: Seconds between requests (~2 req/sec)

    Returns:
        List of successfully parsed observations (may be shorter than
        the date range if some days fail to parse).
    """
    observations: list[WUDailyObs] = []
    current = start

    while current <= end:
        obs = await fetch_wu_daily(station_id, current)
        if obs is not None:
            observations.append(obs)

        current += timedelta(days=1)

        # Rate limit to be polite to WU
        if current <= end:
            await asyncio.sleep(rate_limit_delay)

    logger.info(
        "Fetched %d/%d days for %s (%s to %s)",
        len(observations), (end - start).days + 1,
        station_id, start, end,
    )
    return observations
