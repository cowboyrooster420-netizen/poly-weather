"""Weather Underground personal weather station history scraper.

Scrapes daily history from WU's dashboard page. Observation data is
embedded in JSON ``"observations"`` arrays within the page source
(typically in Apollo/RSC cache entries). Each observation block contains
imperial units (°F) under an ``"imperial"`` key.

Primary parser: extracts ``"observations"`` JSON blobs, picks the
end-of-day summary (latest ``obsTimeLocal`` for the target date),
reads ``imperial.tempHigh`` / ``imperial.tempLow``.

Fallback: ``__NEXT_DATA__`` script tag (legacy, rarely present).

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
_STATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9]+$")


@dataclass
class WUDailyObs:
    """Daily high/low observation from a WU personal weather station."""

    station_id: str
    date: date
    high_temp_c: float
    low_temp_c: float


def _parse_observations_json(html: str, station_id: str, target_date: date) -> WUDailyObs | None:
    """Extract daily high/low from embedded observations JSON.

    WU pages embed observation data in JSON blobs containing
    ``"observations"`` arrays (Apollo/RSC cache entries). Each observation
    has ``obsTimeLocal`` and ``imperial.tempHigh`` / ``imperial.tempLow``.

    We find all observation blocks, filter for the target date, and pick
    the one with the latest timestamp (end-of-day summary).
    """
    # Match each observations array containing a stationID entry.
    # Non-greedy so we get individual blocks, not one giant match.
    pattern = r'"observations":\[(\{"stationID".*?\})\]'
    matches = list(re.finditer(pattern, html, re.DOTALL))

    if not matches:
        return None

    date_str = target_date.isoformat()  # "YYYY-MM-DD"
    best_obs: dict | None = None
    best_time = ""

    for m in matches:
        try:
            obs_list = json.loads("[" + m.group(1) + "]")
        except json.JSONDecodeError:
            continue

        for obs in obs_list:
            if not isinstance(obs, dict):
                continue
            obs_time = obs.get("obsTimeLocal", "")
            # Filter: observation must be for the target date
            if not obs_time.startswith(date_str):
                continue
            imperial = obs.get("imperial", {})
            if imperial.get("tempHigh") is None or imperial.get("tempLow") is None:
                continue
            # Keep the latest observation for this date (end-of-day summary)
            if obs_time > best_time:
                best_time = obs_time
                best_obs = obs

    if best_obs is None:
        return None

    try:
        imperial = best_obs["imperial"]
        high_f = float(imperial["tempHigh"])
        low_f = float(imperial["tempLow"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("Could not extract temps from observation JSON for %s: %s", station_id, exc)
        return None

    return WUDailyObs(
        station_id=station_id,
        date=target_date,
        high_temp_c=fahrenheit_to_celsius(high_f),
        low_temp_c=fahrenheit_to_celsius(low_f),
    )


def _parse_next_data(html: str, station_id: str, target_date: date) -> WUDailyObs | None:
    """Legacy fallback: extract from __NEXT_DATA__ script tag if present."""
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
        return None

    try:
        props = data.get("props", {}).get("pageProps", {})
        observations = (
            props.get("observations")
            or props.get("stationData", {}).get("observations")
            or []
        )
        if isinstance(observations, list):
            for obs in observations:
                if not isinstance(obs, dict):
                    continue
                imperial = obs.get("imperial", obs)
                high_f = imperial.get("tempHigh")
                low_f = imperial.get("tempLow")
                if high_f is not None and low_f is not None:
                    return WUDailyObs(
                        station_id=station_id,
                        date=target_date,
                        high_temp_c=fahrenheit_to_celsius(float(high_f)),
                        low_temp_c=fahrenheit_to_celsius(float(low_f)),
                    )
    except (KeyError, TypeError, ValueError):
        pass

    return None


async def fetch_wu_daily(station_id: str, target_date: date) -> WUDailyObs | None:
    """Fetch a single day's high/low from Weather Underground.

    Returns None if the page can't be fetched or parsed.
    """
    if not _STATION_ID_PATTERN.match(station_id):
        logger.warning("Invalid station_id: %s", station_id)
        return None

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

    # Primary: embedded observations JSON (Apollo/RSC cache)
    result = _parse_observations_json(html, station_id, target_date)
    if result is not None:
        return result

    # Fallback: __NEXT_DATA__ script tag (legacy)
    result = _parse_next_data(html, station_id, target_date)
    if result is not None:
        logger.debug("Used __NEXT_DATA__ fallback for %s on %s", station_id, target_date)
        return result

    logger.warning("Could not parse WU data for %s on %s", station_id, target_date)
    return None


async def fetch_wu_history(
    station_id: str,
    start: date,
    end: date,
    *,
    max_concurrent: int = 5,
) -> list[WUDailyObs]:
    """Fetch daily observations for a date range with bounded concurrency.

    Args:
        station_id: WU station ID
        start: Start date (inclusive)
        end: End date (inclusive)
        max_concurrent: Max parallel requests to WU (be polite)

    Returns:
        List of successfully parsed observations sorted by date (may be
        shorter than the date range if some days fail to parse).
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _fetch_one(target_date: date) -> WUDailyObs | None:
        async with semaphore:
            return await fetch_wu_daily(station_id, target_date)

    # Build list of all dates
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)

    # Fetch all days concurrently (bounded by semaphore)
    results = await asyncio.gather(*[_fetch_one(d) for d in days])

    observations = [obs for obs in results if obs is not None]
    observations.sort(key=lambda o: o.date)

    logger.info(
        "Fetched %d/%d days for %s (%s to %s)",
        len(observations), (end - start).days + 1,
        station_id, start, end,
    )
    return observations
