"""Open-Meteo Historical Archive API client for ERA5 reanalysis.

Fetches daily max/min temperature from the ERA5 reanalysis dataset.
Free, no authentication required.

API docs: https://open-meteo.com/en/docs/historical-weather-api

NOTE: ERA5 has ~5-day lag. The training window must exclude recent days
to avoid treating missing reanalysis data as station outages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from weather_edge.common.http import HttpClient

logger = logging.getLogger(__name__)

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ERA5 reanalysis typically lags ~5 days behind real time.
ERA5_LAG_DAYS = 5


@dataclass(frozen=True)
class OMDailyObs:
    """Daily max/min temperature from Open-Meteo ERA5 reanalysis."""

    obs_date: date
    max_temp_c: float
    min_temp_c: float


async def fetch_openmeteo_history(
    lat: float,
    lon: float,
    start: date,
    end: date,
    timezone: str = "auto",
) -> list[OMDailyObs]:
    """Fetch ERA5 reanalysis daily max/min temperatures.

    Args:
        lat: Latitude
        lon: Longitude
        start: Start date (inclusive)
        end: End date (inclusive). Should be at least ERA5_LAG_DAYS
             before today to ensure data availability.
        timezone: IANA timezone for daily aggregation alignment.

    Returns:
        List of daily observations. Missing days are omitted.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": timezone,
    }

    async with HttpClient() as client:
        try:
            resp = await client.get(_ARCHIVE_URL, params=params)
            data = resp.json()
        except Exception as exc:
            logger.warning("Failed to fetch Open-Meteo archive for (%.2f, %.2f): %s", lat, lon, exc)
            return []

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    maxs = daily.get("temperature_2m_max", [])
    mins = daily.get("temperature_2m_min", [])

    observations: list[OMDailyObs] = []
    for i, date_str in enumerate(dates):
        if i >= len(maxs) or i >= len(mins):
            break
        if maxs[i] is None or mins[i] is None:
            continue
        try:
            obs_date = date.fromisoformat(date_str)
            observations.append(OMDailyObs(
                obs_date=obs_date,
                max_temp_c=float(maxs[i]),
                min_temp_c=float(mins[i]),
            ))
        except (ValueError, TypeError) as exc:
            logger.debug("Skipping invalid entry at index %d: %s", i, exc)

    logger.info(
        "Fetched %d days of ERA5 data for (%.2f, %.2f) from %s to %s",
        len(observations), lat, lon, start, end,
    )
    return observations


def training_window(days: int = 90) -> tuple[date, date]:
    """Return the (start, end) date range for bias training.

    Excludes the last ERA5_LAG_DAYS to avoid missing data at the end
    of the window being confused with station outages.
    """
    today = datetime.now(timezone.utc).date()
    end = today - timedelta(days=ERA5_LAG_DAYS)
    start = end - timedelta(days=days)
    return start, end
