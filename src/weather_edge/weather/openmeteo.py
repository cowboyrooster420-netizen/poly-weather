"""Open-Meteo ensemble API client (GFS + ECMWF) and HRRR deterministic."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

from weather_edge.common.http import HttpClient
from weather_edge.common.types import LatLon
from weather_edge.config import get_settings
from weather_edge.weather.models import EnsembleForecast, HRRRForecast

logger = logging.getLogger(__name__)


async def fetch_ensemble(
    lat: float,
    lon: float,
    source: str = "ecmwf",
) -> EnsembleForecast:
    """Fetch ensemble forecast from Open-Meteo.

    Args:
        lat: Latitude
        lon: Longitude
        source: "ecmwf" (51 members) or "gfs" (31 members)

    Returns:
        EnsembleForecast with temperature and precipitation arrays
    """
    settings = get_settings()

    # Map source to Open-Meteo model names
    model_map = {
        "ecmwf": "ecmwf_ifs025",
        "gfs": "gfs_seamless",
    }
    model = model_map.get(source)
    if model is None:
        raise ValueError(f"Unknown ensemble source: {source!r} (expected 'ecmwf' or 'gfs')")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation",
        "models": model,
        "timeformat": "unixtime",
    }

    async with HttpClient(base_url=settings.openmeteo_api_url) as client:
        resp = await client.get("/ensemble", params=params)
        data = resp.json()

    hourly = data.get("hourly")
    if hourly is None:
        raise ValueError(f"Open-Meteo response missing 'hourly' key for ({lat}, {lon})")
    times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in hourly["time"]]

    # Parse ensemble members — Open-Meteo returns temperature_2m_member01, etc.
    temp_members = []
    precip_members = []

    for key, values in hourly.items():
        if key.startswith("temperature_2m_member"):
            temp_members.append(values)
        elif key.startswith("precipitation_member"):
            precip_members.append(values)

    # Shape: (n_times, n_members), replace None with NaN
    def _to_array(member_lists: list[list]) -> np.ndarray:
        arr = np.array(member_lists, dtype=np.float64).T
        return np.where(np.isnan(arr), np.nan, arr)

    temp_array = _to_array(temp_members) if temp_members else np.empty((len(times), 0))
    precip_array = _to_array(precip_members) if precip_members else np.empty((len(times), 0))

    return EnsembleForecast(
        source=source,
        lat=data.get("latitude", lat),
        lon=data.get("longitude", lon),
        times=times,
        temperature_2m=temp_array,
        precipitation=precip_array,
    )


async def fetch_both_ensembles(lat: float, lon: float) -> tuple[EnsembleForecast, EnsembleForecast]:
    """Fetch both GFS and ECMWF ensemble forecasts concurrently."""
    import asyncio

    gfs_task = asyncio.create_task(fetch_ensemble(lat, lon, "gfs"))
    ecmwf_task = asyncio.create_task(fetch_ensemble(lat, lon, "ecmwf"))
    gfs, ecmwf = await asyncio.gather(gfs_task, ecmwf_task)
    return gfs, ecmwf


def _is_conus(lat: float, lon: float) -> bool:
    """Check whether (lat, lon) is within the CONUS bounding box."""
    return 24.0 <= lat <= 50.0 and -125.0 <= lon <= -66.0


async def fetch_hrrr(lat: float, lon: float) -> HRRRForecast | None:
    """Fetch HRRR deterministic forecast from Open-Meteo.

    Uses the standard forecast endpoint with ``models=hrrr_conus``.
    Returns ``None`` for non-CONUS locations.
    """
    if not _is_conus(lat, lon):
        logger.debug("HRRR skipped: (%.2f, %.2f) outside CONUS", lat, lon)
        return None

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "models": "hrrr_conus",
        "timeformat": "unixtime",
    }

    settings = get_settings()

    async with HttpClient(base_url=settings.openmeteo_forecast_api_url) as client:
        resp = await client.get("/forecast", params=params)
        data = resp.json()

    hourly = data.get("hourly")
    if hourly is None:
        logger.warning("HRRR response missing 'hourly' key for (%.2f, %.2f)", lat, lon)
        return None

    times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in hourly["time"]]
    temps = hourly.get("temperature_2m")
    if temps is None:
        logger.warning("HRRR response missing temperature_2m for (%.2f, %.2f)", lat, lon)
        return None

    temp_array = np.array(temps, dtype=np.float64)

    return HRRRForecast(
        lat=data.get("latitude", lat),
        lon=data.get("longitude", lon),
        times=times,
        temperature_2m=temp_array,
    )
