"""NOAA/NWS API client (US locations only)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from weather_edge.common.http import HttpClient
from weather_edge.config import get_settings
from weather_edge.weather.models import NOAAAlert, NOAAForecast, NOAAPeriod

logger = logging.getLogger(__name__)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s)


async def fetch_noaa_forecast(lat: float, lon: float) -> NOAAForecast | None:
    """Fetch NOAA/NWS forecast for a US location.

    Two-step process: /points → gridpoint → /forecast.
    Returns None if the location is outside the US or the API fails.
    """
    settings = get_settings()
    headers = {"User-Agent": settings.nws_user_agent, "Accept": "application/geo+json"}

    async with HttpClient(base_url=settings.nws_api_url, headers=headers) as client:
        # Step 1: Get grid coordinates from lat/lon
        try:
            points_resp = await client.get(f"/points/{lat:.4f},{lon:.4f}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.debug("Location (%.4f, %.4f) is outside US coverage", lat, lon)
            else:
                logger.warning("NWS points API HTTP %d for (%.4f, %.4f)", exc.response.status_code, lat, lon)
            return None
        except httpx.TimeoutException:
            logger.warning("NWS points API timeout for (%.4f, %.4f)", lat, lon)
            return None

        points_data = points_resp.json()
        props = points_data.get("properties", {})
        office = props.get("gridId", "")
        grid_x = props.get("gridX", 0)
        grid_y = props.get("gridY", 0)
        forecast_url = props.get("forecast", "")

        if not forecast_url:
            return None

        forecast = NOAAForecast(
            lat=lat,
            lon=lon,
            office=office,
            grid_x=grid_x,
            grid_y=grid_y,
        )

        # Step 2: Get hourly forecast
        try:
            hourly_resp = await client.get(
                f"/gridpoints/{office}/{grid_x},{grid_y}/forecast/hourly"
            )
            hourly_data = hourly_resp.json()
            for period in hourly_data.get("properties", {}).get("periods", []):
                forecast.periods.append(
                    NOAAPeriod(
                        start_time=_parse_iso(period["startTime"]),
                        end_time=_parse_iso(period["endTime"]),
                        temperature=float(period["temperature"]),
                        temperature_unit=period.get("temperatureUnit", "F"),
                        wind_speed=period.get("windSpeed", ""),
                        short_forecast=period.get("shortForecast", ""),
                        detailed_forecast=period.get("detailedForecast", ""),
                        precipitation_probability=period.get("probabilityOfPrecipitation", {}).get("value"),
                    )
                )
        except httpx.HTTPStatusError as exc:
            logger.info("NWS hourly forecast HTTP %d for %s/%d,%d", exc.response.status_code, office, grid_x, grid_y)
        except httpx.TimeoutException:
            logger.info("NWS hourly forecast timeout for %s/%d,%d", office, grid_x, grid_y)
        except (KeyError, ValueError, TypeError) as exc:
            logger.info("NWS hourly forecast parse error: %s", exc)

        # Step 3: Get active alerts
        try:
            alerts_resp = await client.get(
                "/alerts/active",
                params={"point": f"{lat:.4f},{lon:.4f}"},
            )
            alerts_data = alerts_resp.json()
            for feature in alerts_data.get("features", []):
                a = feature.get("properties", {})
                forecast.alerts.append(
                    NOAAAlert(
                        event=a.get("event", ""),
                        headline=a.get("headline", ""),
                        severity=a.get("severity", ""),
                        certainty=a.get("certainty", ""),
                        onset=_parse_iso(a.get("onset")),
                        expires=_parse_iso(a.get("expires")),
                        description=a.get("description", ""),
                    )
                )
        except httpx.HTTPStatusError as exc:
            logger.info("NWS alerts API HTTP %d for (%.4f, %.4f)", exc.response.status_code, lat, lon)
        except httpx.TimeoutException:
            logger.info("NWS alerts API timeout for (%.4f, %.4f)", lat, lon)
        except (KeyError, ValueError, TypeError) as exc:
            logger.info("NWS alerts parse error: %s", exc)

        return forecast
