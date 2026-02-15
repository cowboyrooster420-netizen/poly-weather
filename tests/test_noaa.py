"""Tests for NOAA/NWS API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weather_edge.weather.noaa import fetch_noaa_forecast


def _make_mock_response(json_data):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_fetch_noaa_forecast_success(
    noaa_points_response, noaa_hourly_response, noaa_alerts_response
):
    """Test successful NOAA forecast fetch."""
    responses = [
        _make_mock_response(noaa_points_response),
        _make_mock_response(noaa_hourly_response),
        _make_mock_response(noaa_alerts_response),
    ]
    call_idx = 0

    async def mock_get(url, params=None):
        nonlocal call_idx
        resp = responses[call_idx]
        call_idx += 1
        return resp

    with patch("weather_edge.weather.noaa.HttpClient") as MockClient:
        instance = AsyncMock()
        instance.get = mock_get
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        forecast = await fetch_noaa_forecast(33.45, -112.07)

    assert forecast is not None
    assert forecast.office == "PSR"
    assert forecast.grid_x == 100
    assert forecast.grid_y == 50
    assert len(forecast.periods) == 48
    assert forecast.periods[0].temperature == 95.0
    assert forecast.periods[0].temperature_unit == "F"
    assert forecast.periods[0].precipitation_probability == 10
    assert len(forecast.alerts) == 0


@pytest.mark.asyncio
async def test_fetch_noaa_forecast_with_alerts(
    noaa_points_response, noaa_hourly_response, noaa_alerts_response_with_hurricane
):
    """Test NOAA forecast with active hurricane alert."""
    responses = [
        _make_mock_response(noaa_points_response),
        _make_mock_response(noaa_hourly_response),
        _make_mock_response(noaa_alerts_response_with_hurricane),
    ]
    call_idx = 0

    async def mock_get(url, params=None):
        nonlocal call_idx
        resp = responses[call_idx]
        call_idx += 1
        return resp

    with patch("weather_edge.weather.noaa.HttpClient") as MockClient:
        instance = AsyncMock()
        instance.get = mock_get
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        forecast = await fetch_noaa_forecast(33.45, -112.07)

    assert forecast is not None
    assert len(forecast.alerts) == 1
    assert forecast.alerts[0].event == "Hurricane Warning"
    assert forecast.alerts[0].severity == "Extreme"


@pytest.mark.asyncio
async def test_fetch_noaa_forecast_outside_us():
    """Test that non-US locations return None."""
    import httpx

    mock_response = httpx.Response(404, request=httpx.Request("GET", "https://api.weather.gov/points/51.5000,-0.1200"))
    with patch("weather_edge.weather.noaa.HttpClient") as MockClient:
        instance = AsyncMock()
        instance.get = AsyncMock(side_effect=httpx.HTTPStatusError("404", request=mock_response.request, response=mock_response))
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        forecast = await fetch_noaa_forecast(51.5, -0.12)  # London

    assert forecast is None


@pytest.mark.asyncio
async def test_fetch_noaa_forecast_no_forecast_url():
    """Test handling of missing forecast URL in points response."""
    points_data = {
        "properties": {
            "gridId": "PSR",
            "gridX": 100,
            "gridY": 50,
            # Missing "forecast" key
        }
    }

    with patch("weather_edge.weather.noaa.HttpClient") as MockClient:
        instance = AsyncMock()
        instance.get.return_value = _make_mock_response(points_data)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        forecast = await fetch_noaa_forecast(33.45, -112.07)

    assert forecast is None
