"""Tests for Open-Meteo ensemble API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from weather_edge.weather.openmeteo import fetch_ensemble, fetch_both_ensembles


@pytest.fixture
def mock_response(openmeteo_ecmwf_response):
    resp = MagicMock()
    resp.json.return_value = openmeteo_ecmwf_response
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_fetch_ensemble_ecmwf(openmeteo_ecmwf_response):
    """Test parsing of ECMWF ensemble response."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = openmeteo_ecmwf_response
    mock_resp.raise_for_status = MagicMock()

    with patch("weather_edge.weather.openmeteo.HttpClient") as MockClient:
        instance = AsyncMock()
        instance.get.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        forecast = await fetch_ensemble(33.45, -112.07, "ecmwf")

    assert forecast.source == "ecmwf"
    assert forecast.lat == 33.45
    assert forecast.lon == -112.07
    assert forecast.n_times == 48
    assert forecast.n_members == 51
    assert forecast.temperature_2m.shape == (48, 51)
    assert forecast.precipitation.shape == (48, 51)


@pytest.mark.asyncio
async def test_fetch_ensemble_gfs(openmeteo_gfs_response):
    """Test parsing of GFS ensemble response."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = openmeteo_gfs_response
    mock_resp.raise_for_status = MagicMock()

    with patch("weather_edge.weather.openmeteo.HttpClient") as MockClient:
        instance = AsyncMock()
        instance.get.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        forecast = await fetch_ensemble(33.45, -112.07, "gfs")

    assert forecast.source == "gfs"
    assert forecast.n_members == 31
    assert forecast.temperature_2m.shape == (48, 31)


@pytest.mark.asyncio
async def test_fetch_both_ensembles(openmeteo_ecmwf_response, openmeteo_gfs_response):
    """Test concurrent fetch of both GFS and ECMWF."""
    call_count = 0

    async def mock_get(url, params=None):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        if params and params.get("models") == "gfs_seamless":
            resp.json.return_value = openmeteo_gfs_response
        else:
            resp.json.return_value = openmeteo_ecmwf_response
        resp.raise_for_status = MagicMock()
        return resp

    with patch("weather_edge.weather.openmeteo.HttpClient") as MockClient:
        instance = AsyncMock()
        instance.get = mock_get
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        gfs, ecmwf = await fetch_both_ensembles(33.45, -112.07)

    assert gfs.source == "gfs"
    assert ecmwf.source == "ecmwf"
    assert gfs.n_members == 31
    assert ecmwf.n_members == 51


@pytest.mark.asyncio
async def test_fetch_ensemble_empty_response():
    """Test handling of response with no ensemble members."""
    empty_response = {
        "latitude": 33.45,
        "longitude": -112.07,
        "hourly": {
            "time": [1000000000, 1000003600],
        },
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = empty_response
    mock_resp.raise_for_status = MagicMock()

    with patch("weather_edge.weather.openmeteo.HttpClient") as MockClient:
        instance = AsyncMock()
        instance.get.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        forecast = await fetch_ensemble(33.45, -112.07, "ecmwf")

    assert forecast.n_members == 0
    assert forecast.n_times == 2
