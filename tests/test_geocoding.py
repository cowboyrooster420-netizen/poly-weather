"""Tests for geocoding module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weather_edge.weather.geocoding import geocode, _cache


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear geocoding cache before each test."""
    _cache.clear()
    yield
    _cache.clear()


@pytest.mark.asyncio
async def test_geocode_success():
    """Test successful geocoding."""
    mock_location = MagicMock()
    mock_location.latitude = 33.4484
    mock_location.longitude = -112.0740

    mock_geolocator = AsyncMock()
    mock_geolocator.geocode = AsyncMock(return_value=mock_location)
    mock_geolocator.__aenter__ = AsyncMock(return_value=mock_geolocator)
    mock_geolocator.__aexit__ = AsyncMock(return_value=False)

    with patch("weather_edge.weather.geocoding.Nominatim", return_value=mock_geolocator):
        result = await geocode("Phoenix, AZ")

    assert result is not None
    assert abs(result[0] - 33.4484) < 0.001
    assert abs(result[1] - (-112.0740)) < 0.001


@pytest.mark.asyncio
async def test_geocode_not_found():
    """Test geocoding for unknown location."""
    mock_geolocator = AsyncMock()
    mock_geolocator.geocode = AsyncMock(return_value=None)
    mock_geolocator.__aenter__ = AsyncMock(return_value=mock_geolocator)
    mock_geolocator.__aexit__ = AsyncMock(return_value=False)

    with patch("weather_edge.weather.geocoding.Nominatim", return_value=mock_geolocator):
        result = await geocode("Nonexistent Place XYZ123")

    assert result is None


@pytest.mark.asyncio
async def test_geocode_caching():
    """Test that geocoding results are cached."""
    mock_location = MagicMock()
    mock_location.latitude = 40.7128
    mock_location.longitude = -74.0060

    mock_geolocator = AsyncMock()
    mock_geolocator.geocode = AsyncMock(return_value=mock_location)
    mock_geolocator.__aenter__ = AsyncMock(return_value=mock_geolocator)
    mock_geolocator.__aexit__ = AsyncMock(return_value=False)

    with patch("weather_edge.weather.geocoding.Nominatim", return_value=mock_geolocator):
        result1 = await geocode("New York City")
        result2 = await geocode("New York City")
        # Also test normalization
        result3 = await geocode("  New York City  ")

    assert result1 == result2 == result3
    # Nominatim should only be called once (cached after first call)
    assert mock_geolocator.geocode.call_count == 1


@pytest.mark.asyncio
async def test_geocode_exception_handling():
    """Test that geocoding errors return None gracefully."""
    from geopy.exc import GeocoderTimedOut

    mock_geolocator = AsyncMock()
    mock_geolocator.geocode = AsyncMock(side_effect=GeocoderTimedOut("Network error"))
    mock_geolocator.__aenter__ = AsyncMock(return_value=mock_geolocator)
    mock_geolocator.__aexit__ = AsyncMock(return_value=False)

    with patch("weather_edge.weather.geocoding.Nominatim", return_value=mock_geolocator):
        result = await geocode("Phoenix, AZ")

    assert result is None
    # Should be cached as None
    assert "phoenix, az" in _cache
    assert _cache["phoenix, az"] is None
