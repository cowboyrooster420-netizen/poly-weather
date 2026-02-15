"""Location name to lat/lon geocoding using geopy Nominatim."""

from __future__ import annotations

import logging
from collections import OrderedDict

from geopy.adapters import AioHTTPAdapter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable
from geopy.geocoders import Nominatim

from weather_edge.common.types import LatLon

logger = logging.getLogger(__name__)

# Bounded LRU cache for geocoding results (oldest evicted first)
_MAX_CACHE_SIZE = 256
_cache: OrderedDict[str, LatLon | None] = OrderedDict()


def _cache_put(key: str, value: LatLon | None) -> None:
    """Insert into bounded cache, evicting oldest if full."""
    _cache[key] = value
    _cache.move_to_end(key)
    if len(_cache) > _MAX_CACHE_SIZE:
        _cache.popitem(last=False)


async def geocode(location: str) -> LatLon | None:
    """Convert a location name to (lat, lon).

    Uses Nominatim (free, no API key). Results are cached in memory
    with a max size of 256 entries (LRU eviction).
    Returns None if the location cannot be resolved.
    """
    normalized = location.strip().lower()
    if normalized in _cache:
        _cache.move_to_end(normalized)
        return _cache[normalized]

    try:
        async with Nominatim(
            user_agent="weather-edge",
            adapter_factory=AioHTTPAdapter,
        ) as geolocator:
            result = await geolocator.geocode(location)
            if result is None:
                logger.debug("Geocoding returned no results for %r", location)
                _cache_put(normalized, None)
                return None
            latlon: LatLon = (result.latitude, result.longitude)
            _cache_put(normalized, latlon)
            return latlon
    except (GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable) as exc:
        logger.warning("Geocoding service error for %r: %s", location, exc)
        _cache_put(normalized, None)
        return None
    except (ValueError, TypeError) as exc:
        logger.warning("Geocoding parse error for %r: %s", location, exc)
        _cache_put(normalized, None)
        return None
