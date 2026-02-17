"""Location name to lat/lon geocoding using geopy Nominatim."""

from __future__ import annotations

import logging
from collections import OrderedDict

from geopy.adapters import AioHTTPAdapter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable
from geopy.geocoders import Nominatim

from weather_edge.common.types import LatLon

logger = logging.getLogger(__name__)

# Weather station coordinates for Polymarket cities.
# These match the airport/NWS stations where markets resolve.
_STATION_COORDS: dict[str, LatLon] = {
    "atlanta":       (33.6407, -84.4277),   # KATL
    "atlanta, ga":   (33.6407, -84.4277),
    "new york":      (40.7769, -73.8740),   # KLGA
    "new york city":  (40.7769, -73.8740),
    "new york, ny":  (40.7769, -73.8740),
    "nyc":           (40.7769, -73.8740),
    "dallas":        (32.8471, -96.8517),   # KDAL
    "dallas, tx":    (32.8471, -96.8517),
    "miami":         (25.7959, -80.2870),   # KMIA
    "miami, fl":     (25.7959, -80.2870),
    "chicago":       (41.9742, -87.9073),   # KORD
    "chicago, il":   (41.9742, -87.9073),
    "seattle":       (47.4502, -122.3088),  # KSEA
    "seattle, wa":   (47.4502, -122.3088),
    "toronto":       (43.6777, -79.6248),   # CYYZ
    "toronto, on":   (43.6777, -79.6248),
    "london":        (51.5053, 0.0553),     # EGLC
    "london, uk":    (51.5053, 0.0553),
    "paris":         (49.0097, 2.5479),     # LFPG
    "paris, fr":     (49.0097, 2.5479),
    "buenos aires":  (-34.8222, -58.5358),  # SAEZ
    "buenos aires, ar": (-34.8222, -58.5358),
    "sao paulo":     (-23.4356, -46.4731),  # SBGR
    "são paulo":     (-23.4356, -46.4731),
    "sao paulo, br": (-23.4356, -46.4731),
    "são paulo, br": (-23.4356, -46.4731),
    "wellington":    (-41.3272, 174.8053),  # NZWN
    "wellington, nz": (-41.3272, 174.8053),
    "incheon":       (37.4602, 126.4407),   # RKSI
    "incheon, kr":   (37.4602, 126.4407),
    "seoul":         (37.4602, 126.4407),   # resolves at Incheon
    "seoul, kr":     (37.4602, 126.4407),
    "ankara":        (40.1283, 32.9958),    # LTAC
    "ankara, tr":    (40.1283, 32.9958),
}

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

    # Fast path: known Polymarket weather station
    station = _STATION_COORDS.get(normalized)
    if station is not None:
        return station

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
