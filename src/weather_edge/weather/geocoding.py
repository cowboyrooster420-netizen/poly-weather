"""Location name to lat/lon geocoding using geopy Nominatim."""

from __future__ import annotations

import logging
from collections import OrderedDict

from geopy.adapters import AioHTTPAdapter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable
from geopy.geocoders import Nominatim

from weather_edge.common.types import LatLon

logger = logging.getLogger(__name__)

# Weather Underground PWS coordinates for Polymarket cities.
# Markets resolve on these specific personal weather stations, NOT airport ASOS.
_STATION_COORDS: dict[str, LatLon] = {
    "atlanta":       (33.661, -84.399),     # KGAHAPEV1 (Hapeville)
    "atlanta, ga":   (33.661, -84.399),
    "new york":      (40.764, -73.835),     # KNYNEWYO1974 (Queens)
    "new york city":  (40.764, -73.835),
    "new york, ny":  (40.764, -73.835),
    "nyc":           (40.764, -73.835),
    "dallas":        (32.829, -96.878),     # KTXDALLA1276
    "dallas, tx":    (32.829, -96.878),
    "miami":         (25.857, -80.265),     # KFLHIALE117 (Hialeah)
    "miami, fl":     (25.857, -80.265),
    "chicago":       (41.964, -87.946),     # KILBENSE15 (Bensenville)
    "chicago, il":   (41.964, -87.946),
    "seattle":       (47.446, -122.276),    # KWASEATA17 (SeaTac)
    "seattle, wa":   (47.446, -122.276),
    "toronto":       (43.733, -79.637),     # IMISSI113 (Mississauga)
    "toronto, on":   (43.733, -79.637),
    "london":        (51.514, 0.037),       # ILONDO288 (Greenwich)
    "london, uk":    (51.514, 0.037),
    "paris":         (48.974, 2.632),       # IMITRY1 (Mitry-Mory)
    "paris, fr":     (48.974, 2.632),
    "buenos aires":  (-34.817, -58.467),    # IMONTEGR27 (Monte Grande)
    "buenos aires, ar": (-34.817, -58.467),
    "sao paulo":     (-23.448, -46.526),    # IGUARU12 (Guarulhos)
    "são paulo":     (-23.448, -46.526),
    "sao paulo, br": (-23.448, -46.526),
    "são paulo, br": (-23.448, -46.526),
    "wellington":    (-41.325, 174.792),    # IWGNLYAL3 (Lyall Bay)
    "wellington, nz": (-41.325, 174.792),
    "incheon":       (37.482, 126.523),     # IINCHE10
    "incheon, kr":   (37.482, 126.523),
    "seoul":         (37.482, 126.523),     # resolves at Incheon
    "seoul, kr":     (37.482, 126.523),
    "ankara":        (39.931, 32.899),      # IANKAR46
    "ankara, tr":    (39.931, 32.899),
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
