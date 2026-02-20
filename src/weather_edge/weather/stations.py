"""Weather Underground personal weather station registry.

Maps WU station IDs used by Polymarket weather markets to their
metadata (location, coordinates, timezone).

IMPORTANT: Station IDs and coordinates MUST match the authoritative
values in geocoding.py's _STATION_COORDS dict. Those are the stations
that Polymarket markets actually resolve on.
"""

from __future__ import annotations

from dataclasses import dataclass

from weather_edge.common.types import LatLon


@dataclass(frozen=True)
class Station:
    """A Weather Underground personal weather station."""

    wu_id: str        # e.g. "KGAHAPEV1"
    city: str         # e.g. "atlanta"
    lat_lon: LatLon   # (lat, lon)
    timezone: str     # IANA timezone, e.g. "America/New_York"


# All 14 WU stations referenced by Polymarket weather markets.
# Coordinates and station IDs sourced from geocoding.py _STATION_COORDS.
STATIONS: dict[str, Station] = {
    "KGAHAPEV1": Station(
        wu_id="KGAHAPEV1", city="atlanta",
        lat_lon=(33.661, -84.399), timezone="America/New_York",
    ),
    "KNYNEWYO1974": Station(
        wu_id="KNYNEWYO1974", city="new york",
        lat_lon=(40.764, -73.835), timezone="America/New_York",
    ),
    "KTXDALLA1276": Station(
        wu_id="KTXDALLA1276", city="dallas",
        lat_lon=(32.829, -96.878), timezone="America/Chicago",
    ),
    "KFLHIALE117": Station(
        wu_id="KFLHIALE117", city="miami",
        lat_lon=(25.857, -80.265), timezone="America/New_York",
    ),
    "KILBENSE15": Station(
        wu_id="KILBENSE15", city="chicago",
        lat_lon=(41.964, -87.946), timezone="America/Chicago",
    ),
    "KWASEATA17": Station(
        wu_id="KWASEATA17", city="seattle",
        lat_lon=(47.446, -122.276), timezone="America/Los_Angeles",
    ),
    "IMISSI113": Station(
        wu_id="IMISSI113", city="toronto",
        lat_lon=(43.733, -79.637), timezone="America/Toronto",
    ),
    "ILONDO288": Station(
        wu_id="ILONDO288", city="london",
        lat_lon=(51.514, 0.037), timezone="Europe/London",
    ),
    "IMITRY1": Station(
        wu_id="IMITRY1", city="paris",
        lat_lon=(48.974, 2.632), timezone="Europe/Paris",
    ),
    "IMONTEGR27": Station(
        wu_id="IMONTEGR27", city="buenos aires",
        lat_lon=(-34.817, -58.467), timezone="America/Argentina/Buenos_Aires",
    ),
    "IGUARU12": Station(
        wu_id="IGUARU12", city="sao paulo",
        lat_lon=(-23.448, -46.526), timezone="America/Sao_Paulo",
    ),
    "IWGNLYAL3": Station(
        wu_id="IWGNLYAL3", city="wellington",
        lat_lon=(-41.325, 174.792), timezone="Pacific/Auckland",
    ),
    "IINCHE10": Station(
        wu_id="IINCHE10", city="incheon",
        lat_lon=(37.482, 126.523), timezone="Asia/Seoul",
    ),
    "IANKAR46": Station(
        wu_id="IANKAR46", city="ankara",
        lat_lon=(39.931, 32.899), timezone="Europe/Istanbul",
    ),
}

# City name aliases for lookup — maps alternative names to canonical city.
_CITY_ALIASES: dict[str, str] = {
    "new york city": "new york",
    "nyc": "new york",
    "new york, ny": "new york",
    "atlanta, ga": "atlanta",
    "dallas, tx": "dallas",
    "miami, fl": "miami",
    "chicago, il": "chicago",
    "seattle, wa": "seattle",
    "toronto, on": "toronto",
    "london, uk": "london",
    "paris, fr": "paris",
    "buenos aires, ar": "buenos aires",
    "sao paulo, br": "sao paulo",
    "são paulo": "sao paulo",
    "são paulo, br": "sao paulo",
    "wellington, nz": "wellington",
    "incheon, kr": "incheon",
    "seoul": "incheon",
    "seoul, kr": "incheon",
    "ankara, tr": "ankara",
    "saint louis": "st. louis",
    "dc": "washington",
    "washington dc": "washington",
}

# Build reverse index: city name → Station (for O(1) exact lookup).
_CITY_INDEX: dict[str, Station] = {}
for _station in STATIONS.values():
    _CITY_INDEX[_station.city] = _station
for _alias, _canonical in _CITY_ALIASES.items():
    if _canonical in _CITY_INDEX:
        _CITY_INDEX[_alias] = _CITY_INDEX[_canonical]


def station_for_location(city: str) -> Station | None:
    """Find a station by city name (case-insensitive).

    Supports exact match, alias resolution, and input-prefix matching
    (e.g. "new york city" matches "new york"). Does NOT allow the
    station city to prefix-match short inputs (e.g. "d" won't match
    "dallas").

    Examples:
        station_for_location("atlanta") → KGAHAPEV1
        station_for_location("Atlanta, GA") → KGAHAPEV1
        station_for_location("new york city") → KNYNEWYO1974
        station_for_location("Seoul, KR") → IINCHE10
    """
    normalized = city.lower().strip()
    # Strip state/country suffixes like ", GA" or ", TX"
    if "," in normalized:
        normalized = normalized.split(",")[0].strip()

    # Exact match (including aliases)
    match = _CITY_INDEX.get(normalized)
    if match is not None:
        return match

    # Input-prefix match: "new york city" starts with "new york".
    # Only allow the INPUT to be longer than the station city, not shorter.
    # This prevents "d" from matching "dallas".
    for station_city, station in _CITY_INDEX.items():
        if normalized.startswith(station_city) and len(normalized) > len(station_city):
            return station

    return None
