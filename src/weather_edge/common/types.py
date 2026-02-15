"""Shared type aliases."""

from __future__ import annotations

from typing import TypeAlias

# Latitude/longitude pair
LatLon: TypeAlias = tuple[float, float]

# JSON-like dict
JsonDict: TypeAlias = dict[str, object]
