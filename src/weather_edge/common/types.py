"""Shared type aliases."""

from __future__ import annotations

from typing import TypeAlias

# Latitude/longitude pair
LatLon: TypeAlias = tuple[float, float]

# JSON-like dict
JsonDict: TypeAlias = dict[str, object]


def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (f - 32.0) * 5.0 / 9.0


def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9.0 / 5.0 + 32.0
