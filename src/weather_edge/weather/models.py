"""Weather data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray


@dataclass
class EnsembleForecast:
    """Ensemble forecast from Open-Meteo (GFS or ECMWF).

    Attributes:
        source: "gfs" or "ecmwf"
        lat: latitude of forecast point
        lon: longitude of forecast point
        times: array of forecast valid times
        temperature_2m: (n_times, n_members) temperature in Celsius
        precipitation: (n_times, n_members) precipitation in mm
        wind_speed_10m: (n_times, n_members) wind speed in km/h (if available)
    """

    source: str
    lat: float
    lon: float
    times: list[datetime]
    temperature_2m: NDArray[np.float64]
    precipitation: NDArray[np.float64]
    wind_speed_10m: NDArray[np.float64] | None = None

    @property
    def n_members(self) -> int:
        return self.temperature_2m.shape[1]

    @property
    def n_times(self) -> int:
        return len(self.times)


@dataclass
class NOAAForecast:
    """NOAA/NWS forecast data for a US location.

    Attributes:
        lat: latitude
        lon: longitude
        office: NWS office code (e.g. "OKX")
        grid_x: NWS grid X coordinate
        grid_y: NWS grid Y coordinate
        periods: list of forecast periods with temperature, precip, etc.
        alerts: list of active weather alerts
    """

    lat: float
    lon: float
    office: str
    grid_x: int
    grid_y: int
    periods: list[NOAAPeriod] = field(default_factory=list)
    alerts: list[NOAAAlert] = field(default_factory=list)


@dataclass
class NOAAPeriod:
    """A single NOAA forecast period."""

    start_time: datetime
    end_time: datetime
    temperature: float  # Fahrenheit
    temperature_unit: str
    wind_speed: str
    short_forecast: str
    detailed_forecast: str
    precipitation_probability: float | None = None


@dataclass
class NOAAAlert:
    """An active NOAA weather alert."""

    event: str
    headline: str
    severity: str
    certainty: str
    onset: datetime | None = None
    expires: datetime | None = None
    description: str = ""
