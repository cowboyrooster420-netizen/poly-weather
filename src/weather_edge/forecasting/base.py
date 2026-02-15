"""Forecast model protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from weather_edge.markets.models import MarketParams
from weather_edge.weather.models import EnsembleForecast, NOAAForecast


@dataclass
class ProbabilityEstimate:
    """Calibrated probability estimate from a forecast model.

    Attributes:
        probability: calibrated probability of the event occurring (0-1)
        raw_probability: uncalibrated probability before spread inflation
        confidence: model confidence in its estimate (0-1), affects Kelly sizing
        lead_time_hours: hours between forecast init and target time
        sources_used: which data sources contributed
        details: human-readable explanation of the estimate
    """

    probability: float
    raw_probability: float
    confidence: float
    lead_time_hours: float
    sources_used: list[str] = field(default_factory=list)
    details: str = ""


class ForecastModel(Protocol):
    """Protocol for forecast models dispatched by market type."""

    async def estimate(
        self,
        params: MarketParams,
        gfs: EnsembleForecast | None,
        ecmwf: EnsembleForecast | None,
        noaa: NOAAForecast | None,
    ) -> ProbabilityEstimate:
        """Produce a calibrated probability estimate for the market.

        Args:
            params: Parsed market parameters
            gfs: GFS ensemble forecast (31 members)
            ecmwf: ECMWF ensemble forecast (51 members)
            noaa: NOAA/NWS forecast (US only, may be None)

        Returns:
            ProbabilityEstimate with calibrated probability and confidence
        """
        ...
