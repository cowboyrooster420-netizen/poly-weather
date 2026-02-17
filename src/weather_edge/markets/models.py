"""Market data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from weather_edge.common.types import LatLon


class MarketType(Enum):
    """Type of weather market."""

    TEMPERATURE = "temperature"
    PRECIPITATION = "precipitation"
    HURRICANE = "hurricane"
    UNKNOWN = "unknown"


class Comparison(Enum):
    """Threshold comparison direction."""

    ABOVE = "above"  # e.g., "Will temp exceed 100F?"
    BELOW = "below"  # e.g., "Will temp drop below 32F?"
    BETWEEN = "between"  # e.g., "Will temp be between 70-80F?"


@dataclass
class MarketParams:
    """Parsed parameters extracted from a market question.

    These define what the market is asking about in structured form.
    """

    market_type: MarketType
    location: str
    lat_lon: LatLon | None = None
    threshold: float | None = None  # e.g., 100.0 for "above 100F"
    threshold_upper: float | None = None  # for BETWEEN comparisons
    comparison: Comparison = Comparison.ABOVE
    unit: str = "F"  # F or C for temp; mm or in for precip
    target_date: datetime | None = None
    target_date_str: str = ""  # Original date string from market
    period_start: datetime | None = None  # Start of accumulation period (precip)
    period_end: datetime | None = None  # End of accumulation period (precip)
    daily_aggregation: str | None = None  # "max", "min", or None


@dataclass
class WeatherMarket:
    """A Polymarket market identified as weather-related.

    Combines raw market data with parsed parameters.
    """

    market_id: str
    condition_id: str
    question: str
    description: str
    outcome_yes_price: float  # Current YES price (0-1, proxy for market probability)
    outcome_no_price: float
    end_date: datetime | None = None
    volume: float = 0.0
    liquidity: float = 0.0
    params: MarketParams | None = None
    slug: str = ""
    active: bool = True

    @property
    def market_prob(self) -> float:
        """Market-implied probability (YES price), clamped to [0.001, 0.999]."""
        return max(0.001, min(0.999, self.outcome_yes_price))
