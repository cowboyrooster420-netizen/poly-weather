"""Signal data models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Signal:
    """A trading signal with edge and Kelly sizing.

    Attributes:
        market_id: Polymarket market ID
        question: Market question text
        market_type: Type of weather market
        location: Location string
        model_prob: Model's calibrated probability estimate
        market_prob: Market-implied probability (YES price)
        edge: model_prob - market_prob (positive = underpriced YES)
        kelly_fraction: Recommended bet size as fraction of bankroll
        confidence: Model confidence score (0-1)
        direction: "YES" if edge > 0, "NO" if edge < 0
        lead_time_hours: Forecast lead time
        sources: Data sources used
        details: Human-readable model details
        timestamp: When the signal was generated
    """

    market_id: str
    question: str
    market_type: str
    location: str
    model_prob: float
    market_prob: float
    edge: float
    kelly_fraction: float
    confidence: float
    direction: str
    lead_time_hours: float
    sources: list[str]
    details: str
    timestamp: datetime
