"""MarketType → ForecastModel dispatch registry."""

from __future__ import annotations

from weather_edge.forecasting.base import ForecastModel
from weather_edge.forecasting.hurricane import HurricaneModel
from weather_edge.forecasting.precipitation import PrecipitationModel
from weather_edge.forecasting.temperature import TemperatureModel
from weather_edge.markets.models import MarketType

_REGISTRY: dict[MarketType, ForecastModel] = {
    MarketType.TEMPERATURE: TemperatureModel(),
    MarketType.PRECIPITATION: PrecipitationModel(),
    MarketType.HURRICANE: HurricaneModel(),
}


def get_model(market_type: MarketType) -> ForecastModel | None:
    """Get the forecast model for a given market type.

    Returns None for UNKNOWN market types.
    """
    return _REGISTRY.get(market_type)
