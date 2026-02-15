"""Tests for hurricane forecast model (stub)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weather_edge.forecasting.hurricane import HurricaneModel
from weather_edge.markets.models import MarketParams, MarketType


@pytest.mark.asyncio
async def test_hurricane_climatology_peak_season(now):
    """September should have highest base rate."""
    params = MarketParams(
        market_type=MarketType.HURRICANE,
        location="Miami, FL",
        lat_lon=(25.76, -80.19),
        target_date=datetime(now.year, 9, 15, tzinfo=timezone.utc),
    )

    model = HurricaneModel()
    estimate = await model.estimate(params, gfs=None, ecmwf=None, noaa=None)

    assert estimate.probability > 0.10  # Peak season should be elevated
    assert "climatology" in estimate.sources_used
    assert "STUB" in estimate.details


@pytest.mark.asyncio
async def test_hurricane_climatology_off_season(now):
    """January should have very low base rate."""
    params = MarketParams(
        market_type=MarketType.HURRICANE,
        location="Miami, FL",
        lat_lon=(25.76, -80.19),
        target_date=datetime(now.year, 1, 15, tzinfo=timezone.utc),
    )

    model = HurricaneModel()
    estimate = await model.estimate(params, gfs=None, ecmwf=None, noaa=None)

    assert estimate.probability < 0.01  # Off-season should be very low
    assert "climatology" in estimate.sources_used


@pytest.mark.asyncio
async def test_hurricane_with_alert_boost(noaa_forecast_with_hurricane_alert, now):
    """Active hurricane alert should significantly boost probability."""
    params = MarketParams(
        market_type=MarketType.HURRICANE,
        location="Miami, FL",
        lat_lon=(25.76, -80.19),
        target_date=now + timedelta(days=1),
    )

    model = HurricaneModel()

    # Without alerts
    estimate_no_alert = await model.estimate(params, gfs=None, ecmwf=None, noaa=None)

    # With alerts
    estimate_with_alert = await model.estimate(
        params, gfs=None, ecmwf=None, noaa=noaa_forecast_with_hurricane_alert,
    )

    assert estimate_with_alert.probability > estimate_no_alert.probability
    assert "NOAA alerts" in estimate_with_alert.sources_used


@pytest.mark.asyncio
async def test_hurricane_low_confidence(hurricane_params):
    """Hurricane model should have low confidence since it's a stub."""
    model = HurricaneModel()
    estimate = await model.estimate(hurricane_params, gfs=None, ecmwf=None, noaa=None)

    # Stub model confidence should be < 0.5 (base confidence * 0.5)
    assert estimate.confidence < 0.5


@pytest.mark.asyncio
async def test_hurricane_probability_bounds(now):
    """Probability should always be in [0, 1]."""
    model = HurricaneModel()

    for month in range(1, 13):
        params = MarketParams(
            market_type=MarketType.HURRICANE,
            location="Miami, FL",
            target_date=datetime(now.year, month, 15, tzinfo=timezone.utc),
        )
        estimate = await model.estimate(params, gfs=None, ecmwf=None, noaa=None)
        assert 0 <= estimate.probability <= 1, f"Month {month}: prob={estimate.probability}"


@pytest.mark.asyncio
async def test_hurricane_extreme_alert_capped(now):
    """Even with extreme alert, probability shouldn't exceed 0.95."""
    from weather_edge.weather.models import NOAAAlert, NOAAForecast

    noaa = NOAAForecast(
        lat=25.76, lon=-80.19, office="MFL", grid_x=0, grid_y=0,
        alerts=[
            NOAAAlert(
                event="Hurricane Warning",
                headline="Category 5 Hurricane",
                severity="Extreme",
                certainty="Observed",
            ),
            NOAAAlert(
                event="Tropical Storm Warning",
                headline="Tropical Storm",
                severity="Severe",
                certainty="Likely",
            ),
        ],
    )

    params = MarketParams(
        market_type=MarketType.HURRICANE,
        location="Miami, FL",
        target_date=now + timedelta(hours=12),
    )

    model = HurricaneModel()
    estimate = await model.estimate(params, gfs=None, ecmwf=None, noaa=noaa)

    assert estimate.probability <= 0.95
