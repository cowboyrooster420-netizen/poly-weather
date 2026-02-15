"""Tests for precipitation forecast model."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from weather_edge.forecasting.precipitation import PrecipitationModel
from weather_edge.markets.models import Comparison, MarketParams, MarketType
from weather_edge.weather.models import EnsembleForecast


@pytest.fixture
def heavy_rain_ecmwf(now):
    """ECMWF ensemble with heavy rainfall."""
    np.random.seed(42)
    times = [now + timedelta(hours=i) for i in range(48)]
    n = len(times)
    return EnsembleForecast(
        source="ecmwf", lat=29.76, lon=-95.37,
        times=times,
        temperature_2m=np.random.normal(25, 3, (n, 51)),
        # Heavy rain: exponential dist, mean ~10mm
        precipitation=np.random.exponential(10, (n, 51)),
    )


@pytest.fixture
def dry_ecmwf(now):
    """ECMWF ensemble with minimal rainfall."""
    np.random.seed(42)
    times = [now + timedelta(hours=i) for i in range(48)]
    n = len(times)
    return EnsembleForecast(
        source="ecmwf", lat=29.76, lon=-95.37,
        times=times,
        temperature_2m=np.random.normal(25, 3, (n, 51)),
        # Almost no rain: mostly zeros
        precipitation=np.maximum(0, np.random.normal(-2, 0.5, (n, 51))),
    )


@pytest.mark.asyncio
async def test_precipitation_above_heavy_rain(heavy_rain_ecmwf, now):
    """Heavy rain ensemble → higher probability of exceeding threshold."""
    params = MarketParams(
        market_type=MarketType.PRECIPITATION,
        location="Houston, TX",
        lat_lon=(29.76, -95.37),
        threshold=0.5,  # 0.5 inches = 12.7mm, moderate threshold
        comparison=Comparison.ABOVE,
        unit="in",
        target_date=now + timedelta(hours=24),
    )

    model = PrecipitationModel()
    estimate = await model.estimate(params, gfs=None, ecmwf=heavy_rain_ecmwf, noaa=None)

    assert estimate.probability > 0.15  # Should be meaningful given heavy rain


@pytest.mark.asyncio
async def test_precipitation_above_dry(dry_ecmwf, now):
    """Dry ensemble → very low probability of exceeding threshold."""
    params = MarketParams(
        market_type=MarketType.PRECIPITATION,
        location="Houston, TX",
        lat_lon=(29.76, -95.37),
        threshold=1.0,  # 1 inch
        comparison=Comparison.ABOVE,
        unit="in",
        target_date=now + timedelta(hours=24),
    )

    model = PrecipitationModel()
    estimate = await model.estimate(params, gfs=None, ecmwf=dry_ecmwf, noaa=None)

    assert estimate.probability < 0.1


@pytest.mark.asyncio
async def test_precipitation_mm_threshold(heavy_rain_ecmwf, now):
    """Test with mm threshold (no conversion needed)."""
    params = MarketParams(
        market_type=MarketType.PRECIPITATION,
        location="Houston, TX",
        lat_lon=(29.76, -95.37),
        threshold=5.0,  # 5mm
        comparison=Comparison.ABOVE,
        unit="mm",
        target_date=now + timedelta(hours=24),
    )

    model = PrecipitationModel()
    estimate = await model.estimate(params, gfs=None, ecmwf=heavy_rain_ecmwf, noaa=None)

    assert 0 < estimate.probability < 1


@pytest.mark.asyncio
async def test_precipitation_no_data(precipitation_params):
    """Test with no ensemble data."""
    model = PrecipitationModel()
    estimate = await model.estimate(precipitation_params, gfs=None, ecmwf=None, noaa=None)

    assert estimate.probability == 0.5
    assert estimate.confidence == 0.0


@pytest.mark.asyncio
async def test_precipitation_no_threshold(now):
    """Test handling when threshold is None."""
    params = MarketParams(
        market_type=MarketType.PRECIPITATION,
        location="Houston, TX",
        threshold=None,
    )

    model = PrecipitationModel()
    estimate = await model.estimate(params, gfs=None, ecmwf=None, noaa=None)

    assert estimate.probability == 0.5
    assert estimate.confidence == 0.0


@pytest.mark.asyncio
async def test_precipitation_probability_bounds(heavy_rain_ecmwf, now):
    """Probability should be clamped to (0.001, 0.999)."""
    model = PrecipitationModel()

    # Very low threshold — almost certain
    params_low = MarketParams(
        market_type=MarketType.PRECIPITATION,
        location="Houston, TX",
        lat_lon=(29.76, -95.37),
        threshold=0.01,  # 0.01mm, very small
        comparison=Comparison.ABOVE,
        unit="mm",
        target_date=now + timedelta(hours=24),
    )
    est = await model.estimate(params_low, gfs=None, ecmwf=heavy_rain_ecmwf, noaa=None)
    assert est.probability <= 0.999

    # Very high threshold — almost impossible
    params_high = MarketParams(
        market_type=MarketType.PRECIPITATION,
        location="Houston, TX",
        lat_lon=(29.76, -95.37),
        threshold=1000.0,  # 1000mm, absurd
        comparison=Comparison.ABOVE,
        unit="mm",
        target_date=now + timedelta(hours=24),
    )
    est2 = await model.estimate(params_high, gfs=None, ecmwf=heavy_rain_ecmwf, noaa=None)
    assert est2.probability >= 0.001


@pytest.mark.asyncio
async def test_precipitation_lower_confidence_than_temperature(
    ecmwf_forecast, gfs_forecast, temperature_params, precipitation_params, now
):
    """Precipitation model should have lower inherent confidence than temperature."""
    from weather_edge.forecasting.temperature import TemperatureModel

    temp_model = TemperatureModel()
    precip_model = PrecipitationModel()

    temp_est = await temp_model.estimate(temperature_params, gfs_forecast, ecmwf_forecast, noaa=None)

    # Use matching ensemble for precipitation
    np.random.seed(42)
    times = [now + timedelta(hours=i) for i in range(48)]
    n = len(times)
    ecmwf_precip = EnsembleForecast(
        source="ecmwf", lat=29.76, lon=-95.37, times=times,
        temperature_2m=np.random.normal(25, 3, (n, 51)),
        precipitation=np.random.exponential(5, (n, 51)),
    )
    gfs_precip = EnsembleForecast(
        source="gfs", lat=29.76, lon=-95.37, times=times,
        temperature_2m=np.random.normal(24, 3, (n, 31)),
        precipitation=np.random.exponential(5, (n, 31)),
    )

    precip_est = await precip_model.estimate(precipitation_params, gfs_precip, ecmwf_precip, noaa=None)

    # Precip has 0.85 multiplier on base confidence
    assert precip_est.confidence < temp_est.confidence


@pytest.mark.asyncio
async def test_precipitation_with_noaa(heavy_rain_ecmwf, noaa_forecast, now):
    """Test that NOAA precip probability is included in details."""
    params = MarketParams(
        market_type=MarketType.PRECIPITATION,
        location="Phoenix, AZ",
        lat_lon=(33.45, -112.07),
        threshold=1.0,
        comparison=Comparison.ABOVE,
        unit="in",
        target_date=now + timedelta(hours=1),  # Close to first NOAA period
    )

    model = PrecipitationModel()
    estimate = await model.estimate(params, gfs=None, ecmwf=heavy_rain_ecmwf, noaa=noaa_forecast)

    assert "NOAA/NWS" in estimate.sources_used
