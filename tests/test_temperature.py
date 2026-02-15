"""Tests for temperature forecast model with synthetic data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from weather_edge.forecasting.temperature import TemperatureModel
from weather_edge.markets.models import Comparison, MarketParams, MarketType


@pytest.mark.asyncio
async def test_temperature_above_likely(ecmwf_forecast, gfs_forecast, temperature_params):
    """When ensemble mean is well above threshold, probability should be high."""
    # Ensemble mean ~35C, threshold 100F = 37.78C → moderate probability
    model = TemperatureModel()
    estimate = await model.estimate(temperature_params, gfs_forecast, ecmwf_forecast, noaa=None)

    assert 0 < estimate.probability < 1
    assert estimate.confidence > 0
    assert len(estimate.sources_used) >= 2


@pytest.mark.asyncio
async def test_temperature_above_unlikely(ecmwf_forecast, gfs_forecast, now):
    """When threshold is far above ensemble, probability should be very low."""
    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Phoenix, AZ",
        lat_lon=(33.45, -112.07),
        threshold=150.0,  # 150F = 65.5C, way above ensemble mean of ~35C
        comparison=Comparison.ABOVE,
        unit="F",
        target_date=now + timedelta(hours=24),
    )

    model = TemperatureModel()
    estimate = await model.estimate(params, gfs_forecast, ecmwf_forecast, noaa=None)

    assert estimate.probability < 0.01  # Essentially impossible


@pytest.mark.asyncio
async def test_temperature_below(ecmwf_forecast, gfs_forecast, now):
    """Test BELOW comparison direction."""
    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Phoenix, AZ",
        lat_lon=(33.45, -112.07),
        threshold=50.0,  # 50F = 10C, far below ensemble mean of ~35C
        comparison=Comparison.BELOW,
        unit="F",
        target_date=now + timedelta(hours=24),
    )

    model = TemperatureModel()
    estimate = await model.estimate(params, gfs_forecast, ecmwf_forecast, noaa=None)

    assert estimate.probability < 0.01  # Very unlikely


@pytest.mark.asyncio
async def test_temperature_celsius(now):
    """Test with Celsius threshold."""
    np.random.seed(42)
    times = [now + timedelta(hours=i) for i in range(48)]
    n = len(times)
    from weather_edge.weather.models import EnsembleForecast

    ecmwf = EnsembleForecast(
        source="ecmwf", lat=51.5, lon=-0.12,
        times=times,
        temperature_2m=np.random.normal(20, 2, (n, 51)),
        precipitation=np.zeros((n, 51)),
    )

    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="London, UK",
        lat_lon=(51.5, -0.12),
        threshold=25.0,  # 25C, above mean of 20C
        comparison=Comparison.ABOVE,
        unit="C",
        target_date=now + timedelta(hours=24),
    )

    model = TemperatureModel()
    estimate = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None)

    assert 0 < estimate.probability < 0.5  # Somewhat unlikely (2.5 sigma)


@pytest.mark.asyncio
async def test_temperature_no_ensemble_data(temperature_params):
    """Test graceful handling when no ensemble data available."""
    model = TemperatureModel()
    estimate = await model.estimate(temperature_params, gfs=None, ecmwf=None, noaa=None)

    assert estimate.probability == 0.5
    assert estimate.confidence == 0.0


@pytest.mark.asyncio
async def test_temperature_no_threshold(ecmwf_forecast, gfs_forecast, now):
    """Test handling when threshold is None."""
    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Phoenix, AZ",
        threshold=None,  # No threshold
    )

    model = TemperatureModel()
    estimate = await model.estimate(params, gfs_forecast, ecmwf_forecast, noaa=None)

    assert estimate.probability == 0.5
    assert estimate.confidence == 0.0


@pytest.mark.asyncio
async def test_temperature_with_noaa(ecmwf_forecast, gfs_forecast, noaa_forecast, temperature_params):
    """Test that NOAA data is included in sources."""
    model = TemperatureModel()
    estimate = await model.estimate(temperature_params, gfs_forecast, ecmwf_forecast, noaa_forecast)

    assert "NOAA/NWS" in estimate.sources_used


@pytest.mark.asyncio
async def test_temperature_probability_bounds(ecmwf_forecast, gfs_forecast, now):
    """Probability should always be clamped to (0.001, 0.999)."""
    # Extremely high threshold → prob near 0
    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Phoenix, AZ",
        lat_lon=(33.45, -112.07),
        threshold=200.0,
        comparison=Comparison.ABOVE,
        unit="F",
        target_date=now + timedelta(hours=24),
    )

    model = TemperatureModel()
    estimate = await model.estimate(params, gfs_forecast, ecmwf_forecast, noaa=None)

    assert estimate.probability >= 0.001
    assert estimate.probability <= 0.999


@pytest.mark.asyncio
async def test_temperature_ecmwf_weighted_higher(now):
    """ECMWF should have higher weight than GFS in blended result."""
    from weather_edge.weather.models import EnsembleForecast

    times = [now + timedelta(hours=i) for i in range(48)]
    n = len(times)

    # ECMWF: high temps (mean 40C)
    ecmwf = EnsembleForecast(
        source="ecmwf", lat=33.45, lon=-112.07,
        times=times,
        temperature_2m=np.full((n, 51), 40.0),
        precipitation=np.zeros((n, 51)),
    )

    # GFS: low temps (mean 20C)
    gfs = EnsembleForecast(
        source="gfs", lat=33.45, lon=-112.07,
        times=times,
        temperature_2m=np.full((n, 31), 20.0),
        precipitation=np.zeros((n, 31)),
    )

    # Threshold: 30C — ECMWF says definitely above, GFS says definitely below
    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Phoenix, AZ",
        lat_lon=(33.45, -112.07),
        threshold=86.0,  # 30C in F
        comparison=Comparison.ABOVE,
        unit="F",
        target_date=now + timedelta(hours=24),
    )

    model = TemperatureModel()
    estimate = await model.estimate(params, gfs, ecmwf, noaa=None)

    # Blended probability should be > 0.5 since ECMWF (weight 0.6) says ~1.0
    assert estimate.probability > 0.5


@pytest.mark.asyncio
async def test_temperature_single_source_reduced_confidence(ecmwf_forecast, now):
    """Single source should reduce confidence."""
    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Phoenix, AZ",
        lat_lon=(33.45, -112.07),
        threshold=100.0,
        comparison=Comparison.ABOVE,
        unit="F",
        target_date=now + timedelta(hours=24),
    )

    model = TemperatureModel()

    # With both sources
    from weather_edge.weather.models import EnsembleForecast
    np.random.seed(43)
    times = ecmwf_forecast.times
    gfs = EnsembleForecast(
        source="gfs", lat=33.45, lon=-112.07,
        times=times,
        temperature_2m=np.random.normal(34, 3.5, (len(times), 31)),
        precipitation=np.zeros((len(times), 31)),
    )

    est_both = await model.estimate(params, gfs, ecmwf_forecast, noaa=None)
    est_single = await model.estimate(params, gfs=None, ecmwf=ecmwf_forecast, noaa=None)

    assert est_single.confidence < est_both.confidence
