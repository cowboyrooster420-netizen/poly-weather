"""Tests for temperature forecast model with synthetic data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from weather_edge.forecasting.temperature import TemperatureModel, _compute_dynamic_weights, _hrrr_blend_weight
from weather_edge.markets.models import Comparison, MarketParams, MarketType
from weather_edge.weather.models import EnsembleForecast, HRRRForecast


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


@pytest.mark.asyncio
async def test_temperature_between(now):
    """BETWEEN comparison: P(45F < T < 50F) should be meaningful."""
    np.random.seed(42)
    times = [now + timedelta(hours=i) for i in range(48)]
    n = len(times)
    # Mean ~7C (44.6F), std ~2C — so 45-50F (7.2-10C) captures the upper half
    ecmwf = EnsembleForecast(
        source="ecmwf", lat=51.5, lon=-0.12,
        times=times,
        temperature_2m=np.random.normal(7, 2, (n, 51)),
        precipitation=np.zeros((n, 51)),
    )

    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="London, UK",
        lat_lon=(51.5, -0.12),
        threshold=45.0,      # lower bound in F
        threshold_upper=50.0, # upper bound in F
        comparison=Comparison.BETWEEN,
        unit="F",
        target_date=now + timedelta(hours=24),
    )

    model = TemperatureModel()
    estimate = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None)

    # Should give a probability strictly between 0 and 1
    assert 0.01 < estimate.probability < 0.99


@pytest.mark.asyncio
async def test_temperature_between_narrow_band(now):
    """Very narrow band far from mean should give low probability."""
    np.random.seed(42)
    times = [now + timedelta(hours=i) for i in range(48)]
    n = len(times)
    # Mean ~35C, asking about 0-5F (-17 to -15C) — way below
    ecmwf = EnsembleForecast(
        source="ecmwf", lat=33.45, lon=-112.07,
        times=times,
        temperature_2m=np.random.normal(35, 3, (n, 51)),
        precipitation=np.zeros((n, 51)),
    )

    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Phoenix, AZ",
        lat_lon=(33.45, -112.07),
        threshold=0.0,
        threshold_upper=5.0,
        comparison=Comparison.BETWEEN,
        unit="F",
        target_date=now + timedelta(hours=24),
    )

    model = TemperatureModel()
    estimate = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None)

    assert estimate.probability < 0.01


@pytest.mark.asyncio
async def test_daily_max_aggregation(now):
    """daily_aggregation='max' should give higher temps than a single hour."""
    np.random.seed(42)
    # 24 hours of data on the target day; temperature rises then falls
    times = [now + timedelta(hours=i) for i in range(24)]
    n = len(times)
    n_members = 31

    # Each hour has mean = 20 + 5*sin(hour * pi/12) so peak ~25C at hour 6
    base = np.array([20 + 5 * np.sin(i * np.pi / 12) for i in range(n)])
    temp_2m = np.random.normal(base[:, None], 1.5, (n, n_members))

    ecmwf = EnsembleForecast(
        source="ecmwf", lat=33.45, lon=-112.07,
        times=times,
        temperature_2m=temp_2m,
        precipitation=np.zeros((n, n_members)),
    )

    # Threshold that's above the mean of most hours but below the daily max
    threshold_c = 24.0  # ~75.2F

    # Single-hour estimate (use hour 0, mean ~20C → high threshold is unlikely)
    params_single = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Phoenix, AZ",
        lat_lon=(33.45, -112.07),
        threshold=threshold_c,
        comparison=Comparison.ABOVE,
        unit="C",
        target_date=times[0],
        daily_aggregation=None,
    )

    # Daily max estimate (takes max across all 24 hours per member → ~25C)
    params_daily = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Phoenix, AZ",
        lat_lon=(33.45, -112.07),
        threshold=threshold_c,
        comparison=Comparison.ABOVE,
        unit="C",
        target_date=times[0],
        daily_aggregation="max",
    )

    model = TemperatureModel()
    est_single = await model.estimate(params_single, gfs=None, ecmwf=ecmwf, noaa=None)
    est_daily = await model.estimate(params_daily, gfs=None, ecmwf=ecmwf, noaa=None)

    # Daily max should give a higher probability than the single hour
    assert est_daily.probability > est_single.probability


@pytest.mark.asyncio
async def test_daily_min_aggregation(now):
    """daily_aggregation='min' should give lower temps than a single hour."""
    np.random.seed(42)
    times = [now + timedelta(hours=i) for i in range(24)]
    n = len(times)
    n_members = 31

    # Temperature varies: mean = 10 + 5*sin(hour * pi/12), trough ~5C
    base = np.array([10 + 5 * np.sin(i * np.pi / 12) for i in range(n)])
    temp_2m = np.random.normal(base[:, None], 1.5, (n, n_members))

    ecmwf = EnsembleForecast(
        source="ecmwf", lat=51.5, lon=-0.12,
        times=times,
        temperature_2m=temp_2m,
        precipitation=np.zeros((n, n_members)),
    )

    # Threshold at 7C — below mid-day but above minimum
    threshold_c = 7.0

    # Single hour at hour 6 where mean ~15C → prob of BELOW 7C is low
    params_single = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="London, UK",
        lat_lon=(51.5, -0.12),
        threshold=threshold_c,
        comparison=Comparison.BELOW,
        unit="C",
        target_date=times[6],
        daily_aggregation=None,
    )

    # Daily min → picks minimum across all hours per member, closer to 5C
    params_daily = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="London, UK",
        lat_lon=(51.5, -0.12),
        threshold=threshold_c,
        comparison=Comparison.BELOW,
        unit="C",
        target_date=times[6],
        daily_aggregation="min",
    )

    model = TemperatureModel()
    est_single = await model.estimate(params_single, gfs=None, ecmwf=ecmwf, noaa=None)
    est_daily = await model.estimate(params_daily, gfs=None, ecmwf=ecmwf, noaa=None)

    # Daily min should give a higher probability of being BELOW the threshold
    assert est_daily.probability > est_single.probability


@pytest.mark.asyncio
async def test_single_timestep_unchanged(ecmwf_forecast, gfs_forecast, now):
    """None aggregation preserves existing single-timestep behavior."""
    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Phoenix, AZ",
        lat_lon=(33.45, -112.07),
        threshold=100.0,
        comparison=Comparison.ABOVE,
        unit="F",
        target_date=now + timedelta(hours=24),
        daily_aggregation=None,
    )

    model = TemperatureModel()
    estimate = await model.estimate(params, gfs_forecast, ecmwf_forecast, noaa=None)

    # Should behave exactly like the original — produces a valid result
    assert 0 < estimate.probability < 1
    assert estimate.confidence > 0
    assert len(estimate.sources_used) >= 2


class TestDynamicWeighting:
    def test_reduces_outlier_drag(self):
        """4°C gap → outlier model should be downweighted."""
        # ECMWF mean=20, GFS mean=16 → 4°C gap > 2.5 threshold
        models = [(20.0, 0.6), (16.0, 0.4)]
        weights = _compute_dynamic_weights(models, disagreement_threshold=2.5)
        # ECMWF (closer to weighted mean) should get more weight than base 0.6
        assert weights[0] > 0.6
        # GFS (outlier) should get less weight than base 0.4
        assert weights[1] < 0.4
        # Still sums to 1
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_no_change_when_models_agree(self):
        """<2.5°C gap → base weights returned unchanged."""
        models = [(20.0, 0.6), (21.0, 0.4)]
        weights = _compute_dynamic_weights(models, disagreement_threshold=2.5)
        assert abs(weights[0] - 0.6) < 1e-9
        assert abs(weights[1] - 0.4) < 1e-9

    def test_min_weight_floor(self):
        """Huge gap → outlier still gets at least min_ratio of its base weight."""
        # 20°C gap — extreme disagreement
        models = [(30.0, 0.6), (10.0, 0.4)]
        weights = _compute_dynamic_weights(
            models, disagreement_threshold=2.5, min_ratio=0.15,
        )
        # Outlier (GFS at 10°C) should still have meaningful weight
        assert weights[1] >= 0.15 * 0.4 / (0.6 + 0.15 * 0.4) - 1e-9
        # But should be much less than base
        assert weights[1] < 0.4
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_single_model_unchanged(self):
        """Single model returns its weight unchanged."""
        weights = _compute_dynamic_weights([(20.0, 1.0)])
        assert weights == [1.0]

    @pytest.mark.asyncio
    async def test_dynamic_weighting_end_to_end(self, now):
        """With large model disagreement, blend should be closer to ECMWF."""
        times = [now + timedelta(hours=i) for i in range(48)]
        n = len(times)

        # ECMWF: mean ~20°C (near threshold)
        ecmwf = EnsembleForecast(
            source="ecmwf", lat=33.77, lon=-84.39,
            times=times,
            temperature_2m=np.random.RandomState(42).normal(20.0, 0.9, (n, 51)),
            precipitation=np.zeros((n, 51)),
        )

        # GFS: mean ~17°C (well below threshold — the outlier)
        gfs = EnsembleForecast(
            source="gfs", lat=33.77, lon=-84.39,
            times=times,
            temperature_2m=np.random.RandomState(43).normal(17.0, 0.9, (n, 31)),
            precipitation=np.zeros((n, 31)),
        )

        params = MarketParams(
            market_type=MarketType.TEMPERATURE,
            location="Atlanta, GA",
            lat_lon=(33.77, -84.39),
            threshold=20.0,
            comparison=Comparison.ABOVE,
            unit="C",
            target_date=now + timedelta(hours=48),
        )

        model = TemperatureModel()
        estimate = await model.estimate(params, gfs, ecmwf, noaa=None)

        # With dynamic weighting, result should be higher than naive 60/40 blend
        # because GFS (the outlier) gets downweighted
        assert estimate.probability > 0.25


# --- HRRR bias correction tests ---


@pytest.mark.asyncio
async def test_hrrr_bias_correction_short_lead(now):
    """At 8h lead time (50% weight), ensemble mean should shift toward HRRR."""
    np.random.seed(42)
    times = [now + timedelta(hours=i) for i in range(48)]
    n = len(times)

    # Ensemble mean ~20C
    ecmwf = EnsembleForecast(
        source="ecmwf", lat=33.45, lon=-112.07,
        times=times,
        temperature_2m=np.random.normal(20, 2, (n, 51)),
        precipitation=np.zeros((n, 51)),
    )

    # HRRR says 25C — 5C warmer than ensemble mean
    hrrr = HRRRForecast(
        lat=33.45, lon=-112.07,
        times=times,
        temperature_2m=np.full(n, 25.0),
    )

    # Threshold at 22C — ensemble mean (20C) is below, HRRR (25C) well above
    target_time = now + timedelta(hours=8)
    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Phoenix, AZ",
        lat_lon=(33.45, -112.07),
        threshold=22.0,
        comparison=Comparison.ABOVE,
        unit="C",
        target_date=target_time,
    )

    model = TemperatureModel()

    # Without HRRR
    est_no_hrrr = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None)
    # With HRRR (8h lead → 50% weight → mean shifts +2.5C toward 25C)
    est_with_hrrr = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None, hrrr=hrrr)

    # HRRR shifts mean upward → probability of exceeding 22C should increase
    assert est_with_hrrr.probability > est_no_hrrr.probability
    assert "HRRR" in est_with_hrrr.sources_used
    assert "HRRR correction" in est_with_hrrr.details


@pytest.mark.asyncio
async def test_hrrr_ignored_beyond_18h(now):
    """At 24h lead time, HRRR should have no effect."""
    np.random.seed(42)
    times = [now + timedelta(hours=i) for i in range(48)]
    n = len(times)

    ecmwf = EnsembleForecast(
        source="ecmwf", lat=33.45, lon=-112.07,
        times=times,
        temperature_2m=np.random.normal(20, 2, (n, 51)),
        precipitation=np.zeros((n, 51)),
    )

    # HRRR says something very different
    hrrr = HRRRForecast(
        lat=33.45, lon=-112.07,
        times=times,
        temperature_2m=np.full(n, 30.0),
    )

    target_time = now + timedelta(hours=24)
    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Phoenix, AZ",
        lat_lon=(33.45, -112.07),
        threshold=22.0,
        comparison=Comparison.ABOVE,
        unit="C",
        target_date=target_time,
    )

    model = TemperatureModel()
    est_no_hrrr = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None)
    est_with_hrrr = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None, hrrr=hrrr)

    # Should be identical — HRRR ignored beyond 18h
    assert abs(est_with_hrrr.probability - est_no_hrrr.probability) < 1e-9
    assert "HRRR" not in est_with_hrrr.sources_used


@pytest.mark.asyncio
async def test_hrrr_none_graceful(ecmwf_forecast, gfs_forecast, temperature_params):
    """hrrr=None should not crash — same behavior as before."""
    model = TemperatureModel()
    estimate = await model.estimate(
        temperature_params, gfs_forecast, ecmwf_forecast, noaa=None, hrrr=None,
    )

    assert 0 < estimate.probability < 1
    assert estimate.confidence > 0
    assert "HRRR" not in estimate.sources_used


class TestHRRRBlendWeight:
    """Unit tests for _hrrr_blend_weight boundary conditions."""

    def test_under_6h(self):
        assert _hrrr_blend_weight(0) == 0.75
        assert _hrrr_blend_weight(3) == 0.75
        assert _hrrr_blend_weight(5.99) == 0.75

    def test_6_to_12h(self):
        assert _hrrr_blend_weight(6.0) == 0.50
        assert _hrrr_blend_weight(9) == 0.50
        assert _hrrr_blend_weight(11.99) == 0.50

    def test_12_to_18h(self):
        assert _hrrr_blend_weight(12.0) == 0.25
        assert _hrrr_blend_weight(15) == 0.25
        assert _hrrr_blend_weight(17.99) == 0.25

    def test_beyond_18h(self):
        assert _hrrr_blend_weight(18.0) == 0.0
        assert _hrrr_blend_weight(24) == 0.0
        assert _hrrr_blend_weight(100) == 0.0


@pytest.mark.asyncio
async def test_hrrr_nan_temperature_graceful(now):
    """HRRR with NaN temperatures should be silently skipped."""
    np.random.seed(42)
    times = [now + timedelta(hours=i) for i in range(48)]
    n = len(times)

    ecmwf = EnsembleForecast(
        source="ecmwf", lat=33.45, lon=-112.07,
        times=times,
        temperature_2m=np.random.normal(20, 2, (n, 51)),
        precipitation=np.zeros((n, 51)),
    )

    hrrr = HRRRForecast(
        lat=33.45, lon=-112.07,
        times=times,
        temperature_2m=np.full(n, np.nan),
    )

    target_time = now + timedelta(hours=8)
    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Phoenix, AZ",
        lat_lon=(33.45, -112.07),
        threshold=22.0,
        comparison=Comparison.ABOVE,
        unit="C",
        target_date=target_time,
    )

    model = TemperatureModel()
    est_no_hrrr = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None)
    est_nan_hrrr = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None, hrrr=hrrr)

    # NaN HRRR should have no effect (tiny float diff from clock drift between calls)
    assert abs(est_nan_hrrr.probability - est_no_hrrr.probability) < 1e-6
    assert "HRRR" not in est_nan_hrrr.sources_used
