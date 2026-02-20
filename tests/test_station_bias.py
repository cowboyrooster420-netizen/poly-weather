"""Tests for per-station bias correction."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from weather_edge.calibration.station_bias import (
    ConditionBias,
    SkyCondition,
    StationBias,
    StationBiasV2,
    classify_sky_condition,
    compute_station_bias,
    compute_station_bias_stratified,
    get_station_bias,
    get_station_bias_for_condition,
    load_biases,
    save_biases,
)
from weather_edge.forecasting.temperature import (
    TemperatureModel,
    _apply_station_bias_correction,
    _get_forecast_cloud_cover,
)
from weather_edge.markets.models import Comparison, MarketParams, MarketType
from weather_edge.weather.models import EnsembleForecast


# ---------- compute_station_bias ----------


def test_compute_bias_positive():
    """WU reads warmer than reanalysis → positive bias."""
    wu_highs = [32.0, 33.0, 34.0, 35.0]
    wu_lows = [20.0, 21.0, 22.0, 23.0]
    om_maxs = [30.0, 31.0, 32.0, 33.0]
    om_mins = [19.0, 20.0, 21.0, 22.0]

    bias = compute_station_bias(wu_highs, wu_lows, om_maxs, om_mins,
                                station_id="TEST1", city="testville")

    assert bias.high_bias_c == pytest.approx(2.0)
    assert bias.low_bias_c == pytest.approx(1.0)
    assert bias.mean_bias_c == pytest.approx(1.5)
    assert bias.n_days == 4
    assert bias.station_id == "TEST1"


def test_compute_bias_negative():
    """WU reads cooler than reanalysis → negative bias."""
    wu_highs = [28.0, 29.0, 30.0]
    wu_lows = [17.0, 18.0, 19.0]
    om_maxs = [30.0, 31.0, 32.0]
    om_mins = [19.0, 20.0, 21.0]

    bias = compute_station_bias(wu_highs, wu_lows, om_maxs, om_mins)

    assert bias.high_bias_c == pytest.approx(-2.0)
    assert bias.low_bias_c == pytest.approx(-2.0)
    assert bias.mean_bias_c == pytest.approx(-2.0)


def test_compute_bias_with_missing_days():
    """Gaps in data handled — compute from whatever pairs we have."""
    # Only 2 valid days, with different diffs to produce nonzero std
    wu_highs = [32.0, 35.0]
    wu_lows = [20.0, 23.0]
    om_maxs = [30.0, 32.0]  # diffs: 2.0, 3.0
    om_mins = [19.0, 21.0]  # diffs: 1.0, 2.0

    bias = compute_station_bias(wu_highs, wu_lows, om_maxs, om_mins)

    assert bias.high_bias_c == pytest.approx(2.5)
    assert bias.low_bias_c == pytest.approx(1.5)
    assert bias.n_days == 2
    assert bias.high_std_c > 0  # std is defined with n=2


def test_compute_bias_single_day():
    """Single day still works (std = 0)."""
    bias = compute_station_bias([32.0], [20.0], [30.0], [19.0])

    assert bias.high_bias_c == pytest.approx(2.0)
    assert bias.low_bias_c == pytest.approx(1.0)
    assert bias.high_std_c == 0.0  # can't compute std from 1 sample
    assert bias.n_days == 1


# ---------- load / save / get ----------


def test_load_biases_from_json(tmp_path):
    """Load biases from a well-formed JSON file."""
    bias_file = tmp_path / "biases.json"
    bias_file.write_text(json.dumps({
        "version": 1,
        "generated_at": "2026-01-01T00:00:00Z",
        "training_days": 90,
        "stations": {
            "KTEST1": {
                "city": "testville",
                "high_bias_c": 1.5,
                "low_bias_c": 0.8,
                "mean_bias_c": 1.15,
                "n_days": 80,
            }
        },
    }))

    with patch("weather_edge.calibration.station_bias.get_settings") as mock_settings:
        mock_settings.return_value.station_bias_enabled = True
        mock_settings.return_value.station_bias_path = bias_file

        # Force reload
        biases = load_biases(force=True)

        assert "KTEST1" in biases
        assert biases["KTEST1"].high_bias_c == pytest.approx(1.5)
        assert biases["KTEST1"].low_bias_c == pytest.approx(0.8)
        assert biases["KTEST1"].mean_bias_c == pytest.approx(1.15)
        assert biases["KTEST1"].n_days == 80


def test_load_fallback(tmp_path):
    """Falls back to bundled defaults when user file doesn't exist."""
    missing = tmp_path / "nonexistent.json"

    with patch("weather_edge.calibration.station_bias.get_settings") as mock_settings:
        mock_settings.return_value.station_bias_enabled = True
        mock_settings.return_value.station_bias_path = missing

        biases = load_biases(force=True)

        # Should have loaded bundled defaults (14 stations)
        assert len(biases) == 14
        for b in biases.values():
            assert isinstance(b.high_bias_c, float)
            assert isinstance(b.low_bias_c, float)


def test_get_station_bias_unknown():
    """Unknown station returns 0.0 bias."""
    with patch("weather_edge.calibration.station_bias.get_settings") as mock_settings:
        mock_settings.return_value.station_bias_enabled = True
        mock_settings.return_value.station_bias_path = Path("/nonexistent")

        # Force reload to use bundled defaults
        load_biases(force=True)

        result = get_station_bias("KNONEXIST999", "max")
        assert result == 0.0


def test_get_station_bias_disabled():
    """Returns 0.0 when station_bias_enabled is False."""
    with patch("weather_edge.calibration.station_bias.get_settings") as mock_settings:
        mock_settings.return_value.station_bias_enabled = False

        result = get_station_bias("KGAHAPEV1", "max")
        assert result == 0.0


def test_get_station_bias_aggregation(tmp_path):
    """Returns correct bias for each aggregation type."""
    bias_file = tmp_path / "biases.json"
    bias_file.write_text(json.dumps({
        "version": 1,
        "generated_at": "2026-01-01T00:00:00Z",
        "training_days": 90,
        "stations": {
            "KTEST1": {
                "city": "test",
                "high_bias_c": 2.0,
                "low_bias_c": 0.5,
                "mean_bias_c": 1.25,
                "n_days": 50,
            }
        },
    }))

    with patch("weather_edge.calibration.station_bias.get_settings") as mock_settings:
        mock_settings.return_value.station_bias_enabled = True
        mock_settings.return_value.station_bias_path = bias_file
        load_biases(force=True)

        assert get_station_bias("KTEST1", "max") == pytest.approx(2.0)
        assert get_station_bias("KTEST1", "min") == pytest.approx(0.5)
        assert get_station_bias("KTEST1", None) == pytest.approx(1.25)


def test_save_biases(tmp_path):
    """save_biases writes valid v2 JSON and invalidates cache."""
    biases = {
        "KTEST1": StationBiasV2(
            station_id="KTEST1", city="testville",
            high_bias_c=1.2, low_bias_c=0.8, mean_bias_c=1.0,
            high_std_c=0.5, low_std_c=0.4, n_days=82,
        )
    }

    bias_file = tmp_path / "biases.json"
    with patch("weather_edge.calibration.station_bias.get_settings") as mock_settings:
        mock_settings.return_value.station_bias_enabled = True
        mock_settings.return_value.station_bias_path = bias_file

        result_path = save_biases(biases, training_days=90)

        assert result_path == bias_file
        data = json.loads(bias_file.read_text())
        assert data["version"] == 2
        assert data["training_days"] == 90
        assert data["stations"]["KTEST1"]["high_bias_c"] == 1.2
        assert data["stations"]["KTEST1"]["n_days"] == 82


# ---------- _apply_station_bias_correction ----------


def test_apply_shifts_mean():
    """Ensemble mean moves by exactly the bias amount."""
    members = np.array([30.0, 31.0, 32.0, 33.0, 34.0])
    original_mean = float(np.mean(members))

    with patch("weather_edge.forecasting.temperature.get_station_bias_for_condition", return_value=1.5):
        corrected, detail = _apply_station_bias_correction(members, "KTEST1", "max")

    assert detail is not None
    assert "station bias" in detail
    assert "°C" in detail
    assert float(np.mean(corrected)) == pytest.approx(original_mean + 1.5)


def test_apply_preserves_spread():
    """Std dev is unchanged after bias correction."""
    members = np.array([30.0, 31.0, 32.0, 33.0, 34.0])
    original_std = float(np.std(members))

    with patch("weather_edge.forecasting.temperature.get_station_bias_for_condition", return_value=2.0):
        corrected, _ = _apply_station_bias_correction(members, "KTEST1", "max")

    assert float(np.std(corrected)) == pytest.approx(original_std)


def test_apply_no_correction_when_zero():
    """No correction applied when bias is near zero."""
    members = np.array([30.0, 31.0, 32.0])

    with patch("weather_edge.forecasting.temperature.get_station_bias_for_condition", return_value=0.005):
        corrected, detail = _apply_station_bias_correction(members, "KTEST1", "max")

    assert detail is None
    np.testing.assert_array_equal(corrected, members)


# ---------- End-to-end probability shift ----------


@pytest.mark.asyncio
async def test_end_to_end_probability_shift():
    """Probability moves in expected direction when bias is applied.

    A positive high_bias means the station reads warm. We shift ensemble
    members up (warmer), which should increase P(above threshold).
    """
    now = datetime.now(timezone.utc)
    np.random.seed(42)
    times = [now + timedelta(hours=i) for i in range(48)]
    n = len(times)

    ecmwf = EnsembleForecast(
        source="ecmwf", lat=33.661, lon=-84.399,
        times=times,
        temperature_2m=np.random.normal(35, 3, (n, 51)),
        precipitation=np.zeros((n, 51)),
    )

    params = MarketParams(
        market_type=MarketType.TEMPERATURE,
        location="Atlanta, GA",
        lat_lon=(33.661, -84.399),
        threshold=95.0,  # 95F ≈ 35C, near ensemble mean
        comparison=Comparison.ABOVE,
        unit="F",
        target_date=now + timedelta(hours=24),
    )

    model = TemperatureModel()

    # Without bias (zero bias from bundled defaults → no shift)
    with patch("weather_edge.forecasting.temperature.get_station_bias_for_condition", return_value=0.0):
        est_no_bias = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None)

    # With positive bias (+2°C → station reads warm → ensemble shifts up)
    with patch("weather_edge.forecasting.temperature.get_station_bias_for_condition", return_value=2.0):
        est_with_bias = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None)

    # Positive bias should increase P(above threshold)
    assert est_with_bias.probability > est_no_bias.probability

    # With negative bias (-2°C → station reads cool → ensemble shifts down)
    with patch("weather_edge.forecasting.temperature.get_station_bias_for_condition", return_value=-2.0):
        est_neg_bias = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None)

    # Negative bias should decrease P(above threshold)
    assert est_neg_bias.probability < est_no_bias.probability


# ---------- classify_sky_condition ----------


def test_classify_clear():
    assert classify_sky_condition(0.0) == SkyCondition.CLEAR
    assert classify_sky_condition(10.0) == SkyCondition.CLEAR
    assert classify_sky_condition(24.9) == SkyCondition.CLEAR


def test_classify_partly_cloudy():
    assert classify_sky_condition(25.0) == SkyCondition.PARTLY_CLOUDY
    assert classify_sky_condition(50.0) == SkyCondition.PARTLY_CLOUDY
    assert classify_sky_condition(74.9) == SkyCondition.PARTLY_CLOUDY


def test_classify_overcast():
    assert classify_sky_condition(75.0) == SkyCondition.OVERCAST
    assert classify_sky_condition(100.0) == SkyCondition.OVERCAST


# ---------- compute_station_bias_stratified ----------


def test_stratified_basic():
    """Stratified compute separates days by cloud cover bucket."""
    # 6 clear days (cc < 25): WU reads 5C warm
    # 6 overcast days (cc >= 75): WU reads 1C cool
    wu_highs = [35.0] * 6 + [29.0] * 6
    wu_lows = [22.0] * 6 + [18.0] * 6
    om_maxs = [30.0] * 6 + [30.0] * 6
    om_mins = [20.0] * 6 + [20.0] * 6
    cloud_covers: list[float | None] = [10.0] * 6 + [80.0] * 6

    bias = compute_station_bias_stratified(
        wu_highs, wu_lows, om_maxs, om_mins, cloud_covers,
        station_id="KTEST1", city="test",
    )

    assert bias.n_days == 12
    assert len(bias.condition_biases) == 3

    clear = next(cb for cb in bias.condition_biases if cb.condition == SkyCondition.CLEAR)
    overcast = next(cb for cb in bias.condition_biases if cb.condition == SkyCondition.OVERCAST)
    partly = next(cb for cb in bias.condition_biases if cb.condition == SkyCondition.PARTLY_CLOUDY)

    assert clear.high_bias_c == pytest.approx(5.0)
    assert clear.n_days == 6
    assert overcast.high_bias_c == pytest.approx(-1.0)
    assert overcast.n_days == 6
    assert partly.n_days == 0


def test_stratified_none_cloud_covers():
    """Days with None cloud cover are excluded from condition buckets."""
    wu_highs = [32.0, 33.0, 34.0]
    wu_lows = [20.0, 21.0, 22.0]
    om_maxs = [30.0, 31.0, 32.0]
    om_mins = [19.0, 20.0, 21.0]
    cloud_covers: list[float | None] = [None, None, None]

    bias = compute_station_bias_stratified(
        wu_highs, wu_lows, om_maxs, om_mins, cloud_covers,
    )

    # Global bias is still computed
    assert bias.n_days == 3
    assert bias.high_bias_c == pytest.approx(2.0)
    # All condition buckets are empty
    for cb in bias.condition_biases:
        assert cb.n_days == 0


# ---------- get_station_bias_for_condition ----------


def test_condition_bias_clear_bucket(tmp_path):
    """When clear bucket has enough days, returns condition-specific bias."""
    bias_file = tmp_path / "biases.json"
    bias_file.write_text(json.dumps({
        "version": 2,
        "stations": {
            "KTEST1": {
                "city": "test",
                "high_bias_c": 1.0,
                "low_bias_c": 0.5,
                "mean_bias_c": 0.75,
                "n_days": 80,
                "conditions": {
                    "clear": {
                        "high_bias_c": 3.5,
                        "low_bias_c": 2.0,
                        "mean_bias_c": 2.75,
                        "n_days": 25,
                    },
                    "partly": {
                        "high_bias_c": 0.5,
                        "low_bias_c": 0.3,
                        "mean_bias_c": 0.4,
                        "n_days": 35,
                    },
                    "overcast": {
                        "high_bias_c": -1.0,
                        "low_bias_c": -0.5,
                        "mean_bias_c": -0.75,
                        "n_days": 20,
                    },
                },
            }
        },
    }))

    with patch("weather_edge.calibration.station_bias.get_settings") as mock_settings:
        mock_settings.return_value.station_bias_enabled = True
        mock_settings.return_value.station_bias_path = bias_file
        load_biases(force=True)

        # Clear sky (cloud_cover=10%) → clear bucket bias
        assert get_station_bias_for_condition("KTEST1", "max", 10.0) == pytest.approx(3.5)
        # Partly cloudy (cloud_cover=50%) → partly bucket
        assert get_station_bias_for_condition("KTEST1", "max", 50.0) == pytest.approx(0.5)
        # Overcast (cloud_cover=80%) → overcast bucket
        assert get_station_bias_for_condition("KTEST1", "max", 80.0) == pytest.approx(-1.0)


def test_condition_bias_fallback_to_global(tmp_path):
    """Falls back to global when bucket has < 10 days."""
    bias_file = tmp_path / "biases.json"
    bias_file.write_text(json.dumps({
        "version": 2,
        "stations": {
            "KTEST1": {
                "city": "test",
                "high_bias_c": 1.0,
                "low_bias_c": 0.5,
                "mean_bias_c": 0.75,
                "n_days": 50,
                "conditions": {
                    "clear": {
                        "high_bias_c": 5.0,
                        "low_bias_c": 3.0,
                        "mean_bias_c": 4.0,
                        "n_days": 5,
                    },
                },
            }
        },
    }))

    with patch("weather_edge.calibration.station_bias.get_settings") as mock_settings:
        mock_settings.return_value.station_bias_enabled = True
        mock_settings.return_value.station_bias_path = bias_file
        load_biases(force=True)

        # Clear bucket has only 5 days (< 10 threshold) → falls back to global
        assert get_station_bias_for_condition("KTEST1", "max", 10.0) == pytest.approx(1.0)


def test_condition_bias_fallback_none_cloud_cover(tmp_path):
    """Falls back to global when cloud_cover_pct is None."""
    bias_file = tmp_path / "biases.json"
    bias_file.write_text(json.dumps({
        "version": 2,
        "stations": {
            "KTEST1": {
                "city": "test",
                "high_bias_c": 1.0,
                "low_bias_c": 0.5,
                "mean_bias_c": 0.75,
                "n_days": 50,
                "conditions": {
                    "clear": {
                        "high_bias_c": 5.0,
                        "low_bias_c": 3.0,
                        "mean_bias_c": 4.0,
                        "n_days": 30,
                    },
                },
            }
        },
    }))

    with patch("weather_edge.calibration.station_bias.get_settings") as mock_settings:
        mock_settings.return_value.station_bias_enabled = True
        mock_settings.return_value.station_bias_path = bias_file
        load_biases(force=True)

        # cloud_cover_pct is None → falls back to global
        assert get_station_bias_for_condition("KTEST1", "max", None) == pytest.approx(1.0)


def test_condition_bias_unknown_station():
    """Unknown station returns 0.0."""
    with patch("weather_edge.calibration.station_bias.get_settings") as mock_settings:
        mock_settings.return_value.station_bias_enabled = True
        mock_settings.return_value.station_bias_path = Path("/nonexistent")
        load_biases(force=True)

        assert get_station_bias_for_condition("KNONEXIST999", "max", 50.0) == 0.0


# ---------- v2 JSON round-trip ----------


def test_save_and_load_v2_with_conditions(tmp_path):
    """v2 JSON with conditions round-trips correctly."""
    biases = {
        "KTEST1": StationBiasV2(
            station_id="KTEST1", city="testville",
            high_bias_c=1.0, low_bias_c=0.5, mean_bias_c=0.75,
            high_std_c=2.0, low_std_c=1.5, n_days=60,
            condition_biases=(
                ConditionBias(
                    condition=SkyCondition.CLEAR,
                    high_bias_c=3.0, low_bias_c=2.0, mean_bias_c=2.5,
                    high_std_c=1.0, low_std_c=0.8, n_days=20,
                ),
                ConditionBias(
                    condition=SkyCondition.PARTLY_CLOUDY,
                    high_bias_c=0.5, low_bias_c=0.3, mean_bias_c=0.4,
                    high_std_c=1.5, low_std_c=1.2, n_days=25,
                ),
                ConditionBias(
                    condition=SkyCondition.OVERCAST,
                    high_bias_c=-1.5, low_bias_c=-0.5, mean_bias_c=-1.0,
                    high_std_c=0.8, low_std_c=0.6, n_days=15,
                ),
            ),
        )
    }

    bias_file = tmp_path / "biases.json"
    with patch("weather_edge.calibration.station_bias.get_settings") as mock_settings:
        mock_settings.return_value.station_bias_enabled = True
        mock_settings.return_value.station_bias_path = bias_file

        save_biases(biases, training_days=90)

        data = json.loads(bias_file.read_text())
        assert data["version"] == 2
        assert "conditions" in data["stations"]["KTEST1"]
        assert data["stations"]["KTEST1"]["conditions"]["clear"]["high_bias_c"] == 3.0
        assert data["stations"]["KTEST1"]["conditions"]["clear"]["n_days"] == 20
        assert data["stations"]["KTEST1"]["conditions"]["partly"]["high_bias_c"] == 0.5
        assert data["stations"]["KTEST1"]["conditions"]["overcast"]["high_bias_c"] == -1.5

        # Reload and verify
        loaded = load_biases(force=True)
        b = loaded["KTEST1"]
        assert len(b.condition_biases) == 3
        clear = next(cb for cb in b.condition_biases if cb.condition == SkyCondition.CLEAR)
        assert clear.high_bias_c == pytest.approx(3.0)
        assert clear.n_days == 20


def test_load_v1_json_no_conditions(tmp_path):
    """v1 JSON (no conditions key) loads with empty condition_biases."""
    bias_file = tmp_path / "biases.json"
    bias_file.write_text(json.dumps({
        "version": 1,
        "stations": {
            "KTEST1": {
                "city": "test",
                "high_bias_c": 2.0,
                "low_bias_c": 1.0,
                "mean_bias_c": 1.5,
                "n_days": 50,
            }
        },
    }))

    with patch("weather_edge.calibration.station_bias.get_settings") as mock_settings:
        mock_settings.return_value.station_bias_enabled = True
        mock_settings.return_value.station_bias_path = bias_file
        loaded = load_biases(force=True)

        b = loaded["KTEST1"]
        assert b.high_bias_c == pytest.approx(2.0)
        assert b.condition_biases == ()
        # get_station_bias_for_condition falls back to global
        assert get_station_bias_for_condition("KTEST1", "max", 10.0) == pytest.approx(2.0)


def test_apply_with_cloud_cover():
    """_apply_station_bias_correction passes cloud_cover_pct through."""
    members = np.array([30.0, 31.0, 32.0, 33.0, 34.0])
    original_mean = float(np.mean(members))

    with patch("weather_edge.forecasting.temperature.get_station_bias_for_condition", return_value=3.0) as mock_fn:
        corrected, detail = _apply_station_bias_correction(members, "KTEST1", "max", cloud_cover_pct=15.0)

    # Verify cloud_cover_pct was forwarded
    mock_fn.assert_called_once_with("KTEST1", "max", 15.0)
    assert float(np.mean(corrected)) == pytest.approx(original_mean + 3.0)


# ---------- _get_forecast_cloud_cover ----------


def _make_ensemble_with_cloud(
    start_utc: datetime,
    hours: int = 24,
    n_members: int = 3,
    cloud_value: float = 40.0,
) -> EnsembleForecast:
    """Helper: build an EnsembleForecast with constant cloud cover."""
    times = [start_utc + timedelta(hours=h) for h in range(hours)]
    n = len(times)
    return EnsembleForecast(
        source="ecmwf",
        lat=33.0,
        lon=-84.0,
        times=times,
        temperature_2m=np.full((n, n_members), 20.0),
        precipitation=np.zeros((n, n_members)),
        cloud_cover=np.full((n, n_members), cloud_value),
    )


def test_cloud_cover_max_daytime_only():
    """For daily_aggregation='max', only 10:00-18:00 local hours are used."""
    # Build forecast with 24 hourly steps starting at 2026-02-21 00:00 UTC.
    # Use UTC as the "local" timezone for simplicity.
    start = datetime(2026, 2, 21, 0, 0, 0, tzinfo=timezone.utc)
    times = [start + timedelta(hours=h) for h in range(24)]
    n = len(times)
    n_members = 3

    # Set cloud cover: nighttime hours = 90%, daytime 10-17 = 20%
    cloud = np.full((n, n_members), 90.0)
    for h in range(10, 18):
        cloud[h, :] = 20.0

    fc = EnsembleForecast(
        source="ecmwf", lat=33.0, lon=-84.0, times=times,
        temperature_2m=np.full((n, n_members), 20.0),
        precipitation=np.zeros((n, n_members)),
        cloud_cover=cloud,
    )

    target = datetime(2026, 2, 21, 12, 0, 0, tzinfo=timezone.utc)

    # "max" should use daytime only → ~20%
    result = _get_forecast_cloud_cover(fc, target, tz_name="UTC", daily_aggregation="max")
    assert result is not None
    assert result == pytest.approx(20.0)

    # "min" should use full 24h → weighted mix of 90 and 20
    result_min = _get_forecast_cloud_cover(fc, target, tz_name="UTC", daily_aggregation="min")
    assert result_min is not None
    assert result_min > 60.0  # heavily weighted toward 90


def test_cloud_cover_full_day_for_min():
    """For daily_aggregation='min' or None, full 24h is used."""
    start = datetime(2026, 2, 21, 0, 0, 0, tzinfo=timezone.utc)
    fc = _make_ensemble_with_cloud(start, hours=24, cloud_value=55.0)
    target = datetime(2026, 2, 21, 12, 0, 0, tzinfo=timezone.utc)

    result = _get_forecast_cloud_cover(fc, target, tz_name="UTC", daily_aggregation="min")
    assert result == pytest.approx(55.0)

    result_none = _get_forecast_cloud_cover(fc, target, tz_name="UTC", daily_aggregation=None)
    assert result_none == pytest.approx(55.0)


def test_cloud_cover_fallback_daytime_empty():
    """Falls back to full 24h when daytime window has no data."""
    # Forecast only covers 00:00-06:00 UTC — no daytime hours
    start = datetime(2026, 2, 21, 0, 0, 0, tzinfo=timezone.utc)
    fc = _make_ensemble_with_cloud(start, hours=6, cloud_value=30.0)
    target = datetime(2026, 2, 21, 0, 0, 0, tzinfo=timezone.utc)

    result = _get_forecast_cloud_cover(fc, target, tz_name="UTC", daily_aggregation="max")
    # Daytime window (10-18) has no data, falls back to full day → 30.0
    assert result == pytest.approx(30.0)


def test_cloud_cover_all_nan():
    """Returns None when all cloud cover values are NaN."""
    start = datetime(2026, 2, 21, 0, 0, 0, tzinfo=timezone.utc)
    times = [start + timedelta(hours=h) for h in range(24)]
    n = len(times)
    n_members = 3

    fc = EnsembleForecast(
        source="ecmwf", lat=33.0, lon=-84.0, times=times,
        temperature_2m=np.full((n, n_members), 20.0),
        precipitation=np.zeros((n, n_members)),
        cloud_cover=np.full((n, n_members), np.nan),
    )
    target = datetime(2026, 2, 21, 12, 0, 0, tzinfo=timezone.utc)

    result = _get_forecast_cloud_cover(fc, target, tz_name="UTC", daily_aggregation="max")
    assert result is None


def test_cloud_cover_none_field():
    """Returns None when forecast has no cloud_cover at all."""
    start = datetime(2026, 2, 21, 0, 0, 0, tzinfo=timezone.utc)
    times = [start + timedelta(hours=h) for h in range(24)]
    n = len(times)

    fc = EnsembleForecast(
        source="ecmwf", lat=33.0, lon=-84.0, times=times,
        temperature_2m=np.full((n, 3), 20.0),
        precipitation=np.zeros((n, 3)),
        cloud_cover=None,
    )
    target = datetime(2026, 2, 21, 12, 0, 0, tzinfo=timezone.utc)

    result = _get_forecast_cloud_cover(fc, target, tz_name="UTC", daily_aggregation="max")
    assert result is None


# ---------- condition bias with "min" aggregation ----------


def test_condition_bias_min_aggregation(tmp_path):
    """Condition-dependent bias returns low_bias_c for aggregation='min'."""
    bias_file = tmp_path / "biases.json"
    bias_file.write_text(json.dumps({
        "version": 2,
        "stations": {
            "KTEST1": {
                "city": "test",
                "high_bias_c": 1.0,
                "low_bias_c": 0.5,
                "mean_bias_c": 0.75,
                "n_days": 80,
                "conditions": {
                    "clear": {
                        "high_bias_c": 3.5,
                        "low_bias_c": 2.0,
                        "mean_bias_c": 2.75,
                        "n_days": 25,
                    },
                    "overcast": {
                        "high_bias_c": -1.0,
                        "low_bias_c": -0.5,
                        "mean_bias_c": -0.75,
                        "n_days": 20,
                    },
                },
            }
        },
    }))

    with patch("weather_edge.calibration.station_bias.get_settings") as mock_settings:
        mock_settings.return_value.station_bias_enabled = True
        mock_settings.return_value.station_bias_path = bias_file
        load_biases(force=True)

        # Clear, min → low_bias_c from clear bucket
        assert get_station_bias_for_condition("KTEST1", "min", 10.0) == pytest.approx(2.0)
        # Overcast, min → low_bias_c from overcast bucket
        assert get_station_bias_for_condition("KTEST1", "min", 80.0) == pytest.approx(-0.5)
        # Clear, None (point-in-time) → mean_bias_c from clear bucket
        assert get_station_bias_for_condition("KTEST1", None, 10.0) == pytest.approx(2.75)
