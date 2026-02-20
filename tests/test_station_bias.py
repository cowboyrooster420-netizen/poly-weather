"""Tests for per-station bias correction."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from weather_edge.calibration.station_bias import (
    StationBias,
    compute_station_bias,
    get_station_bias,
    load_biases,
    save_biases,
)
from weather_edge.forecasting.temperature import (
    TemperatureModel,
    _apply_station_bias_correction,
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

        # Should have loaded bundled defaults (14 stations, all zero)
        assert len(biases) == 14
        for b in biases.values():
            assert b.high_bias_c == 0.0
            assert b.low_bias_c == 0.0


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
    """save_biases writes valid JSON and invalidates cache."""
    biases = {
        "KTEST1": StationBias(
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
        assert data["version"] == 1
        assert data["training_days"] == 90
        assert data["stations"]["KTEST1"]["high_bias_c"] == 1.2
        assert data["stations"]["KTEST1"]["n_days"] == 82


# ---------- _apply_station_bias_correction ----------


def test_apply_shifts_mean():
    """Ensemble mean moves by exactly the bias amount."""
    members = np.array([30.0, 31.0, 32.0, 33.0, 34.0])
    original_mean = float(np.mean(members))

    with patch("weather_edge.forecasting.temperature.get_station_bias", return_value=1.5):
        corrected, detail = _apply_station_bias_correction(members, "KTEST1", "max")

    assert detail is not None
    assert "station bias" in detail
    assert "°C" in detail
    assert float(np.mean(corrected)) == pytest.approx(original_mean + 1.5)


def test_apply_preserves_spread():
    """Std dev is unchanged after bias correction."""
    members = np.array([30.0, 31.0, 32.0, 33.0, 34.0])
    original_std = float(np.std(members))

    with patch("weather_edge.forecasting.temperature.get_station_bias", return_value=2.0):
        corrected, _ = _apply_station_bias_correction(members, "KTEST1", "max")

    assert float(np.std(corrected)) == pytest.approx(original_std)


def test_apply_no_correction_when_zero():
    """No correction applied when bias is near zero."""
    members = np.array([30.0, 31.0, 32.0])

    with patch("weather_edge.forecasting.temperature.get_station_bias", return_value=0.005):
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
    with patch("weather_edge.forecasting.temperature.get_station_bias", return_value=0.0):
        est_no_bias = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None)

    # With positive bias (+2°C → station reads warm → ensemble shifts up)
    with patch("weather_edge.forecasting.temperature.get_station_bias", return_value=2.0):
        est_with_bias = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None)

    # Positive bias should increase P(above threshold)
    assert est_with_bias.probability > est_no_bias.probability

    # With negative bias (-2°C → station reads cool → ensemble shifts down)
    with patch("weather_edge.forecasting.temperature.get_station_bias", return_value=-2.0):
        est_neg_bias = await model.estimate(params, gfs=None, ecmwf=ecmwf, noaa=None)

    # Negative bias should decrease P(above threshold)
    assert est_neg_bias.probability < est_no_bias.probability
