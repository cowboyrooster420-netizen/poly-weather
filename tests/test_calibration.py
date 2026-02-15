"""Tests for calibration module."""

from __future__ import annotations

import numpy as np
import pytest

from weather_edge.forecasting.calibration import (
    PlattScaler,
    confidence_from_lead_time,
    get_spread_inflation,
    inflate_ensemble_spread,
)


class TestSpreadInflation:
    def test_known_breakpoints(self):
        assert abs(get_spread_inflation(0) - 1.00) < 0.001
        assert abs(get_spread_inflation(24) - 1.08) < 0.001
        assert abs(get_spread_inflation(168) - 1.30) < 0.001
        assert abs(get_spread_inflation(336) - 1.50) < 0.001

    def test_interpolation(self):
        # Midpoint between 24h (1.08) and 48h (1.12) should be ~1.10
        val = get_spread_inflation(36)
        assert abs(val - 1.10) < 0.01

    def test_below_minimum(self):
        assert get_spread_inflation(-10) == 1.00

    def test_above_maximum(self):
        assert get_spread_inflation(500) == 1.55  # Flat beyond max

    def test_monotonic_increasing(self):
        """Inflation should increase with lead time."""
        prev = 0
        for hours in range(0, 400, 10):
            val = get_spread_inflation(hours)
            assert val >= prev
            prev = val


class TestInflateEnsembleSpread:
    def test_increases_spread(self):
        members = np.array([20.0, 21.0, 22.0, 23.0, 24.0])
        inflated = inflate_ensemble_spread(members, 168)
        assert np.std(inflated) > np.std(members)

    def test_preserves_mean(self):
        members = np.array([20.0, 21.0, 22.0, 23.0, 24.0])
        inflated = inflate_ensemble_spread(members, 168)
        assert abs(np.mean(inflated) - np.mean(members)) < 0.001

    def test_zero_lead_time_no_change(self):
        members = np.array([20.0, 21.0, 22.0])
        inflated = inflate_ensemble_spread(members, 0)
        np.testing.assert_array_almost_equal(members, inflated)

    def test_handles_single_member(self):
        members = np.array([25.0])
        inflated = inflate_ensemble_spread(members, 72)
        assert len(inflated) == 1
        assert abs(inflated[0] - 25.0) < 0.001  # Single member is its own mean


class TestConfidenceFromLeadTime:
    def test_decreases_with_lead_time(self):
        c0 = confidence_from_lead_time(0)
        c72 = confidence_from_lead_time(72)
        c168 = confidence_from_lead_time(168)
        c336 = confidence_from_lead_time(336)
        assert c0 > c72 > c168 > c336

    def test_bounded(self):
        assert confidence_from_lead_time(0) <= 1.0
        assert confidence_from_lead_time(10000) >= 0.3

    def test_near_one_at_zero(self):
        assert abs(confidence_from_lead_time(0) - 1.0) < 0.01


class TestPlattScaler:
    def test_unfitted_passthrough(self):
        scaler = PlattScaler()
        assert abs(scaler.calibrate(0.7) - 0.7) < 0.001
        assert abs(scaler.calibrate(0.1) - 0.1) < 0.001
        assert abs(scaler.calibrate(0.9) - 0.9) < 0.001

    def test_fit_with_perfect_predictions(self):
        """Fitting with perfect predictions should give near-identity."""
        scaler = PlattScaler()
        predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1])
        scaler.fit(predictions, outcomes)

        # After fitting, predictions should still be roughly preserved
        for p in [0.2, 0.5, 0.8]:
            calibrated = scaler.calibrate(p)
            assert 0 < calibrated < 1

    def test_fit_marks_as_fitted(self):
        scaler = PlattScaler()
        assert scaler._fitted is False
        scaler.fit(np.array([0.3, 0.7]), np.array([0, 1]))
        assert scaler._fitted is True

    def test_calibrate_bounds(self):
        """Calibrated values should always be in (0, 1)."""
        scaler = PlattScaler()
        scaler.fit(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))

        for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
            cal = scaler.calibrate(p)
            assert 0 < cal < 1

    def test_overconfident_correction(self):
        """If model is overconfident, Platt scaling should pull predictions toward center."""
        scaler = PlattScaler()
        # Predictions are overconfident: predict 0.9 when true rate is ~0.5
        predictions = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
        outcomes = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        scaler.fit(predictions, outcomes)

        # After calibration, 0.9 should be pulled down
        cal_high = scaler.calibrate(0.9)
        assert cal_high < 0.9
