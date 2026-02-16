"""Tests for signal analyzer (edge calculation + Kelly criterion)."""

from __future__ import annotations

import pytest

from weather_edge.forecasting.base import ProbabilityEstimate
from weather_edge.markets.models import Comparison, MarketParams, MarketType, WeatherMarket
from weather_edge.signals.analyzer import compute_kelly, generate_signal


class TestComputeKelly:
    def test_positive_edge_yes(self):
        """Positive edge → bet YES, Kelly > 0."""
        k = compute_kelly(0.6, 0.4, fraction=0.25, confidence=1.0)
        assert k > 0

    def test_negative_edge_no(self):
        """Negative edge → bet NO, Kelly > 0."""
        k = compute_kelly(0.3, 0.5, fraction=0.25, confidence=1.0)
        assert k > 0

    def test_no_edge_zero(self):
        """No edge → Kelly is 0."""
        k = compute_kelly(0.5, 0.5)
        assert k == 0.0

    def test_quarter_kelly_smaller(self):
        """Quarter Kelly should be smaller than full Kelly."""
        full = compute_kelly(0.7, 0.4, fraction=1.0, confidence=1.0)
        quarter = compute_kelly(0.7, 0.4, fraction=0.25, confidence=1.0)
        # Quarter Kelly is either 25% of full Kelly or capped at 0.25
        assert quarter <= full
        assert quarter <= 0.25

    def test_confidence_scaling(self):
        """Lower confidence should reduce Kelly."""
        high_conf = compute_kelly(0.7, 0.4, fraction=0.25, confidence=1.0)
        low_conf = compute_kelly(0.7, 0.4, fraction=0.25, confidence=0.5)
        assert abs(low_conf - high_conf * 0.5) < 0.001

    def test_extreme_market_prob_yes(self):
        """Kelly should be 0 when market_prob >= 0.999."""
        k = compute_kelly(1.0, 0.999, fraction=0.25, confidence=1.0)
        assert k == 0.0

    def test_extreme_market_prob_no(self):
        """Kelly should be 0 when market_prob <= 0.001."""
        k = compute_kelly(0.0, 0.001, fraction=0.25, confidence=1.0)
        assert k == 0.0

    def test_kelly_never_negative(self):
        """Kelly should never be negative."""
        for model_p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for market_p in [0.1, 0.3, 0.5, 0.7, 0.9]:
                k = compute_kelly(model_p, market_p)
                assert k >= 0, f"Negative Kelly: model={model_p}, market={market_p}, k={k}"

    def test_small_edge_small_kelly(self):
        """Small edge should produce small Kelly."""
        k = compute_kelly(0.51, 0.50, fraction=0.25, confidence=1.0)
        assert k < 0.01  # Very small


class TestGenerateSignal:
    def test_generates_yes_signal(self, sample_weather_market, sample_estimate):
        """Should generate YES signal when model_prob > market_prob."""
        signal = generate_signal(sample_weather_market, sample_estimate)

        assert signal is not None
        assert signal.direction == "YES"
        assert signal.edge > 0
        assert signal.kelly_fraction > 0
        assert signal.market_id == "test-market-001"

    def test_generates_no_signal(self):
        """Should generate NO signal when model_prob < market_prob."""
        market = WeatherMarket(
            market_id="m1", condition_id="c1",
            question="Q", description="",
            outcome_yes_price=0.70, outcome_no_price=0.30,
            params=MarketParams(market_type=MarketType.TEMPERATURE, location="X"),
        )
        estimate = ProbabilityEstimate(
            probability=0.40, raw_probability=0.40,
            confidence=0.8, lead_time_hours=24,
        )

        signal = generate_signal(market, estimate)

        assert signal is not None
        assert signal.direction == "NO"
        assert signal.edge < 0

    def test_no_signal_below_threshold(self):
        """Should return None when edge is below min_edge (10%)."""
        market = WeatherMarket(
            market_id="m1", condition_id="c1",
            question="Q", description="",
            outcome_yes_price=0.50, outcome_no_price=0.50,
            params=MarketParams(market_type=MarketType.TEMPERATURE, location="X"),
        )
        estimate = ProbabilityEstimate(
            probability=0.52, raw_probability=0.52,  # Only 2% edge
            confidence=0.8, lead_time_hours=24,
        )

        signal = generate_signal(market, estimate)

        assert signal is None

    def test_signal_at_exact_threshold(self):
        """Edge exactly at threshold should not generate signal."""
        market = WeatherMarket(
            market_id="m1", condition_id="c1",
            question="Q", description="",
            outcome_yes_price=0.45, outcome_no_price=0.55,
            params=MarketParams(market_type=MarketType.TEMPERATURE, location="X"),
        )
        estimate = ProbabilityEstimate(
            probability=0.50, raw_probability=0.50,  # Exactly 5% edge
            confidence=0.8, lead_time_hours=24,
        )

        signal = generate_signal(market, estimate)

        # 0.05 < 0.10, so below min_edge → no signal
        assert signal is None

    def test_no_signal_below_min_confidence(self):
        """Should return None when confidence is below min_confidence (0.30)."""
        market = WeatherMarket(
            market_id="m1", condition_id="c1",
            question="Q", description="",
            outcome_yes_price=0.30, outcome_no_price=0.70,
            params=MarketParams(market_type=MarketType.TEMPERATURE, location="X"),
        )
        estimate = ProbabilityEstimate(
            probability=0.60, raw_probability=0.60,  # 30% edge, but low confidence
            confidence=0.20, lead_time_hours=24,
        )

        signal = generate_signal(market, estimate)

        assert signal is None

    def test_signal_fields_populated(self, sample_weather_market, sample_estimate):
        """All signal fields should be properly populated."""
        signal = generate_signal(sample_weather_market, sample_estimate)

        assert signal is not None
        assert signal.market_type == "temperature"
        assert signal.location == "Phoenix, AZ"
        assert signal.confidence > 0
        assert signal.lead_time_hours == 24.0
        assert len(signal.sources) > 0
        assert signal.timestamp is not None

    def test_signal_no_params(self):
        """Market without params should still generate signal."""
        market = WeatherMarket(
            market_id="m1", condition_id="c1",
            question="Q", description="",
            outcome_yes_price=0.30, outcome_no_price=0.70,
            params=None,  # No parsed params
        )
        estimate = ProbabilityEstimate(
            probability=0.60, raw_probability=0.60,
            confidence=0.8, lead_time_hours=24,
        )

        signal = generate_signal(market, estimate)

        assert signal is not None
        assert signal.location == ""
        assert signal.market_type == "unknown"
