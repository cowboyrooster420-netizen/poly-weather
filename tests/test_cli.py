"""Tests for CLI commands with mocked dependencies."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from weather_edge.cli import app
from weather_edge.markets.models import Comparison, MarketParams, MarketType, WeatherMarket
from weather_edge.signals.models import Signal
from weather_edge.weather.models import EnsembleForecast

runner = CliRunner()


def _make_ensemble(now):
    np.random.seed(42)
    times = [now + timedelta(hours=i) for i in range(48)]
    n = len(times)
    return EnsembleForecast(
        source="ecmwf", lat=33.45, lon=-112.07,
        times=times,
        temperature_2m=np.random.normal(35, 3, (n, 51)),
        precipitation=np.maximum(0, np.random.exponential(2, (n, 51)) - 1),
    )


@pytest.fixture
def now():
    return datetime.now(timezone.utc)


@pytest.fixture
def sample_signal(now):
    return Signal(
        market_id="m1", question="Will Phoenix hit 120F?",
        market_type="temperature", location="Phoenix, AZ",
        model_prob=0.55, market_prob=0.35, edge=0.20,
        kelly_fraction=0.08, confidence=0.85, direction="YES",
        lead_time_hours=24.0, sources=["ECMWF"], details="Test",
        timestamp=now,
    )


class TestScanCommand:
    def test_scan_json_output(self, sample_signal):
        """Test scan with JSON output."""
        async def mock_run():
            return [sample_signal]

        with patch("weather_edge.pipeline.run_pipeline", side_effect=mock_run):
            result = runner.invoke(app, ["scan", "--output", "json"])

        assert result.exit_code == 0
        assert "market_id" in result.output

    def test_scan_csv_output(self, sample_signal):
        """Test scan with CSV output."""
        async def mock_run():
            return [sample_signal]

        with patch("weather_edge.pipeline.run_pipeline", side_effect=mock_run):
            result = runner.invoke(app, ["scan", "--output", "csv"])

        assert result.exit_code == 0
        assert "market_id" in result.output

    def test_scan_table_output(self, sample_signal):
        """Test scan with default table output."""
        async def mock_run():
            return [sample_signal]

        with patch("weather_edge.pipeline.run_pipeline", side_effect=mock_run):
            result = runner.invoke(app, ["scan"])

        assert result.exit_code == 0

    def test_scan_no_markets(self):
        """Test scan when no weather markets found."""
        async def mock_run():
            return []

        with patch("weather_edge.pipeline.run_pipeline", side_effect=mock_run):
            result = runner.invoke(app, ["scan"])

        assert result.exit_code == 0


class TestListMarketsCommand:
    def test_list_markets(self, now):
        """Test list-markets command."""
        markets = [
            WeatherMarket(
                market_id="m1", condition_id="c1",
                question="Will Phoenix exceed 100F?", description="",
                outcome_yes_price=0.35, outcome_no_price=0.65,
                volume=50000.0,
                params=MarketParams(
                    market_type=MarketType.TEMPERATURE,
                    location="Phoenix, AZ",
                    threshold=100.0,
                    comparison=Comparison.ABOVE,
                    unit="F",
                ),
            ),
        ]

        async def mock_scan():
            return markets

        with patch("weather_edge.pipeline.scan_markets", side_effect=mock_scan):
            result = runner.invoke(app, ["list-markets"])

        assert result.exit_code == 0

    def test_list_markets_empty(self):
        """Test list-markets when no markets found."""
        async def mock_scan():
            return []

        with patch("weather_edge.pipeline.scan_markets", side_effect=mock_scan):
            result = runner.invoke(app, ["list-markets"])

        assert result.exit_code == 0
        assert "No weather markets" in result.output


class TestInspectCommand:
    def test_inspect_market(self, now):
        """Test inspect command for a specific market."""
        raw_market = {
            "id": "test-market-1",
            "conditionId": "cond-1",
            "question": "Will Phoenix temperature exceed 120F?",
            "description": "Temperature market",
            "outcomePrices": '["0.30","0.70"]',
            "active": True,
            "volume": "50000",
            "slug": "phoenix-120f",
        }

        ecmwf = _make_ensemble(now)
        gfs = _make_ensemble(now)
        mock_latlon = (33.45, -112.07)

        # Patch at the source modules since cli.py imports inside _run()
        with patch("weather_edge.markets.client.fetch_all_active_markets", new_callable=AsyncMock) as mock_fetch, \
             patch("weather_edge.markets.classifier.classify_market", new_callable=AsyncMock) as mock_classify, \
             patch("weather_edge.markets.parser.parse_market", new_callable=AsyncMock) as mock_parse, \
             patch("weather_edge.weather.openmeteo.fetch_both_ensembles", new_callable=AsyncMock) as mock_ens, \
             patch("weather_edge.weather.noaa.fetch_noaa_forecast", new_callable=AsyncMock) as mock_noaa:

            mock_fetch.return_value = [raw_market]
            mock_classify.return_value = (True, 0.95)
            mock_parse.return_value = MarketParams(
                market_type=MarketType.TEMPERATURE,
                location="Phoenix, AZ",
                lat_lon=mock_latlon,
                threshold=120.0,
                comparison=Comparison.ABOVE,
                unit="F",
                target_date=now + timedelta(hours=24),
            )
            mock_ens.return_value = (gfs, ecmwf)
            mock_noaa.return_value = None

            result = runner.invoke(app, ["inspect", "test-market-1"])

        assert result.exit_code == 0
        assert "Inspecting" in result.output

    def test_inspect_not_found(self):
        """Test inspect when market is not found."""
        with patch("weather_edge.markets.client.fetch_all_active_markets", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []
            result = runner.invoke(app, ["inspect", "nonexistent"])

        assert result.exit_code == 0
        assert "not found" in result.output


class TestStatsCommand:
    def test_stats_with_data(self):
        """Test stats command with logged data."""
        summary = {
            "total_signals": 100,
            "resolved": 50,
            "wins": 35,
            "win_rate": 0.70,
            "avg_abs_edge": 0.12,
            "brier_score": 0.182,
        }

        with patch("weather_edge.signals.tracker.SignalTracker.get_performance_summary", new_callable=AsyncMock) as mock_perf:
            mock_perf.return_value = summary
            # Also need to mock _ensure_db since it tries to create files
            with patch("weather_edge.signals.tracker.SignalTracker._ensure_db", new_callable=AsyncMock):
                result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "100" in result.output
        assert "70.0%" in result.output
        assert "0.182" in result.output

    def test_stats_empty(self):
        """Test stats command with no data."""
        summary = {
            "total_signals": 0,
            "resolved": 0,
            "wins": 0,
            "win_rate": None,
            "avg_abs_edge": None,
            "brier_score": None,
        }

        with patch("weather_edge.signals.tracker.SignalTracker.get_performance_summary", new_callable=AsyncMock) as mock_perf:
            mock_perf.return_value = summary
            with patch("weather_edge.signals.tracker.SignalTracker._ensure_db", new_callable=AsyncMock):
                result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0
        assert "N/A" in result.output


class TestResolveCommand:
    def test_resolve_with_results(self):
        """Test resolve command when markets are resolved."""
        resolved = [
            {"market_id": "m1", "question": "Atlanta 57F?", "outcome": 0, "direction": "NO", "correct": True},
            {"market_id": "m2", "question": "Dallas 40F?", "outcome": 1, "direction": "YES", "correct": True},
        ]
        summary = {
            "total_signals": 5,
            "resolved": 3,
            "wins": 2,
            "win_rate": 0.667,
            "avg_abs_edge": 0.10,
            "brier_score": 0.150,
        }

        with patch("weather_edge.signals.resolver.resolve_pending_signals", new_callable=AsyncMock) as mock_resolve, \
             patch("weather_edge.signals.tracker.SignalTracker.get_performance_summary", new_callable=AsyncMock) as mock_perf, \
             patch("weather_edge.signals.tracker.SignalTracker._ensure_db", new_callable=AsyncMock):
            mock_resolve.return_value = resolved
            mock_perf.return_value = summary
            result = runner.invoke(app, ["resolve"])

        assert result.exit_code == 0
        assert "Atlanta" in result.output
        assert "Dallas" in result.output

    def test_resolve_no_pending(self):
        """Test resolve command when nothing to resolve."""
        summary = {
            "total_signals": 5,
            "resolved": 5,
            "wins": 3,
            "win_rate": 0.60,
            "avg_abs_edge": 0.10,
            "brier_score": 0.200,
        }

        with patch("weather_edge.signals.resolver.resolve_pending_signals", new_callable=AsyncMock) as mock_resolve, \
             patch("weather_edge.signals.tracker.SignalTracker.get_performance_summary", new_callable=AsyncMock) as mock_perf, \
             patch("weather_edge.signals.tracker.SignalTracker._ensure_db", new_callable=AsyncMock):
            mock_resolve.return_value = []
            mock_perf.return_value = summary
            result = runner.invoke(app, ["resolve"])

        assert result.exit_code == 0
        assert "No markets newly resolved" in result.output
