"""Tests for the auto-resolver."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from weather_edge.signals.models import Signal
from weather_edge.signals.resolver import _parse_outcome, resolve_pending_signals
from weather_edge.signals.tracker import SignalTracker


def _make_signal(market_id="m1", direction="YES", model_prob=0.70, market_prob=0.50):
    edge = model_prob - market_prob
    return Signal(
        market_id=market_id,
        question=f"Question for {market_id}",
        market_type="temperature",
        location="Atlanta, GA",
        model_prob=model_prob,
        market_prob=market_prob,
        edge=edge,
        kelly_fraction=0.05,
        confidence=0.80,
        direction=direction,
        lead_time_hours=24.0,
        sources=["ECMWF"],
        details="test",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def tracker():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch("weather_edge.signals.tracker.get_settings") as mock_settings:
            settings = mock_settings.return_value
            settings.db_path = db_path
            t = SignalTracker()
            t._db_path = db_path
            yield t


class TestParseOutcome:
    def test_yes_won(self):
        raw = {"outcomePrices": '["1","0"]'}
        assert _parse_outcome(raw) == 1

    def test_no_won(self):
        raw = {"outcomePrices": '["0","1"]'}
        assert _parse_outcome(raw) == 0

    def test_yes_won_near_one(self):
        raw = {"outcomePrices": '["0.995","0.005"]'}
        assert _parse_outcome(raw) == 1

    def test_no_won_near_one(self):
        raw = {"outcomePrices": '["0.003","0.997"]'}
        assert _parse_outcome(raw) == 0

    def test_ambiguous(self):
        raw = {"outcomePrices": '["0.6","0.4"]'}
        assert _parse_outcome(raw) is None

    def test_empty_prices(self):
        raw = {"outcomePrices": ""}
        assert _parse_outcome(raw) is None

    def test_missing_prices(self):
        raw = {}
        assert _parse_outcome(raw) is None

    def test_malformed_json(self):
        raw = {"outcomePrices": "not-json"}
        assert _parse_outcome(raw) is None

    def test_list_input(self):
        """outcomePrices may already be a list (not a JSON string)."""
        raw = {"outcomePrices": ["1", "0"]}
        assert _parse_outcome(raw) == 1


@pytest.mark.asyncio
async def test_resolve_yes_won(tracker):
    """outcomePrices=["1","0"] → outcome=1, direction YES is correct."""
    await tracker.log_signal(_make_signal("m1", direction="YES"))

    closed_market = {
        "id": "m1",
        "closed": True,
        "outcomePrices": '["1","0"]',
        "closedTime": "2025-07-05T00:00:00Z",
    }

    with patch("weather_edge.signals.resolver.SignalTracker", return_value=tracker), \
         patch("weather_edge.signals.resolver.fetch_market_by_id", new_callable=AsyncMock) as mock_fetch, \
         patch("weather_edge.signals.resolver.get_settings") as mock_settings:
        mock_settings.return_value.telegram_enabled = False
        mock_fetch.return_value = closed_market

        results = await resolve_pending_signals()

    assert len(results) == 1
    assert results[0]["outcome"] == 1
    assert results[0]["direction"] == "YES"
    assert results[0]["correct"] is True


@pytest.mark.asyncio
async def test_resolve_no_won(tracker):
    """outcomePrices=["0","1"] → outcome=0, direction NO is correct."""
    await tracker.log_signal(_make_signal("m1", direction="NO", model_prob=0.30))

    closed_market = {
        "id": "m1",
        "closed": True,
        "outcomePrices": '["0","1"]',
        "closedTime": "2025-07-05T00:00:00Z",
    }

    with patch("weather_edge.signals.resolver.SignalTracker", return_value=tracker), \
         patch("weather_edge.signals.resolver.fetch_market_by_id", new_callable=AsyncMock) as mock_fetch, \
         patch("weather_edge.signals.resolver.get_settings") as mock_settings:
        mock_settings.return_value.telegram_enabled = False
        mock_fetch.return_value = closed_market

        results = await resolve_pending_signals()

    assert len(results) == 1
    assert results[0]["outcome"] == 0
    assert results[0]["direction"] == "NO"
    assert results[0]["correct"] is True


@pytest.mark.asyncio
async def test_resolve_still_open(tracker):
    """Market with closed=false should not be resolved."""
    await tracker.log_signal(_make_signal("m1"))

    open_market = {
        "id": "m1",
        "closed": False,
        "outcomePrices": '["0.55","0.45"]',
    }

    with patch("weather_edge.signals.resolver.SignalTracker", return_value=tracker), \
         patch("weather_edge.signals.resolver.fetch_market_by_id", new_callable=AsyncMock) as mock_fetch, \
         patch("weather_edge.signals.resolver.get_settings") as mock_settings:
        mock_settings.return_value.telegram_enabled = False
        mock_fetch.return_value = open_market

        results = await resolve_pending_signals()

    assert len(results) == 0


@pytest.mark.asyncio
async def test_resolve_ambiguous(tracker):
    """Prices not near 0/1 should be skipped."""
    await tracker.log_signal(_make_signal("m1"))

    ambiguous_market = {
        "id": "m1",
        "closed": True,
        "outcomePrices": '["0.6","0.4"]',
    }

    with patch("weather_edge.signals.resolver.SignalTracker", return_value=tracker), \
         patch("weather_edge.signals.resolver.fetch_market_by_id", new_callable=AsyncMock) as mock_fetch, \
         patch("weather_edge.signals.resolver.get_settings") as mock_settings:
        mock_settings.return_value.telegram_enabled = False
        mock_fetch.return_value = ambiguous_market

        results = await resolve_pending_signals()

    assert len(results) == 0


@pytest.mark.asyncio
async def test_resolve_no_pending(tracker):
    """No unresolved signals should mean no API calls."""
    with patch("weather_edge.signals.resolver.SignalTracker", return_value=tracker), \
         patch("weather_edge.signals.resolver.fetch_market_by_id", new_callable=AsyncMock) as mock_fetch, \
         patch("weather_edge.signals.resolver.get_settings") as mock_settings:
        mock_settings.return_value.telegram_enabled = False

        results = await resolve_pending_signals()

    assert len(results) == 0
    mock_fetch.assert_not_called()
