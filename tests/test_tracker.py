"""Tests for SQLite signal tracker."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from weather_edge.signals.models import Signal
from weather_edge.signals.tracker import SignalTracker


@pytest.fixture
def tmp_db():
    """Create a temporary database file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_signals.db"
        yield db_path


@pytest.fixture
def tracker(tmp_db):
    """Create a tracker with a temporary database."""
    with patch("weather_edge.signals.tracker.get_settings") as mock_settings:
        settings = mock_settings.return_value
        settings.db_path = tmp_db
        settings.database_url = ""
        t = SignalTracker()
        yield t


def _make_signal(market_id="test-001", model_prob=0.55, market_prob=0.35, edge=0.20):
    return Signal(
        market_id=market_id,
        question="Will it be hot?",
        market_type="temperature",
        location="Phoenix, AZ",
        model_prob=model_prob,
        market_prob=market_prob,
        edge=edge,
        kelly_fraction=0.08,
        confidence=0.85,
        direction="YES" if edge > 0 else "NO",
        lead_time_hours=24.0,
        sources=["ECMWF", "GFS"],
        details="Test signal",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_log_signal(tracker):
    """Test logging a single signal."""
    signal = _make_signal()
    row_id = await tracker.log_signal(signal)
    assert row_id == 1


@pytest.mark.asyncio
async def test_log_multiple_signals(tracker):
    """Test logging multiple signals."""
    signals = [_make_signal(f"m{i}") for i in range(5)]
    ids = await tracker.log_signals(signals)
    assert len(ids) == 5
    assert ids == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_backfill_outcome(tracker):
    """Test backfilling an outcome for logged signals."""
    signal = _make_signal()
    await tracker.log_signal(signal)

    updated = await tracker.backfill_outcome(
        "test-001", outcome=1, resolved_at="2025-07-05T00:00:00Z",
    )
    assert updated == 1


@pytest.mark.asyncio
async def test_backfill_no_match(tracker):
    """Test backfilling for a non-existent market_id."""
    signal = _make_signal()
    await tracker.log_signal(signal)

    updated = await tracker.backfill_outcome("nonexistent-id", outcome=0)
    assert updated == 0


@pytest.mark.asyncio
async def test_backfill_idempotent(tracker):
    """Test that backfill only updates NULL outcomes."""
    signal = _make_signal()
    await tracker.log_signal(signal)

    # First backfill
    await tracker.backfill_outcome("test-001", outcome=1)
    # Second backfill should not update (already resolved)
    updated = await tracker.backfill_outcome("test-001", outcome=0)
    assert updated == 0


@pytest.mark.asyncio
async def test_get_calibration_data(tracker):
    """Test getting calibration data (predictions + outcomes)."""
    # Log signals
    for i in range(10):
        signal = _make_signal(f"m{i}", model_prob=0.1 * (i + 1))
        await tracker.log_signal(signal)

    # Backfill some outcomes
    for i in range(5):
        await tracker.backfill_outcome(f"m{i}", outcome=1 if i % 2 == 0 else 0)

    data = await tracker.get_calibration_data()
    assert len(data) == 5  # Only resolved signals
    for prob, outcome in data:
        assert 0 <= prob <= 1
        assert outcome in (0, 1)


@pytest.mark.asyncio
async def test_get_calibration_data_empty(tracker):
    """Test calibration data when no outcomes are backfilled."""
    signal = _make_signal()
    await tracker.log_signal(signal)

    data = await tracker.get_calibration_data()
    assert len(data) == 0


@pytest.mark.asyncio
async def test_performance_summary(tracker):
    """Test performance summary calculation."""
    # Log some signals
    signals = [
        _make_signal("win-yes", model_prob=0.7, market_prob=0.5, edge=0.2),
        _make_signal("win-no", model_prob=0.2, market_prob=0.5, edge=-0.3),
        _make_signal("lose-yes", model_prob=0.7, market_prob=0.5, edge=0.2),
        _make_signal("unresolved", model_prob=0.6, market_prob=0.4, edge=0.2),
    ]
    for s in signals:
        await tracker.log_signal(s)

    # Backfill outcomes
    await tracker.backfill_outcome("win-yes", outcome=1)   # YES won, we bet YES → win
    await tracker.backfill_outcome("win-no", outcome=0)     # NO won, we bet NO → win
    await tracker.backfill_outcome("lose-yes", outcome=0)   # NO won, we bet YES → loss

    summary = await tracker.get_performance_summary()

    assert summary["total_signals"] == 4
    assert summary["resolved"] == 3
    assert summary["wins"] == 2
    assert abs(summary["win_rate"] - 2 / 3) < 0.01


@pytest.mark.asyncio
async def test_performance_summary_empty(tracker):
    """Test performance summary with no signals."""
    summary = await tracker.get_performance_summary()

    assert summary["total_signals"] == 0
    assert summary["resolved"] == 0
    assert summary["win_rate"] is None


@pytest.mark.asyncio
async def test_db_created_on_first_use(tmp_db):
    """Test that the database file is created on first use."""
    assert not tmp_db.exists()

    with patch("weather_edge.signals.tracker.get_settings") as mock_settings:
        mock_settings.return_value.db_path = tmp_db
        mock_settings.return_value.database_url = ""
        tracker = SignalTracker()
    signal = _make_signal()
    await tracker.log_signal(signal)

    assert tmp_db.exists()


@pytest.mark.asyncio
async def test_db_parent_dirs_created(tmp_db):
    """Test that parent directories are created if needed."""
    deep_path = tmp_db.parent / "subdir" / "deep" / "signals.db"

    with patch("weather_edge.signals.tracker.get_settings") as mock_settings:
        mock_settings.return_value.db_path = deep_path
        mock_settings.return_value.database_url = ""
        tracker = SignalTracker()
    signal = _make_signal()
    await tracker.log_signal(signal)

    assert deep_path.exists()


@pytest.mark.asyncio
async def test_get_unresolved_market_ids(tracker):
    """Test getting unresolved market IDs."""
    # Log signals for 3 different markets
    await tracker.log_signal(_make_signal("m1"))
    await tracker.log_signal(_make_signal("m2"))
    await tracker.log_signal(_make_signal("m3"))

    # Resolve one
    await tracker.backfill_outcome("m2", outcome=1)

    unresolved = await tracker.get_unresolved_market_ids()
    market_ids = [mid for mid, _ in unresolved]

    assert len(unresolved) == 2
    assert "m1" in market_ids
    assert "m3" in market_ids
    assert "m2" not in market_ids


@pytest.mark.asyncio
async def test_get_unresolved_market_ids_empty(tracker):
    """Test that no unresolved IDs are returned when all resolved."""
    await tracker.log_signal(_make_signal("m1"))
    await tracker.backfill_outcome("m1", outcome=0)

    unresolved = await tracker.get_unresolved_market_ids()
    assert len(unresolved) == 0


@pytest.mark.asyncio
async def test_get_unresolved_market_ids_distinct(tracker):
    """Test that duplicate market IDs are deduplicated."""
    # Log two signals for the same market
    await tracker.log_signal(_make_signal("m1"))
    await tracker.log_signal(_make_signal("m1"))

    unresolved = await tracker.get_unresolved_market_ids()
    assert len(unresolved) == 1


@pytest.mark.asyncio
async def test_brier_score_in_summary(tracker):
    """Test that Brier score is computed correctly."""
    # Log signals with known model_prob
    s1 = _make_signal("m1", model_prob=0.9, market_prob=0.5, edge=0.4)
    s2 = _make_signal("m2", model_prob=0.2, market_prob=0.5, edge=-0.3)
    await tracker.log_signal(s1)
    await tracker.log_signal(s2)

    # m1: model_prob=0.9, outcome=1 → (0.9-1)^2 = 0.01
    # m2: model_prob=0.2, outcome=0 → (0.2-0)^2 = 0.04
    # Brier = (0.01 + 0.04) / 2 = 0.025
    await tracker.backfill_outcome("m1", outcome=1)
    await tracker.backfill_outcome("m2", outcome=0)

    summary = await tracker.get_performance_summary()
    assert summary["brier_score"] is not None
    assert abs(summary["brier_score"] - 0.025) < 0.001


@pytest.mark.asyncio
async def test_brier_score_none_when_no_resolved(tracker):
    """Test that Brier score is None when no outcomes are resolved."""
    await tracker.log_signal(_make_signal("m1"))

    summary = await tracker.get_performance_summary()
    assert summary["brier_score"] is None
