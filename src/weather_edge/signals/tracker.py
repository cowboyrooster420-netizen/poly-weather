"""SQLite historical signal and outcome logger."""

from __future__ import annotations

import aiosqlite

from weather_edge.config import get_settings
from weather_edge.signals.models import Signal

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    question TEXT,
    market_type TEXT,
    location TEXT,
    model_prob REAL NOT NULL,
    market_prob REAL NOT NULL,
    edge REAL NOT NULL,
    kelly_fraction REAL,
    confidence REAL,
    direction TEXT,
    lead_time_hours REAL,
    sources TEXT,
    details TEXT,
    timestamp TEXT NOT NULL,
    outcome INTEGER,  -- NULL until resolved, 1 = YES won, 0 = NO won
    resolved_at TEXT
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_signals_market_id ON signals(market_id);
"""


class SignalTracker:
    """SQLite-backed signal tracker for calibration training data."""

    def __init__(self) -> None:
        self._db_path = get_settings().db_path

    async def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute(_CREATE_TABLE)
            await db.execute(_CREATE_INDEX)
            await db.commit()

    async def log_signal(self, signal: Signal) -> int:
        """Log a signal to the database. Returns the row ID."""
        await self._ensure_db()
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                """INSERT INTO signals
                   (market_id, question, market_type, location,
                    model_prob, market_prob, edge, kelly_fraction,
                    confidence, direction, lead_time_hours,
                    sources, details, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal.market_id,
                    signal.question,
                    signal.market_type,
                    signal.location,
                    signal.model_prob,
                    signal.market_prob,
                    signal.edge,
                    signal.kelly_fraction,
                    signal.confidence,
                    signal.direction,
                    signal.lead_time_hours,
                    ",".join(signal.sources),
                    signal.details,
                    signal.timestamp.isoformat(),
                ),
            )
            await db.commit()
            return cursor.lastrowid

    async def log_signals(self, signals: list[Signal]) -> list[int]:
        """Log multiple signals. Returns list of row IDs."""
        ids = []
        for signal in signals:
            row_id = await self.log_signal(signal)
            ids.append(row_id)
        return ids

    async def backfill_outcome(
        self, market_id: str, outcome: int, resolved_at: str | None = None,
    ) -> int:
        """Backfill the outcome for all signals matching a market_id.

        Args:
            market_id: The market ID to update
            outcome: 1 if YES won, 0 if NO won
            resolved_at: ISO timestamp of resolution

        Returns:
            Number of rows updated
        """
        await self._ensure_db()
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                """UPDATE signals
                   SET outcome = ?, resolved_at = ?
                   WHERE market_id = ? AND outcome IS NULL""",
                (outcome, resolved_at, market_id),
            )
            await db.commit()
            return cursor.rowcount

    async def get_calibration_data(self) -> list[tuple[float, int]]:
        """Get (model_prob, outcome) pairs for Platt scaling calibration.

        Only returns signals where outcome has been backfilled.
        """
        await self._ensure_db()
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT model_prob, outcome FROM signals WHERE outcome IS NOT NULL"
            )
            rows = await cursor.fetchall()
            return [(row["model_prob"], row["outcome"]) for row in rows]

    async def get_performance_summary(self) -> dict:
        """Get a summary of signal performance."""
        await self._ensure_db()
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("SELECT COUNT(*) as total FROM signals")
            total = (await cursor.fetchone())["total"]

            cursor = await db.execute(
                "SELECT COUNT(*) as resolved FROM signals WHERE outcome IS NOT NULL"
            )
            resolved = (await cursor.fetchone())["resolved"]

            # Win rate: signals where direction matches outcome
            cursor = await db.execute(
                """SELECT COUNT(*) as wins FROM signals
                   WHERE outcome IS NOT NULL
                   AND ((direction = 'YES' AND outcome = 1)
                        OR (direction = 'NO' AND outcome = 0))"""
            )
            wins = (await cursor.fetchone())["wins"]

            cursor = await db.execute(
                "SELECT AVG(ABS(edge)) as avg_edge FROM signals"
            )
            avg_edge = (await cursor.fetchone())["avg_edge"]

            return {
                "total_signals": total,
                "resolved": resolved,
                "wins": wins,
                "win_rate": wins / resolved if resolved > 0 else None,
                "avg_abs_edge": avg_edge,
            }
