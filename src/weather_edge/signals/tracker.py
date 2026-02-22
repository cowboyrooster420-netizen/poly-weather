"""Historical signal and outcome logger with PostgreSQL and SQLite backends."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import aiosqlite

from weather_edge.config import get_settings
from weather_edge.signals.models import Signal

_SQLITE_CREATE_TABLE = """
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

_PG_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    market_id TEXT NOT NULL,
    question TEXT,
    market_type TEXT,
    location TEXT,
    model_prob DOUBLE PRECISION NOT NULL,
    market_prob DOUBLE PRECISION NOT NULL,
    edge DOUBLE PRECISION NOT NULL,
    kelly_fraction DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    direction TEXT,
    lead_time_hours DOUBLE PRECISION,
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

_CREATE_CALIBRATION = """
CREATE TABLE IF NOT EXISTS calibration (
    key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class SignalTracker:
    """Signal tracker with PostgreSQL (via asyncpg) or SQLite (via aiosqlite) backend."""

    def __init__(self) -> None:
        settings = get_settings()
        self._database_url = settings.database_url
        self._use_pg = bool(self._database_url)
        self._db_path = settings.db_path
        self._pool = None  # asyncpg pool, created lazily
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self):
        """Get or create the asyncpg connection pool."""
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    import asyncpg
                    self._pool = await asyncpg.create_pool(
                        self._database_url, min_size=1, max_size=5,
                    )
        return self._pool

    async def close(self) -> None:
        """Close the connection pool, if open."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        if self._use_pg:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(_PG_CREATE_TABLE)
                await conn.execute(_CREATE_INDEX)
                await conn.execute(_CREATE_CALIBRATION)
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(str(self._db_path)) as db:
                await db.execute(_SQLITE_CREATE_TABLE)
                await db.execute(_CREATE_INDEX)
                await db.execute(_CREATE_CALIBRATION)
                await db.commit()

    async def log_signal(self, signal: Signal) -> int:
        """Log a signal to the database. Returns the row ID."""
        await self._ensure_db()
        params = (
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
        )
        if self._use_pg:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """INSERT INTO signals
                       (market_id, question, market_type, location,
                        model_prob, market_prob, edge, kelly_fraction,
                        confidence, direction, lead_time_hours,
                        sources, details, timestamp)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                       RETURNING id""",
                    *params,
                )
                return row["id"]
        else:
            async with aiosqlite.connect(str(self._db_path)) as db:
                cursor = await db.execute(
                    """INSERT INTO signals
                       (market_id, question, market_type, location,
                        model_prob, market_prob, edge, kelly_fraction,
                        confidence, direction, lead_time_hours,
                        sources, details, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    params,
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
        if self._use_pg:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """UPDATE signals
                       SET outcome = $1, resolved_at = $2
                       WHERE market_id = $3 AND outcome IS NULL""",
                    outcome, resolved_at, market_id,
                )
                # asyncpg returns e.g. "UPDATE 3"
                return int(result.split()[-1])
        else:
            async with aiosqlite.connect(str(self._db_path)) as db:
                cursor = await db.execute(
                    """UPDATE signals
                       SET outcome = ?, resolved_at = ?
                       WHERE market_id = ? AND outcome IS NULL""",
                    (outcome, resolved_at, market_id),
                )
                await db.commit()
                return cursor.rowcount

    async def get_unresolved_market_ids(self) -> list[tuple[str, str]]:
        """Get distinct market IDs that have no outcome yet.

        Returns:
            List of (market_id, question) tuples.
        """
        await self._ensure_db()
        if self._use_pg:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT DISTINCT market_id, question FROM signals WHERE outcome IS NULL"
                )
                return [(row["market_id"], row["question"]) for row in rows]
        else:
            async with aiosqlite.connect(str(self._db_path)) as db:
                cursor = await db.execute(
                    "SELECT DISTINCT market_id, question FROM signals WHERE outcome IS NULL"
                )
                rows = await cursor.fetchall()
                return [(row[0], row[1]) for row in rows]

    async def get_calibration_data(self) -> list[tuple[float, int]]:
        """Get (model_prob, outcome) pairs for Platt scaling calibration.

        Only returns signals where outcome has been backfilled.
        """
        await self._ensure_db()
        if self._use_pg:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT model_prob, outcome FROM signals WHERE outcome IS NOT NULL"
                )
                return [(row["model_prob"], row["outcome"]) for row in rows]
        else:
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
        if self._use_pg:
            return await self._get_performance_summary_pg()
        else:
            return await self._get_performance_summary_sqlite()

    async def _get_performance_summary_pg(self) -> dict:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM signals")
            resolved = await conn.fetchval(
                "SELECT COUNT(*) FROM signals WHERE outcome IS NOT NULL"
            )
            wins = await conn.fetchval(
                """SELECT COUNT(*) FROM signals
                   WHERE outcome IS NOT NULL
                   AND ((direction = 'YES' AND outcome = 1)
                        OR (direction = 'NO' AND outcome = 0))"""
            )
            avg_edge = await conn.fetchval(
                "SELECT AVG(ABS(edge)) FROM signals"
            )
            brier = await conn.fetchval(
                """SELECT AVG((model_prob - outcome) * (model_prob - outcome))
                   FROM signals WHERE outcome IS NOT NULL"""
            )
            return {
                "total_signals": total,
                "resolved": resolved,
                "wins": wins,
                "win_rate": wins / resolved if resolved > 0 else None,
                "avg_abs_edge": avg_edge,
                "brier_score": brier,
            }

    async def _get_performance_summary_sqlite(self) -> dict:
        async with aiosqlite.connect(str(self._db_path)) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("SELECT COUNT(*) as total FROM signals")
            total = (await cursor.fetchone())["total"]

            cursor = await db.execute(
                "SELECT COUNT(*) as resolved FROM signals WHERE outcome IS NOT NULL"
            )
            resolved = (await cursor.fetchone())["resolved"]

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

            cursor = await db.execute(
                """SELECT AVG((model_prob - outcome) * (model_prob - outcome))
                   as brier FROM signals WHERE outcome IS NOT NULL"""
            )
            brier = (await cursor.fetchone())["brier"]

            return {
                "total_signals": total,
                "resolved": resolved,
                "wins": wins,
                "win_rate": wins / resolved if resolved > 0 else None,
                "avg_abs_edge": avg_edge,
                "brier_score": brier,
            }

    async def get_resolved_signals(
        self, market_type: str | None = None,
    ) -> list[dict]:
        """Get all resolved signals with full detail.

        Args:
            market_type: Optional filter (e.g. "temperature"). None = all types.

        Returns:
            List of dicts with all signal fields plus outcome and resolved_at.
        """
        await self._ensure_db()

        # Deduplicate: keep only the latest signal per market_id (closest to resolution).
        base_query = """
            SELECT s.market_id, s.question, s.market_type, s.location,
                   s.model_prob, s.market_prob, s.edge, s.confidence,
                   s.direction, s.lead_time_hours, s.timestamp,
                   s.outcome, s.resolved_at
            FROM signals s
            INNER JOIN (
                SELECT market_id, MAX(id) AS max_id
                FROM signals
                WHERE outcome IS NOT NULL
                GROUP BY market_id
            ) latest ON s.id = latest.max_id
        """

        if self._use_pg:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                if market_type:
                    rows = await conn.fetch(
                        base_query + " WHERE s.market_type = $1 ORDER BY s.resolved_at DESC, s.id DESC",
                        market_type,
                    )
                else:
                    rows = await conn.fetch(
                        base_query + " ORDER BY s.resolved_at DESC, s.id DESC"
                    )
                return [dict(row) for row in rows]
        else:
            async with aiosqlite.connect(str(self._db_path)) as db:
                db.row_factory = aiosqlite.Row
                if market_type:
                    cursor = await db.execute(
                        base_query + " WHERE s.market_type = ? ORDER BY s.resolved_at DESC, s.id DESC",
                        (market_type,),
                    )
                else:
                    cursor = await db.execute(
                        base_query + " ORDER BY s.resolved_at DESC, s.id DESC"
                    )
                rows = await cursor.fetchall()
                return [
                    {
                        "market_id": row["market_id"],
                        "question": row["question"],
                        "market_type": row["market_type"],
                        "location": row["location"],
                        "model_prob": row["model_prob"],
                        "market_prob": row["market_prob"],
                        "edge": row["edge"],
                        "confidence": row["confidence"],
                        "direction": row["direction"],
                        "lead_time_hours": row["lead_time_hours"],
                        "timestamp": row["timestamp"],
                        "outcome": row["outcome"],
                        "resolved_at": row["resolved_at"],
                    }
                    for row in rows
                ]

    async def get_signal_direction(self, market_id: str) -> str | None:
        """Get the direction of the first signal logged for a market.

        Returns:
            "YES" or "NO", or None if no signal found.
        """
        await self._ensure_db()
        if self._use_pg:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT direction FROM signals WHERE market_id = $1 ORDER BY id LIMIT 1",
                    market_id,
                )
                return row["direction"] if row else None
        else:
            async with aiosqlite.connect(str(self._db_path)) as db:
                cursor = await db.execute(
                    "SELECT direction FROM signals WHERE market_id = ? ORDER BY id LIMIT 1",
                    (market_id,),
                )
                row = await cursor.fetchone()
                return row[0] if row else None

    async def save_calibration(self, key: str, data: str) -> None:
        """Upsert a calibration row (key → JSON blob)."""
        now = datetime.now(timezone.utc).isoformat()
        await self._ensure_db()
        if self._use_pg:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO calibration (key, data, updated_at)
                       VALUES ($1, $2, $3)
                       ON CONFLICT (key) DO UPDATE
                       SET data = EXCLUDED.data, updated_at = EXCLUDED.updated_at""",
                    key, data, now,
                )
        else:
            async with aiosqlite.connect(str(self._db_path)) as db:
                await db.execute(
                    """INSERT INTO calibration (key, data, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT (key) DO UPDATE
                       SET data = excluded.data, updated_at = excluded.updated_at""",
                    (key, data, now),
                )
                await db.commit()

    async def load_calibration(self, key: str) -> str | None:
        """Read the JSON blob for a calibration key. Returns None if not found."""
        await self._ensure_db()
        if self._use_pg:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT data FROM calibration WHERE key = $1", key,
                )
                return row["data"] if row else None
        else:
            async with aiosqlite.connect(str(self._db_path)) as db:
                cursor = await db.execute(
                    "SELECT data FROM calibration WHERE key = ?", (key,),
                )
                row = await cursor.fetchone()
                return row[0] if row else None
