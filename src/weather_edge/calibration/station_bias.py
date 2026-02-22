"""Per-station bias correction for WU personal weather stations.

Computes, stores, and applies bias offsets that capture the systematic
difference between what a WU station reports and what ERA5 reanalysis
(which our ensemble models are calibrated against) shows for the same
location and time.

    bias = WU_reading - ERA5_reanalysis

A positive bias means the station reads warm relative to reanalysis.
We shift our ensemble members by this amount so our probability estimates
reflect what the specific station will actually report.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

from weather_edge.config import get_settings

logger = logging.getLogger(__name__)

# Bundled defaults ship with zero biases so the system works without calibration.
_BUNDLED_DEFAULTS = Path(__file__).resolve().parent.parent / "data" / "station_biases.json"


@dataclass(frozen=True)
class StationBias:
    """Bias statistics for a single station."""

    station_id: str
    city: str
    high_bias_c: float  # mean(WU_high - OM_max); positive = station reads warm
    low_bias_c: float   # mean(WU_low - OM_min)
    mean_bias_c: float  # (high_bias_c + low_bias_c) / 2
    high_std_c: float = 0.0
    low_std_c: float = 0.0
    n_days: int = 0


class SkyCondition(Enum):
    """Cloud cover classification buckets."""

    CLEAR = "clear"
    PARTLY_CLOUDY = "partly"
    OVERCAST = "overcast"


@dataclass(frozen=True)
class ConditionBias:
    """Bias statistics for a single sky-condition bucket."""

    condition: SkyCondition
    high_bias_c: float
    low_bias_c: float
    mean_bias_c: float
    high_std_c: float = 0.0
    low_std_c: float = 0.0
    n_days: int = 0


@dataclass(frozen=True)
class StationBiasV2:
    """Bias statistics with per-condition stratification."""

    station_id: str
    city: str
    high_bias_c: float
    low_bias_c: float
    mean_bias_c: float
    high_std_c: float = 0.0
    low_std_c: float = 0.0
    n_days: int = 0
    condition_biases: tuple[ConditionBias, ...] = ()


# Minimum days in a bucket before we trust its bias estimate.
_MIN_BUCKET_DAYS = 10


def classify_sky_condition(cloud_cover_pct: float) -> SkyCondition:
    """Classify cloud cover percentage into a sky-condition bucket.

    - Clear: cloud_cover < 25%
    - Partly cloudy: 25% <= cloud_cover < 75%
    - Overcast: cloud_cover >= 75%
    """
    if cloud_cover_pct < 25.0:
        return SkyCondition.CLEAR
    elif cloud_cover_pct < 75.0:
        return SkyCondition.PARTLY_CLOUDY
    else:
        return SkyCondition.OVERCAST


# Module-level cache so we only load from disk once.
_cache: dict[str, StationBiasV2] | None = None

_CONDITION_KEY_MAP = {
    "clear": SkyCondition.CLEAR,
    "partly": SkyCondition.PARTLY_CLOUDY,
    "overcast": SkyCondition.OVERCAST,
}


def _parse_biases_json(data: dict) -> dict[str, StationBiasV2]:
    """Parse a biases JSON dict into StationBiasV2 objects."""
    biases: dict[str, StationBiasV2] = {}
    for station_id, info in data.get("stations", {}).items():
        condition_biases: list[ConditionBias] = []
        conditions_data = info.get("conditions", {})
        for key, cond_info in conditions_data.items():
            sky_cond = _CONDITION_KEY_MAP.get(key)
            if sky_cond is None:
                continue
            condition_biases.append(ConditionBias(
                condition=sky_cond,
                high_bias_c=cond_info.get("high_bias_c", 0.0),
                low_bias_c=cond_info.get("low_bias_c", 0.0),
                mean_bias_c=cond_info.get("mean_bias_c", 0.0),
                high_std_c=cond_info.get("high_std_c", 0.0),
                low_std_c=cond_info.get("low_std_c", 0.0),
                n_days=cond_info.get("n_days", 0),
            ))

        biases[station_id] = StationBiasV2(
            station_id=station_id,
            city=info.get("city", ""),
            high_bias_c=info.get("high_bias_c", 0.0),
            low_bias_c=info.get("low_bias_c", 0.0),
            mean_bias_c=info.get("mean_bias_c", 0.0),
            high_std_c=info.get("high_std_c", 0.0),
            low_std_c=info.get("low_std_c", 0.0),
            n_days=info.get("n_days", 0),
            condition_biases=tuple(condition_biases),
        )
    return biases


def _run_async(coro):
    """Run an async coroutine from sync context, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside an async context — create a new thread to run the coroutine.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=15)
    else:
        return asyncio.run(coro)


def _try_load_from_db() -> dict[str, StationBiasV2] | None:
    """Try loading biases from the DB. Returns None if unavailable."""
    settings = get_settings()
    if not settings.database_url:
        return None

    try:
        from weather_edge.signals.tracker import SignalTracker

        async def _fetch():
            tracker = SignalTracker()
            try:
                return await tracker.load_calibration("station_biases")
            finally:
                await tracker.close()

        raw = _run_async(_fetch())
        if raw is None:
            return None

        data = json.loads(raw)
        biases = _parse_biases_json(data)
        logger.info("Loaded biases for %d stations from database", len(biases))
        return biases
    except Exception as exc:
        logger.warning("Failed to load biases from database: %s", exc)
        return None


def _load_biases() -> dict[str, StationBiasV2]:
    """Load biases with fallback chain: DB → user file → bundled defaults → empty.

    Handles both v1 (no conditions) and v2 (with conditions) JSON.
    v1 files are loaded with empty condition_biases.
    """
    # 1. Try PostgreSQL calibration table
    db_biases = _try_load_from_db()
    if db_biases is not None:
        return db_biases

    # 2. Try local JSON file → bundled defaults
    settings = get_settings()
    user_path = settings.station_bias_path

    path: Path | None = None
    if user_path.exists():
        path = user_path
    elif _BUNDLED_DEFAULTS.exists():
        path = _BUNDLED_DEFAULTS
    else:
        logger.warning("No station bias file found; returning empty biases")
        return {}

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read station bias file %s: %s", path, exc)
        return {}

    biases = _parse_biases_json(data)
    source = "user" if path == user_path else "bundled"
    logger.info("Loaded biases for %d stations from %s (%s)", len(biases), path, source)
    return biases


def load_biases(*, force: bool = False) -> dict[str, StationBiasV2]:
    """Return cached station biases, loading from disk on first call."""
    global _cache
    if _cache is None or force:
        _cache = _load_biases()
    return _cache


def get_station_bias(station_id: str, aggregation: str | None) -> float:
    """Return bias offset in °C for the given station and aggregation.

    Args:
        station_id: WU station ID (e.g. "KGAHAPEV1")
        aggregation: "max" for daily-high markets, "min" for daily-low,
                     None for point-in-time (uses mean_bias_c)

    Returns:
        Bias offset in °C.  Positive means station reads warm.
        Returns 0.0 for unknown stations or when bias correction is disabled.
    """
    settings = get_settings()
    if not settings.station_bias_enabled:
        return 0.0

    biases = load_biases()
    bias = biases.get(station_id)
    if bias is None:
        return 0.0

    if aggregation == "max":
        return bias.high_bias_c
    elif aggregation == "min":
        return bias.low_bias_c
    else:
        return bias.mean_bias_c


def get_station_bias_for_condition(
    station_id: str,
    aggregation: str | None,
    cloud_cover_pct: float | None = None,
) -> float:
    """Return condition-dependent bias offset in °C.

    Fallback chain:
        condition bias (bucket n_days >= _MIN_BUCKET_DAYS)
        → global bias (station found)
        → 0.0 (station unknown or bias disabled)
    """
    settings = get_settings()
    if not settings.station_bias_enabled:
        return 0.0

    biases = load_biases()
    bias = biases.get(station_id)
    if bias is None:
        return 0.0

    # Try condition-specific bias if cloud cover is provided
    if cloud_cover_pct is not None and bias.condition_biases:
        condition = classify_sky_condition(cloud_cover_pct)
        for cb in bias.condition_biases:
            if cb.condition == condition and cb.n_days >= _MIN_BUCKET_DAYS:
                if aggregation == "max":
                    return cb.high_bias_c
                elif aggregation == "min":
                    return cb.low_bias_c
                else:
                    return cb.mean_bias_c

    # Fall back to global bias
    if aggregation == "max":
        return bias.high_bias_c
    elif aggregation == "min":
        return bias.low_bias_c
    else:
        return bias.mean_bias_c


def _biases_to_json(biases: dict[str, StationBiasV2], training_days: int = 90) -> str:
    """Serialize biases to a JSON string."""
    stations_data: dict[str, dict] = {}
    for sid, b in biases.items():
        station_entry: dict = {
            "city": b.city,
            "high_bias_c": round(b.high_bias_c, 3),
            "low_bias_c": round(b.low_bias_c, 3),
            "mean_bias_c": round(b.mean_bias_c, 3),
            "high_std_c": round(b.high_std_c, 3),
            "low_std_c": round(b.low_std_c, 3),
            "n_days": b.n_days,
        }
        if b.condition_biases:
            conditions: dict[str, dict] = {}
            for cb in b.condition_biases:
                conditions[cb.condition.value] = {
                    "high_bias_c": round(cb.high_bias_c, 3),
                    "low_bias_c": round(cb.low_bias_c, 3),
                    "mean_bias_c": round(cb.mean_bias_c, 3),
                    "high_std_c": round(cb.high_std_c, 3),
                    "low_std_c": round(cb.low_std_c, 3),
                    "n_days": cb.n_days,
                }
            station_entry["conditions"] = conditions
        stations_data[sid] = station_entry

    data = {
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_days": training_days,
        "stations": stations_data,
    }
    return json.dumps(data, indent=2)


def save_biases(biases: dict[str, StationBiasV2], *, training_days: int = 90) -> Path:
    """Persist biases to the user's station_bias_path (and DB when available).

    Returns the path written to.
    """
    settings = get_settings()
    path = settings.station_bias_path
    path.parent.mkdir(parents=True, exist_ok=True)

    json_str = _biases_to_json(biases, training_days)

    path.write_text(json_str + "\n")
    logger.info("Saved biases for %d stations to %s", len(biases), path)

    # Also persist to database when DATABASE_URL is set.
    if settings.database_url:
        try:
            from weather_edge.signals.tracker import SignalTracker

            async def _save():
                tracker = SignalTracker()
                try:
                    await tracker.save_calibration("station_biases", json_str)
                finally:
                    await tracker.close()

            _run_async(_save())
            logger.info("Saved biases to database")
        except Exception as exc:
            logger.warning("Failed to save biases to database: %s", exc)

    # Invalidate cache so next call picks up new values.
    global _cache
    _cache = None

    return path


def compute_station_bias(
    wu_highs: list[float],
    wu_lows: list[float],
    om_maxs: list[float],
    om_mins: list[float],
    station_id: str = "",
    city: str = "",
) -> StationBias:
    """Compute bias statistics from matched WU and Open-Meteo observations.

    All inputs are paired lists of the same length (days with valid data
    from both sources). Missing days should already be excluded.

    Args:
        wu_highs: WU daily high temps in °C
        wu_lows: WU daily low temps in °C
        om_maxs: Open-Meteo ERA5 daily max temps in °C
        om_mins: Open-Meteo ERA5 daily min temps in °C
    """
    if len(wu_highs) != len(om_maxs):
        raise ValueError(f"Mismatched high/max lengths: {len(wu_highs)} vs {len(om_maxs)}")
    if len(wu_lows) != len(om_mins):
        raise ValueError(f"Mismatched low/min lengths: {len(wu_lows)} vs {len(om_mins)}")

    high_diffs = np.array(wu_highs) - np.array(om_maxs)
    low_diffs = np.array(wu_lows) - np.array(om_mins)

    high_bias = float(np.mean(high_diffs)) if len(high_diffs) > 0 else 0.0
    low_bias = float(np.mean(low_diffs)) if len(low_diffs) > 0 else 0.0
    mean_bias = (high_bias + low_bias) / 2.0

    high_std = float(np.std(high_diffs, ddof=1)) if len(high_diffs) > 1 else 0.0
    low_std = float(np.std(low_diffs, ddof=1)) if len(low_diffs) > 1 else 0.0

    n_days = min(len(wu_highs), len(wu_lows))

    return StationBias(
        station_id=station_id,
        city=city,
        high_bias_c=high_bias,
        low_bias_c=low_bias,
        mean_bias_c=mean_bias,
        high_std_c=high_std,
        low_std_c=low_std,
        n_days=n_days,
    )


def _compute_bucket_bias(
    wu_highs: list[float],
    wu_lows: list[float],
    om_maxs: list[float],
    om_mins: list[float],
    condition: SkyCondition,
) -> ConditionBias:
    """Compute bias for a single sky-condition bucket."""
    if not wu_highs:
        return ConditionBias(condition=condition, high_bias_c=0.0, low_bias_c=0.0, mean_bias_c=0.0)

    high_diffs = np.array(wu_highs) - np.array(om_maxs)
    low_diffs = np.array(wu_lows) - np.array(om_mins)

    high_bias = float(np.mean(high_diffs))
    low_bias = float(np.mean(low_diffs))
    mean_bias = (high_bias + low_bias) / 2.0

    high_std = float(np.std(high_diffs, ddof=1)) if len(high_diffs) > 1 else 0.0
    low_std = float(np.std(low_diffs, ddof=1)) if len(low_diffs) > 1 else 0.0

    return ConditionBias(
        condition=condition,
        high_bias_c=high_bias,
        low_bias_c=low_bias,
        mean_bias_c=mean_bias,
        high_std_c=high_std,
        low_std_c=low_std,
        n_days=len(wu_highs),
    )


def compute_station_bias_stratified(
    wu_highs: list[float],
    wu_lows: list[float],
    om_maxs: list[float],
    om_mins: list[float],
    cloud_covers: list[float | None],
    station_id: str = "",
    city: str = "",
) -> StationBiasV2:
    """Compute bias statistics stratified by sky condition.

    All inputs are paired lists of the same length. cloud_covers
    contains daily mean cloud cover percentage (0-100), or None
    for days where cloud cover data is unavailable.
    """
    n = len(wu_highs)
    if len(wu_lows) != n or len(om_maxs) != n or len(om_mins) != n or len(cloud_covers) != n:
        raise ValueError("All input lists must have the same length")

    # Compute global bias
    global_bias = compute_station_bias(wu_highs, wu_lows, om_maxs, om_mins, station_id, city)

    # Stratify by sky condition
    buckets: dict[SkyCondition, tuple[list[float], list[float], list[float], list[float]]] = {
        SkyCondition.CLEAR: ([], [], [], []),
        SkyCondition.PARTLY_CLOUDY: ([], [], [], []),
        SkyCondition.OVERCAST: ([], [], [], []),
    }

    for i in range(n):
        cc = cloud_covers[i]
        if cc is None:
            continue
        condition = classify_sky_condition(cc)
        bh, bl, bm, bn = buckets[condition]
        bh.append(wu_highs[i])
        bl.append(wu_lows[i])
        bm.append(om_maxs[i])
        bn.append(om_mins[i])

    condition_biases: list[ConditionBias] = []
    for condition in SkyCondition:
        bh, bl, bm, bn = buckets[condition]
        condition_biases.append(_compute_bucket_bias(bh, bl, bm, bn, condition))

    return StationBiasV2(
        station_id=station_id,
        city=city,
        high_bias_c=global_bias.high_bias_c,
        low_bias_c=global_bias.low_bias_c,
        mean_bias_c=global_bias.mean_bias_c,
        high_std_c=global_bias.high_std_c,
        low_std_c=global_bias.low_std_c,
        n_days=global_bias.n_days,
        condition_biases=tuple(condition_biases),
    )
