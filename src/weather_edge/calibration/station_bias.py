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

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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


# Module-level cache so we only load from disk once.
_cache: dict[str, StationBias] | None = None


def _load_biases() -> dict[str, StationBias]:
    """Load biases from user file, falling back to bundled defaults."""
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

    biases: dict[str, StationBias] = {}
    for station_id, info in data.get("stations", {}).items():
        biases[station_id] = StationBias(
            station_id=station_id,
            city=info.get("city", ""),
            high_bias_c=info.get("high_bias_c", 0.0),
            low_bias_c=info.get("low_bias_c", 0.0),
            mean_bias_c=info.get("mean_bias_c", 0.0),
            high_std_c=info.get("high_std_c", 0.0),
            low_std_c=info.get("low_std_c", 0.0),
            n_days=info.get("n_days", 0),
        )

    source = "user" if path == user_path else "bundled"
    logger.info("Loaded biases for %d stations from %s (%s)", len(biases), path, source)
    return biases


def load_biases(*, force: bool = False) -> dict[str, StationBias]:
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


def save_biases(biases: dict[str, StationBias], *, training_days: int = 90) -> Path:
    """Persist biases to the user's station_bias_path.

    Returns the path written to.
    """
    settings = get_settings()
    path = settings.station_bias_path
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_days": training_days,
        "stations": {
            sid: {
                "city": b.city,
                "high_bias_c": round(b.high_bias_c, 3),
                "low_bias_c": round(b.low_bias_c, 3),
                "mean_bias_c": round(b.mean_bias_c, 3),
                "high_std_c": round(b.high_std_c, 3),
                "low_std_c": round(b.low_std_c, 3),
                "n_days": b.n_days,
            }
            for sid, b in biases.items()
        },
    }

    path.write_text(json.dumps(data, indent=2) + "\n")
    logger.info("Saved biases for %d stations to %s", len(biases), path)

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
