"""Shared forecasting utilities."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# Maximum allowed time gap (hours) between target time and closest ensemble time
# before issuing a warning
_MAX_TIME_GAP_HOURS = 72


def find_closest_time_idx(times: list[datetime], target: datetime) -> int | None:
    """Find index of the closest time to the target.

    Returns None if times is empty.
    Logs a warning if the closest time is more than 72h from the target.
    """
    if not times:
        return None
    diffs = [abs((t - target).total_seconds()) for t in times]
    idx = int(np.argmin(diffs))

    distance_hours = diffs[idx] / 3600
    if distance_hours > _MAX_TIME_GAP_HOURS:
        logger.warning(
            "Target time %s is %.0fh from closest ensemble time %s — "
            "forecast data may not be representative",
            target.isoformat(),
            distance_hours,
            times[idx].isoformat(),
        )

    return idx


def find_period_time_indices(
    times: list[datetime],
    period_start: datetime,
    period_end: datetime,
) -> list[int]:
    """Return indices of *times* that fall within [period_start, period_end]."""
    return [
        i for i, t in enumerate(times)
        if period_start <= t <= period_end
    ]


def compute_coverage_fraction(
    times: list[datetime],
    period_start: datetime,
    period_end: datetime,
) -> float:
    """Fraction of the requested period covered by *times*.

    Returns 0.0 when the period is zero-length or no times overlap,
    up to 1.0 when every hour is present.
    """
    total_hours = (period_end - period_start).total_seconds() / 3600
    if total_hours <= 0:
        return 0.0
    n_hits = len(find_period_time_indices(times, period_start, period_end))
    return min(1.0, n_hits / total_hours)
