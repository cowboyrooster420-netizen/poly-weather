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
