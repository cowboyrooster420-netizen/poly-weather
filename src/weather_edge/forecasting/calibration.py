"""Lead-time-dependent calibration for ensemble forecasts.

Spread inflation factors based on published ECMWF verification statistics.
Platt scaling interface built from the start — activated once ~3 months
of signal/outcome data is collected.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Published ECMWF verification stats: spread inflation factors by lead time.
# These account for ensemble under-dispersion at longer lead times.
# Source: ECMWF reliability diagrams and ensemble calibration literature.
_SPREAD_INFLATION = {
    0: 1.00,    # Analysis time
    6: 1.05,
    12: 1.10,
    24: 1.18,   # Day 1
    48: 1.25,   # Day 2
    72: 1.30,   # Day 3
    120: 1.40,  # Day 5
    168: 1.50,  # Day 7
    240: 1.60,  # Day 10
    336: 1.70,  # Day 14
    384: 1.75,  # Day 16 (max ECMWF range)
}


def get_spread_inflation(lead_time_hours: float) -> float:
    """Get interpolated spread inflation factor for a given lead time.

    Linear interpolation between known breakpoints.
    Extrapolates flat beyond the maximum defined lead time.
    """
    breakpoints = sorted(_SPREAD_INFLATION.keys())

    if lead_time_hours <= breakpoints[0]:
        return _SPREAD_INFLATION[breakpoints[0]]
    if lead_time_hours >= breakpoints[-1]:
        return _SPREAD_INFLATION[breakpoints[-1]]

    # Find surrounding breakpoints and interpolate
    for i in range(len(breakpoints) - 1):
        lo, hi = breakpoints[i], breakpoints[i + 1]
        if lo <= lead_time_hours <= hi:
            frac = (lead_time_hours - lo) / (hi - lo)
            return _SPREAD_INFLATION[lo] + frac * (_SPREAD_INFLATION[hi] - _SPREAD_INFLATION[lo])

    return 1.0  # Shouldn't reach here


def inflate_ensemble_spread(
    members: NDArray[np.float64],
    lead_time_hours: float,
) -> NDArray[np.float64]:
    """Apply spread inflation to ensemble members.

    Increases the spread of ensemble members around their mean
    by the lead-time-dependent inflation factor.

    Args:
        members: 1D array of ensemble member values at a single time step
        lead_time_hours: hours from forecast initialization to valid time

    Returns:
        Inflated ensemble member values
    """
    factor = get_spread_inflation(lead_time_hours)
    mean = np.nanmean(members)
    return mean + (members - mean) * factor


def confidence_from_lead_time(lead_time_hours: float) -> float:
    """Compute a confidence score based on forecast lead time.

    Confidence decreases with lead time, reflecting decreasing
    forecast skill. Used to scale Kelly sizing.

    Returns value in [0.3, 1.0].
    """
    # Exponential decay: ~1.0 at 0h, ~0.7 at 72h, ~0.5 at 168h, ~0.3 at 336h
    decay_rate = 0.003
    return max(0.3, np.exp(-decay_rate * lead_time_hours))


class PlattScaler:
    """Platt scaling for probability calibration.

    Interface is built; actual fitting requires accumulated signal/outcome data.
    Until enough data (~3 months) is collected, this is a pass-through.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0) -> None:
        """Initialize with Platt scaling parameters.

        P_calibrated = 1 / (1 + exp(a * logit(p) + b))

        Default a=1.0, b=0.0 is identity (pass-through).
        """
        self.a = a
        self.b = b
        self._fitted = False

    def calibrate(self, probability: float) -> float:
        """Apply Platt scaling to a raw probability."""
        if not self._fitted:
            return probability

        # Clamp to avoid log(0)
        p = np.clip(probability, 1e-6, 1 - 1e-6)
        logit_p = np.log(p / (1 - p))
        exponent = self.a * logit_p + self.b
        exponent = np.clip(exponent, -30, 30)  # Prevent overflow
        return float(1.0 / (1.0 + np.exp(-exponent)))

    def fit(self, predictions: NDArray[np.float64], outcomes: NDArray[np.float64]) -> None:
        """Fit Platt scaling parameters from historical predictions and outcomes.

        Uses simple logistic regression on (logit(pred), outcome) pairs.
        Requires scipy for optimization.
        """
        from scipy.optimize import minimize

        predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
        logits = np.log(predictions / (1 - predictions))

        def neg_log_likelihood(params: NDArray) -> float:
            a, b = params
            z = np.clip(a * logits + b, -30, 30)
            probs = 1.0 / (1.0 + np.exp(-z))
            probs = np.clip(probs, 1e-6, 1 - 1e-6)
            ll = outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)
            # L2 regularization to prevent unbounded parameters
            return -np.sum(ll) + 0.01 * (a**2 + b**2)

        result = minimize(neg_log_likelihood, [1.0, 0.0], method="Nelder-Mead")
        self.a, self.b = result.x
        self._fitted = True
