"""Temperature threshold forecast model.

Fits a distribution to ensemble members at the target time,
then computes CDF at the threshold to get exceedance probability.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from weather_edge.config import get_settings
from weather_edge.forecasting.base import ProbabilityEstimate
from weather_edge.forecasting.calibration import (
    confidence_from_lead_time,
    inflate_ensemble_spread,
)
from weather_edge.forecasting.utils import find_closest_time_idx
from weather_edge.markets.models import Comparison, MarketParams
from weather_edge.weather.models import EnsembleForecast, NOAAForecast

logger = logging.getLogger(__name__)


def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def _compute_ensemble_prob(
    members: np.ndarray,
    threshold: float,
    comparison: Comparison,
    lead_time_hours: float,
    threshold_upper: float | None = None,
) -> tuple[float, float]:
    """Compute probability from ensemble members.

    Returns (raw_prob, calibrated_prob).
    """
    valid = members[~np.isnan(members)]
    if len(valid) < 3:
        return 0.5, 0.5

    # Apply spread inflation for calibration
    inflated = inflate_ensemble_spread(valid, lead_time_hours)

    # Fit normal distribution to inflated ensemble
    mu, sigma = np.mean(inflated), np.std(inflated, ddof=1)
    if sigma < 0.01:
        sigma = 0.01  # Prevent degenerate distribution

    # Raw probability from uninflated members
    raw_mu, raw_sigma = np.mean(valid), np.std(valid, ddof=1)
    if raw_sigma < 0.01:
        raw_sigma = 0.01

    if comparison == Comparison.ABOVE:
        raw_prob = 1.0 - stats.norm.cdf(threshold, raw_mu, raw_sigma)
        cal_prob = 1.0 - stats.norm.cdf(threshold, mu, sigma)
    elif comparison == Comparison.BETWEEN and threshold_upper is not None:
        raw_prob = (
            stats.norm.cdf(threshold_upper, raw_mu, raw_sigma)
            - stats.norm.cdf(threshold, raw_mu, raw_sigma)
        )
        cal_prob = (
            stats.norm.cdf(threshold_upper, mu, sigma)
            - stats.norm.cdf(threshold, mu, sigma)
        )
    elif comparison == Comparison.BELOW:
        raw_prob = stats.norm.cdf(threshold, raw_mu, raw_sigma)
        cal_prob = stats.norm.cdf(threshold, mu, sigma)
    else:
        raw_prob = 0.5
        cal_prob = 0.5

    return float(raw_prob), float(cal_prob)


class TemperatureModel:
    """Temperature threshold forecast model."""

    async def estimate(
        self,
        params: MarketParams,
        gfs: EnsembleForecast | None,
        ecmwf: EnsembleForecast | None,
        noaa: NOAAForecast | None,
    ) -> ProbabilityEstimate:
        """Estimate probability of temperature exceeding/falling below threshold."""
        settings = get_settings()
        threshold = params.threshold
        if threshold is None:
            return ProbabilityEstimate(
                probability=0.5, raw_probability=0.5, confidence=0.0,
                lead_time_hours=0, details="No threshold specified",
            )

        if params.target_date is None:
            return ProbabilityEstimate(
                probability=0.5, raw_probability=0.5, confidence=0.0,
                lead_time_hours=0, details="No target date specified",
            )

        # Convert threshold to Celsius for ensemble comparison (Open-Meteo returns C)
        threshold_c = threshold
        if params.unit.upper() == "F":
            threshold_c = _fahrenheit_to_celsius(threshold)

        threshold_upper_c: float | None = None
        if params.threshold_upper is not None:
            threshold_upper_c = params.threshold_upper
            if params.unit.upper() == "F":
                threshold_upper_c = _fahrenheit_to_celsius(params.threshold_upper)

        target_time = params.target_date
        now = datetime.now(timezone.utc)
        lead_time_hours = (target_time - now).total_seconds() / 3600

        # Reject past dates and targets beyond ensemble range (~16 days)
        if lead_time_hours < -6:
            return ProbabilityEstimate(
                probability=0.5, raw_probability=0.5, confidence=0.0,
                lead_time_hours=0, details="Target date is in the past",
            )
        if lead_time_hours > 384:
            return ProbabilityEstimate(
                probability=0.5, raw_probability=0.5, confidence=0.0,
                lead_time_hours=lead_time_hours,
                details="Target date beyond ensemble forecast range",
            )
        lead_time_hours = max(0, lead_time_hours)

        probs: list[tuple[float, float, float]] = []  # (raw, cal, weight)
        sources: list[str] = []
        details_parts: list[str] = []

        # ECMWF ensemble
        if ecmwf is not None and ecmwf.n_members > 0:
            idx = find_closest_time_idx(ecmwf.times, target_time)
            if idx is not None:
                members = ecmwf.temperature_2m[idx, :]
                raw_p, cal_p = _compute_ensemble_prob(
                    members, threshold_c, params.comparison, lead_time_hours,
                    threshold_upper_c,
                )
                probs.append((raw_p, cal_p, settings.ecmwf_weight))
                sources.append(f"ECMWF ({ecmwf.n_members} members)")
                details_parts.append(
                    f"ECMWF: mean={np.nanmean(members):.1f}C, "
                    f"std={np.nanstd(members):.1f}C, p={cal_p:.3f}"
                )

        # GFS ensemble
        if gfs is not None and gfs.n_members > 0:
            idx = find_closest_time_idx(gfs.times, target_time)
            if idx is not None:
                members = gfs.temperature_2m[idx, :]
                raw_p, cal_p = _compute_ensemble_prob(
                    members, threshold_c, params.comparison, lead_time_hours,
                    threshold_upper_c,
                )
                gfs_weight = 1.0 - settings.ecmwf_weight
                probs.append((raw_p, cal_p, gfs_weight))
                sources.append(f"GFS ({gfs.n_members} members)")
                details_parts.append(
                    f"GFS: mean={np.nanmean(members):.1f}C, "
                    f"std={np.nanstd(members):.1f}C, p={cal_p:.3f}"
                )

        # NOAA forecast (supplementary, not blended directly)
        if noaa is not None and noaa.periods:
            noaa_temp = None
            for period in noaa.periods:
                if period.start_time and abs((period.start_time - target_time).total_seconds()) < 7200:
                    noaa_temp = period.temperature
                    break
            if noaa_temp is not None:
                # NOAA temps are in F
                noaa_threshold = threshold if params.unit.upper() == "F" else _celsius_to_fahrenheit(threshold)
                sources.append("NOAA/NWS")
                details_parts.append(f"NOAA point forecast: {noaa_temp}F (threshold: {noaa_threshold}F)")

        if not probs:
            return ProbabilityEstimate(
                probability=0.5, raw_probability=0.5, confidence=0.0,
                lead_time_hours=lead_time_hours, sources_used=sources,
                details="No ensemble data available",
            )

        # Weighted blend of calibrated probabilities
        total_weight = sum(w for _, _, w in probs)
        blended_cal = sum(cal * w for _, cal, w in probs) / total_weight
        blended_raw = sum(raw * w for raw, _, w in probs) / total_weight

        confidence = confidence_from_lead_time(lead_time_hours)
        # Reduce confidence if only one source
        if len(probs) == 1:
            confidence *= 0.85
        # Reduce confidence if no NOAA data for US location
        if noaa is None and len(sources) < 3:
            confidence *= 0.95

        return ProbabilityEstimate(
            probability=float(np.clip(blended_cal, 0.001, 0.999)),
            raw_probability=float(np.clip(blended_raw, 0.001, 0.999)),
            confidence=confidence,
            lead_time_hours=lead_time_hours,
            sources_used=sources,
            details=" | ".join(details_parts),
        )
