"""Temperature threshold forecast model.

Fits a distribution to ensemble members at the target time,
then computes CDF at the threshold to get exceedance probability.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy import stats

from weather_edge.config import get_settings
from weather_edge.forecasting.base import ProbabilityEstimate
from weather_edge.forecasting.calibration import (
    confidence_from_lead_time,
    inflate_ensemble_spread,
)
from weather_edge.forecasting.utils import find_closest_time_idx, find_period_time_indices
from weather_edge.markets.models import Comparison, MarketParams
from weather_edge.weather.models import EnsembleForecast, NOAAForecast

logger = logging.getLogger(__name__)


def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def _get_daily_members(
    forecast: EnsembleForecast,
    target_date: datetime,
    aggregation: str,
    longitude: float | None = None,
) -> np.ndarray | None:
    """Get daily max/min temperature per ensemble member.

    Finds all hours on target_date in *local* time (approximated from
    longitude), takes max or min across hours per member.
    Returns shape (n_members,) or None if no hours found.
    """
    # Shift day boundaries to local midnight using longitude
    utc_offset_hours = -(longitude / 15.0) if longitude is not None else 0.0
    offset = timedelta(hours=utc_offset_hours)
    local_midnight = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start = local_midnight + offset
    day_end = day_start + timedelta(hours=24) - timedelta(seconds=1)
    indices = find_period_time_indices(forecast.times, day_start, day_end)
    if not indices:
        return None
    # shape (n_hours, n_members)
    temps = forecast.temperature_2m[indices, :]
    if aggregation == "max":
        return np.max(temps, axis=0)
    elif aggregation == "min":
        return np.min(temps, axis=0)
    return None


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


def _compute_dynamic_weights(
    models: list[tuple[float, float]],
    disagreement_threshold: float = 2.5,
    min_ratio: float = 0.15,
    scale: float = 2.0,
) -> list[float]:
    """Adjust model weights based on inter-model disagreement.

    Args:
        models: list of (ensemble_mean, base_weight) per model
        disagreement_threshold: °C gap before penalizing outliers
        min_ratio: floor on weight penalty (outlier keeps at least this fraction)
        scale: controls how fast penalty grows with excess disagreement

    Returns:
        Adjusted weights (sum to 1.0), same order as input.
    """
    if len(models) < 2:
        return [w for _, w in models]

    # Weighted multi-model mean
    total_w = sum(w for _, w in models)
    mm_mean = sum(m * w for m, w in models) / total_w

    # Max pairwise disagreement
    means = [m for m, _ in models]
    max_gap = max(means) - min(means)

    if max_gap <= disagreement_threshold:
        return [w for _, w in models]

    # Penalize each model proportional to its distance from the multi-model mean
    adjusted: list[float] = []
    for mean, base_w in models:
        dist = abs(mean - mm_mean)
        excess = max(0.0, dist - disagreement_threshold / 2)
        if excess > 0:
            penalty = max(min_ratio, 1.0 / (1.0 + excess / scale))
            adjusted.append(base_w * penalty)
        else:
            adjusted.append(base_w)

    # Renormalize
    adj_total = sum(adjusted)
    return [w / adj_total for w in adjusted]


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

        # For daily aggregation markets, compute lead time to local noon
        # instead of midnight UTC.  Approximate local noon from longitude:
        # noon_utc_hour = 12 - (longitude / 15)
        if params.daily_aggregation is not None and params.lat_lon is not None:
            _, lon = params.lat_lon
            noon_utc_hour = 12.0 - lon / 15.0  # e.g. Atlanta -84.4 → ~17.6 UTC
            local_noon = target_time.replace(
                hour=int(noon_utc_hour) % 24,
                minute=int((noon_utc_hour % 1) * 60),
                second=0, microsecond=0,
            )
            lead_time_hours = (local_noon - now).total_seconds() / 3600
        else:
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

        probs: list[tuple[float, float, float, float]] = []  # (raw, cal, weight, mean_c)
        sources: list[str] = []
        details_parts: list[str] = []

        # ECMWF ensemble
        if ecmwf is not None and ecmwf.n_members > 0:
            members = None
            if params.daily_aggregation is not None:
                members = _get_daily_members(ecmwf, target_time, params.daily_aggregation, longitude=params.lat_lon[1] if params.lat_lon else None)
            else:
                idx = find_closest_time_idx(ecmwf.times, target_time)
                if idx is not None:
                    members = ecmwf.temperature_2m[idx, :]
            if members is not None:
                raw_p, cal_p = _compute_ensemble_prob(
                    members, threshold_c, params.comparison, lead_time_hours,
                    threshold_upper_c,
                )
                mean_c = float(np.nanmean(members))
                probs.append((raw_p, cal_p, settings.ecmwf_weight, mean_c))
                sources.append(f"ECMWF ({ecmwf.n_members} members)")
                details_parts.append(
                    f"ECMWF: mean={mean_c:.1f}C, "
                    f"std={np.nanstd(members):.1f}C, p={cal_p:.3f}"
                )

        # GFS ensemble
        if gfs is not None and gfs.n_members > 0:
            members = None
            if params.daily_aggregation is not None:
                members = _get_daily_members(gfs, target_time, params.daily_aggregation, longitude=params.lat_lon[1] if params.lat_lon else None)
            else:
                idx = find_closest_time_idx(gfs.times, target_time)
                if idx is not None:
                    members = gfs.temperature_2m[idx, :]
            if members is not None:
                raw_p, cal_p = _compute_ensemble_prob(
                    members, threshold_c, params.comparison, lead_time_hours,
                    threshold_upper_c,
                )
                gfs_weight = 1.0 - settings.ecmwf_weight
                mean_c = float(np.nanmean(members))
                probs.append((raw_p, cal_p, gfs_weight, mean_c))
                sources.append(f"GFS ({gfs.n_members} members)")
                details_parts.append(
                    f"GFS: mean={mean_c:.1f}C, "
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

        # Dynamic weighting: adjust base weights when models disagree
        base_models = [(mean_c, w) for _, _, w, mean_c in probs]
        adjusted_weights = _compute_dynamic_weights(
            base_models,
            disagreement_threshold=settings.model_disagreement_threshold,
            min_ratio=settings.model_min_weight_ratio,
        )

        # Log when weights are adjusted
        base_weights = [w for _, _, w, _ in probs]
        if adjusted_weights != base_weights:
            model_names = [s.split(" (")[0] for s in sources[:len(probs)]]
            pairs = ", ".join(
                f"{name}: {bw:.2f}->{aw:.2f}"
                for name, bw, aw in zip(model_names, base_weights, adjusted_weights)
            )
            logger.info("Dynamic weighting adjusted: %s", pairs)

        # Weighted blend of calibrated probabilities
        total_weight = sum(adjusted_weights)
        blended_cal = sum(cal * w for (_, cal, _, _), w in zip(probs, adjusted_weights)) / total_weight
        blended_raw = sum(raw * w for (raw, _, _, _), w in zip(probs, adjusted_weights)) / total_weight

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
