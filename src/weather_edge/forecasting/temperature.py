"""Temperature threshold forecast model.

Fits a distribution to ensemble members at the target time,
then computes CDF at the threshold to get exceedance probability.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
from scipy import stats

from weather_edge.config import get_settings
from weather_edge.forecasting.base import ProbabilityEstimate
from weather_edge.forecasting.calibration import (
    confidence_from_lead_time,
    inflate_ensemble_spread,
)
from weather_edge.calibration.station_bias import get_station_bias
from weather_edge.common.types import celsius_to_fahrenheit, fahrenheit_to_celsius
from weather_edge.forecasting.utils import find_closest_time_idx, find_period_time_indices
from weather_edge.markets.models import Comparison, MarketParams
from weather_edge.weather.models import EnsembleForecast, HRRRForecast, NOAAForecast
from weather_edge.weather.stations import station_for_location

logger = logging.getLogger(__name__)

# Local aliases for backward compatibility with callers (and brevity).
_celsius_to_fahrenheit = celsius_to_fahrenheit
_fahrenheit_to_celsius = fahrenheit_to_celsius


def _get_daily_members(
    forecast: EnsembleForecast,
    target_date: datetime,
    aggregation: str,
    tz_name: str | None = None,
) -> np.ndarray | None:
    """Get daily max/min temperature per ensemble member.

    Finds all hours on target_date in *local* time (using IANA timezone),
    takes max or min across hours per member.
    Returns shape (n_members,) or None if no hours found.
    """
    # Use actual timezone for correct day boundaries (incl. DST)
    if tz_name is not None:
        tz = ZoneInfo(tz_name)
    else:
        tz = timezone.utc
    local_midnight = target_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=tz)
    day_start = local_midnight.astimezone(timezone.utc)
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


def _compute_ensemble_prob_between(
    valid: np.ndarray,
    lower: float,
    upper: float,
    lead_time_hours: float,
) -> tuple[float, float]:
    """Compute BETWEEN probability using hybrid empirical-parametric blend.

    Narrow bands lean on empirical member counting; wide bands converge
    to parametric CDF difference.  Returns (raw_prob, calibrated_prob).

    Args:
        valid: 1-D array of ensemble member values with NaN already removed.
    """
    if len(valid) < 3:
        return 0.5, 0.5

    band_width = upper - lower
    if band_width <= 0:
        return 0.0, 0.0

    n_members = len(valid)

    # --- Empirical probability (raw): fraction of members in [lower, upper] ---
    def _empirical(arr: np.ndarray) -> float:
        count = np.sum((arr >= lower) & (arr <= upper))
        prob = float(count) / len(arr)
        # Laplace continuity correction: if no members land in the band but
        # the band overlaps the member range, use a small floor instead of 0.
        if prob == 0.0:
            arr_min, arr_max = float(np.min(arr)), float(np.max(arr))
            if upper >= arr_min and lower <= arr_max:
                prob = 0.5 / len(arr)
        return prob

    raw_empirical = _empirical(valid)

    # --- Parametric probabilities ---
    raw_mu, raw_sigma = float(np.mean(valid)), float(np.std(valid, ddof=1))
    if raw_sigma < 0.01:
        raw_sigma = 0.01

    raw_parametric = float(
        stats.norm.cdf(upper, raw_mu, raw_sigma)
        - stats.norm.cdf(lower, raw_mu, raw_sigma)
    )

    inflated = inflate_ensemble_spread(valid, lead_time_hours)
    cal_mu, cal_sigma = float(np.mean(inflated)), float(np.std(inflated, ddof=1))
    if cal_sigma < 0.01:
        cal_sigma = 0.01

    cal_parametric = float(
        stats.norm.cdf(upper, cal_mu, cal_sigma)
        - stats.norm.cdf(lower, cal_mu, cal_sigma)
    )

    # Empirical count from inflated members for calibrated path
    cal_empirical = _empirical(inflated)

    # --- Sigmoid blend weighted by band_width / sigma ---
    # Crossover at 1.5σ band width; ~88% parametric at 2.5σ, ~5% at 0.5σ
    normalized_width = band_width / raw_sigma
    w_parametric = 1.0 / (1.0 + np.exp(-2.0 * (normalized_width - 1.5)))
    w_empirical = 1.0 - w_parametric

    raw_prob = w_empirical * raw_empirical + w_parametric * raw_parametric
    cal_prob = w_empirical * cal_empirical + w_parametric * cal_parametric

    return float(raw_prob), float(cal_prob)


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
        return _compute_ensemble_prob_between(
            valid, threshold, threshold_upper, lead_time_hours,
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


def _hrrr_blend_weight(lead_time_hours: float) -> float:
    """Return the HRRR blend weight for the given lead time.

    Beyond 18h → 0%, 12-18h → 25%, 6-12h → 50%, <6h → 75%.
    """
    if lead_time_hours >= 18:
        return 0.0
    if lead_time_hours >= 12:
        return 0.25
    if lead_time_hours >= 6:
        return 0.50
    return 0.75


def _apply_hrrr_correction(
    members: np.ndarray,
    hrrr: HRRRForecast,
    target_time: datetime,
    lead_time_hours: float,
) -> tuple[np.ndarray, str | None]:
    """Shift ensemble members toward HRRR deterministic forecast.

    Preserves spread — only moves the center.

    Returns:
        (corrected_members, detail_string_or_None)
    """
    weight = _hrrr_blend_weight(lead_time_hours)
    if weight == 0.0:
        return members, None

    # Find closest HRRR time
    idx = find_closest_time_idx(hrrr.times, target_time)
    if idx is None:
        return members, None

    hrrr_temp_c = float(hrrr.temperature_2m[idx])
    if np.isnan(hrrr_temp_c):
        return members, None

    ensemble_mean = float(np.nanmean(members))
    shift = weight * (hrrr_temp_c - ensemble_mean)
    corrected = members + shift

    detail = f"HRRR correction: {shift:+.1f}°C ({weight:.0%} @ {lead_time_hours:.0f}h)"
    return corrected, detail


def _apply_station_bias_correction(
    members: np.ndarray,
    station_id: str,
    daily_aggregation: str | None,
) -> tuple[np.ndarray, str | None]:
    """Shift ensemble members by the station's learned bias.

    Preserves spread — only moves the center, identical to HRRR correction
    mechanism.

    Returns:
        (corrected_members, detail_string_or_None)
    """
    bias = get_station_bias(station_id, daily_aggregation)
    if abs(bias) < 0.01:
        return members, None
    corrected = members + bias
    detail = f"station bias: {bias:+.1f}°C ({station_id})"
    return corrected, detail


class TemperatureModel:
    """Temperature threshold forecast model."""

    async def estimate(
        self,
        params: MarketParams,
        gfs: EnsembleForecast | None,
        ecmwf: EnsembleForecast | None,
        noaa: NOAAForecast | None,
        *,
        hrrr: HRRRForecast | None = None,
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

        # Resolve WU station for this location (may be None)
        station = station_for_location(params.location) if params.location else None
        station_id = station.wu_id if station is not None else None
        station_tz = station.timezone if station is not None else None

        # For daily aggregation markets, compute lead time to local noon
        # using the station's actual IANA timezone (handles DST correctly).
        if params.daily_aggregation is not None and station_tz is not None:
            tz = ZoneInfo(station_tz)
            local_noon = target_time.replace(
                hour=12, minute=0, second=0, microsecond=0, tzinfo=tz,
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

        hrrr_applied = False
        station_bias_applied = False

        # ECMWF ensemble
        if ecmwf is not None and ecmwf.n_members > 0:
            members = None
            if params.daily_aggregation is not None:
                members = _get_daily_members(ecmwf, target_time, params.daily_aggregation, tz_name=station_tz)
            else:
                idx = find_closest_time_idx(ecmwf.times, target_time)
                if idx is not None:
                    members = ecmwf.temperature_2m[idx, :]
            if members is not None:
                # Apply HRRR bias correction before computing CDF.
                # Skip for daily-aggregated markets — HRRR provides single-hour
                # values which are not comparable to daily max/min.
                if hrrr is not None and params.daily_aggregation is None:
                    members, hrrr_detail = _apply_hrrr_correction(
                        members, hrrr, target_time, lead_time_hours,
                    )
                    if hrrr_detail is not None:
                        if not hrrr_applied:
                            details_parts.append(hrrr_detail)
                        hrrr_applied = True

                # Station bias correction (after HRRR, before CDF)
                if station_id is not None:
                    members, sb_detail = _apply_station_bias_correction(
                        members, station_id, params.daily_aggregation,
                    )
                    if sb_detail is not None and not station_bias_applied:
                        details_parts.append(sb_detail)
                        station_bias_applied = True

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
                members = _get_daily_members(gfs, target_time, params.daily_aggregation, tz_name=station_tz)
            else:
                idx = find_closest_time_idx(gfs.times, target_time)
                if idx is not None:
                    members = gfs.temperature_2m[idx, :]
            if members is not None:
                # Apply HRRR bias correction before computing CDF (single-timestep only)
                if hrrr is not None and params.daily_aggregation is None:
                    members, hrrr_detail = _apply_hrrr_correction(
                        members, hrrr, target_time, lead_time_hours,
                    )
                    if hrrr_detail is not None:
                        if not hrrr_applied:
                            details_parts.append(hrrr_detail)
                        hrrr_applied = True

                # Station bias correction (after HRRR, before CDF)
                if station_id is not None:
                    members, sb_detail = _apply_station_bias_correction(
                        members, station_id, params.daily_aggregation,
                    )
                    if sb_detail is not None and not station_bias_applied:
                        details_parts.append(sb_detail)
                        station_bias_applied = True

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

        if hrrr_applied:
            sources.append("HRRR")

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
