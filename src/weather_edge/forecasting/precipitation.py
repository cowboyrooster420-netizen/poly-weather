"""Precipitation forecast model using gamma distribution.

Precipitation is skewed (many zero/low values, fat right tail),
so we use a gamma distribution rather than normal.
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
    get_spread_inflation,
)
from weather_edge.forecasting.utils import (
    compute_coverage_fraction,
    find_closest_time_idx,
    find_period_time_indices,
)
from weather_edge.markets.models import Comparison, MarketParams
from weather_edge.weather.models import EnsembleForecast, NOAAForecast

logger = logging.getLogger(__name__)


def _mm_to_inches(mm: float) -> float:
    return mm / 25.4


def _inches_to_mm(inches: float) -> float:
    return inches * 25.4


def _compute_precip_prob(
    members: np.ndarray,
    threshold_mm: float,
    comparison: Comparison,
    lead_time_hours: float,
    threshold_upper_mm: float | None = None,
) -> tuple[float, float]:
    """Compute precipitation probability using gamma distribution.

    Returns (raw_prob, calibrated_prob).
    """
    valid = members[~np.isnan(members)]
    if len(valid) < 3:
        return 0.5, 0.5

    # Fraction of members with zero precip
    zero_frac = np.sum(valid <= 0.01) / len(valid)
    nonzero = valid[valid > 0.01]

    if len(nonzero) < 2:
        # Almost all members predict zero precip
        if comparison == Comparison.ABOVE:
            return 1.0 - zero_frac, 1.0 - zero_frac
        elif comparison == Comparison.BETWEEN:
            # If almost all zero and thresholds are above zero → ~0
            return 0.0, 0.0
        else:
            return zero_frac, zero_frac

    # Fit gamma to nonzero values
    try:
        shape, _, scale = stats.gamma.fit(nonzero, floc=0)
    except (RuntimeError, ValueError) as exc:
        # Gamma fit can fail with poorly conditioned data
        logger.debug("Gamma fit failed, using empirical fallback: %s", exc)
        if comparison == Comparison.BETWEEN and threshold_upper_mm is not None:
            emp = np.sum((valid > threshold_mm) & (valid < threshold_upper_mm)) / len(valid)
        else:
            emp = np.sum(valid > threshold_mm) / len(valid)
        return emp, emp

    # Apply spread inflation to gamma parameters
    inflation = get_spread_inflation(lead_time_hours)
    inflated_scale = scale * inflation

    # P(X > 0) = 1 - zero_frac
    if comparison == Comparison.ABOVE:
        raw_exc = 1.0 - stats.gamma.cdf(threshold_mm, shape, scale=scale)
        cal_exc = 1.0 - stats.gamma.cdf(threshold_mm, shape, scale=inflated_scale)
        raw_prob = (1.0 - zero_frac) * raw_exc
        cal_prob = (1.0 - zero_frac) * cal_exc
    elif comparison == Comparison.BETWEEN and threshold_upper_mm is not None:
        # P(lower < X < upper) = CDF(upper) - CDF(lower)   (conditional on X > 0)
        raw_band = (
            stats.gamma.cdf(threshold_upper_mm, shape, scale=scale)
            - stats.gamma.cdf(threshold_mm, shape, scale=scale)
        )
        cal_band = (
            stats.gamma.cdf(threshold_upper_mm, shape, scale=inflated_scale)
            - stats.gamma.cdf(threshold_mm, shape, scale=inflated_scale)
        )
        raw_prob = (1.0 - zero_frac) * raw_band
        cal_prob = (1.0 - zero_frac) * cal_band
    else:
        # BELOW: P(X < threshold) = zero_frac + (1 - zero_frac) * P(X < threshold | X > 0)
        raw_cdf = stats.gamma.cdf(threshold_mm, shape, scale=scale)
        cal_cdf = stats.gamma.cdf(threshold_mm, shape, scale=inflated_scale)
        raw_prob = zero_frac + (1.0 - zero_frac) * raw_cdf
        cal_prob = zero_frac + (1.0 - zero_frac) * cal_cdf

    return float(raw_prob), float(cal_prob)


def _aggregate_period_precip(
    forecast: EnsembleForecast,
    indices: list[int],
) -> np.ndarray:
    """Sum hourly precipitation across *indices* per ensemble member.

    Returns shape ``(n_members,)``.
    """
    return forecast.precipitation[indices, :].sum(axis=0)


class PrecipitationModel:
    """Precipitation threshold forecast model."""

    async def estimate(
        self,
        params: MarketParams,
        gfs: EnsembleForecast | None,
        ecmwf: EnsembleForecast | None,
        noaa: NOAAForecast | None,
    ) -> ProbabilityEstimate:
        """Estimate probability of precipitation exceeding threshold."""
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

        # Convert threshold to mm (Open-Meteo returns mm)
        threshold_mm = threshold
        if params.unit.lower() in ("in", "inches"):
            threshold_mm = _inches_to_mm(threshold)

        threshold_upper_mm: float | None = None
        if params.threshold_upper is not None:
            threshold_upper_mm = params.threshold_upper
            if params.unit.lower() in ("in", "inches"):
                threshold_upper_mm = _inches_to_mm(params.threshold_upper)

        target_time = params.target_date
        now = datetime.now(timezone.utc)
        lead_time_hours = (target_time - now).total_seconds() / 3600

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

        # Route to period or single-timestep path
        if params.period_start is not None and params.period_end is not None:
            return await self._estimate_period(
                params, gfs, ecmwf, noaa,
                threshold_mm, threshold_upper_mm, lead_time_hours,
            )
        return await self._estimate_single(
            params, gfs, ecmwf, noaa,
            threshold_mm, threshold_upper_mm, lead_time_hours,
        )

    # ------------------------------------------------------------------
    # Single-timestep path (original logic, also used for non-period markets)
    # ------------------------------------------------------------------

    async def _estimate_single(
        self,
        params: MarketParams,
        gfs: EnsembleForecast | None,
        ecmwf: EnsembleForecast | None,
        noaa: NOAAForecast | None,
        threshold_mm: float,
        threshold_upper_mm: float | None,
        lead_time_hours: float,
    ) -> ProbabilityEstimate:
        settings = get_settings()
        target_time = params.target_date

        probs: list[tuple[float, float, float]] = []
        sources: list[str] = []
        details_parts: list[str] = []

        # ECMWF ensemble
        if ecmwf is not None and ecmwf.n_members > 0:
            idx = find_closest_time_idx(ecmwf.times, target_time)
            if idx is not None:
                members = ecmwf.precipitation[idx, :]
                raw_p, cal_p = _compute_precip_prob(
                    members, threshold_mm, params.comparison, lead_time_hours,
                    threshold_upper_mm,
                )
                probs.append((raw_p, cal_p, settings.ecmwf_weight))
                sources.append(f"ECMWF ({ecmwf.n_members} members)")
                details_parts.append(
                    f"ECMWF: median={np.nanmedian(members):.1f}mm, "
                    f"max={np.nanmax(members):.1f}mm, p={cal_p:.3f}"
                )

        # GFS ensemble
        if gfs is not None and gfs.n_members > 0:
            idx = find_closest_time_idx(gfs.times, target_time)
            if idx is not None:
                members = gfs.precipitation[idx, :]
                raw_p, cal_p = _compute_precip_prob(
                    members, threshold_mm, params.comparison, lead_time_hours,
                    threshold_upper_mm,
                )
                gfs_weight = 1.0 - settings.ecmwf_weight
                probs.append((raw_p, cal_p, gfs_weight))
                sources.append(f"GFS ({gfs.n_members} members)")
                details_parts.append(
                    f"GFS: median={np.nanmedian(members):.1f}mm, "
                    f"max={np.nanmax(members):.1f}mm, p={cal_p:.3f}"
                )

        # NOAA precip probability (supplementary)
        if noaa is not None and noaa.periods:
            for period in noaa.periods:
                if (
                    period.start_time
                    and abs((period.start_time - target_time).total_seconds()) < 7200
                    and period.precipitation_probability is not None
                ):
                    sources.append("NOAA/NWS")
                    details_parts.append(
                        f"NOAA PoP: {period.precipitation_probability}%"
                    )
                    break

        if not probs:
            return ProbabilityEstimate(
                probability=0.5, raw_probability=0.5, confidence=0.0,
                lead_time_hours=lead_time_hours, sources_used=sources,
                details="No ensemble data available",
            )

        # Weighted blend
        total_weight = sum(w for _, _, w in probs)
        blended_cal = sum(cal * w for _, cal, w in probs) / total_weight
        blended_raw = sum(raw * w for raw, _, w in probs) / total_weight

        # Precipitation has lower inherent predictability than temperature
        confidence = confidence_from_lead_time(lead_time_hours) * 0.85
        if len(probs) == 1:
            confidence *= 0.85

        return ProbabilityEstimate(
            probability=float(np.clip(blended_cal, 0.001, 0.999)),
            raw_probability=float(np.clip(blended_raw, 0.001, 0.999)),
            confidence=confidence,
            lead_time_hours=lead_time_hours,
            sources_used=sources,
            details=" | ".join(details_parts),
        )

    # ------------------------------------------------------------------
    # Period-aggregated path (weekly / monthly precipitation totals)
    # ------------------------------------------------------------------

    async def _estimate_period(
        self,
        params: MarketParams,
        gfs: EnsembleForecast | None,
        ecmwf: EnsembleForecast | None,
        noaa: NOAAForecast | None,
        threshold_mm: float,
        threshold_upper_mm: float | None,
        lead_time_hours: float,
    ) -> ProbabilityEstimate:
        settings = get_settings()
        period_start = params.period_start
        period_end = params.period_end

        probs: list[tuple[float, float, float]] = []
        sources: list[str] = []
        details_parts: list[str] = []
        coverage = 0.0

        # ECMWF ensemble (period aggregation)
        if ecmwf is not None and ecmwf.n_members > 0:
            indices = find_period_time_indices(ecmwf.times, period_start, period_end)
            cov = compute_coverage_fraction(ecmwf.times, period_start, period_end)
            if indices:
                members = _aggregate_period_precip(ecmwf, indices)
                # Scale up if partial coverage (assumes uniform rate)
                if 0 < cov < 1.0:
                    members = members / cov
                raw_p, cal_p = _compute_precip_prob(
                    members, threshold_mm, params.comparison, lead_time_hours,
                    threshold_upper_mm,
                )
                probs.append((raw_p, cal_p, settings.ecmwf_weight))
                sources.append(f"ECMWF ({ecmwf.n_members} members, {len(indices)}h)")
                details_parts.append(
                    f"ECMWF period: sum_median={np.nanmedian(members):.1f}mm, "
                    f"coverage={cov:.0%}, p={cal_p:.3f}"
                )
                coverage = max(coverage, cov)

        # GFS ensemble (period aggregation)
        if gfs is not None and gfs.n_members > 0:
            indices = find_period_time_indices(gfs.times, period_start, period_end)
            cov = compute_coverage_fraction(gfs.times, period_start, period_end)
            if indices:
                members = _aggregate_period_precip(gfs, indices)
                if 0 < cov < 1.0:
                    members = members / cov
                raw_p, cal_p = _compute_precip_prob(
                    members, threshold_mm, params.comparison, lead_time_hours,
                    threshold_upper_mm,
                )
                gfs_weight = 1.0 - settings.ecmwf_weight
                probs.append((raw_p, cal_p, gfs_weight))
                sources.append(f"GFS ({gfs.n_members} members, {len(indices)}h)")
                details_parts.append(
                    f"GFS period: sum_median={np.nanmedian(members):.1f}mm, "
                    f"coverage={cov:.0%}, p={cal_p:.3f}"
                )
                coverage = max(coverage, cov)

        if not probs:
            return ProbabilityEstimate(
                probability=0.5, raw_probability=0.5, confidence=0.0,
                lead_time_hours=lead_time_hours, sources_used=sources,
                details="No ensemble data available for period",
            )

        # Weighted blend
        total_weight = sum(w for _, _, w in probs)
        blended_cal = sum(cal * w for _, cal, w in probs) / total_weight
        blended_raw = sum(raw * w for raw, _, w in probs) / total_weight

        # Base confidence
        confidence = confidence_from_lead_time(lead_time_hours) * 0.85
        if len(probs) == 1:
            confidence *= 0.85

        # Reduce confidence proportionally to coverage
        if coverage < 0.25:
            confidence *= 0.3
        elif coverage < 0.50:
            confidence *= 0.5
        elif coverage < 0.75:
            confidence *= 0.7
        elif coverage < 1.0:
            confidence *= 0.85

        details_parts.append(f"period_coverage={coverage:.0%}")

        return ProbabilityEstimate(
            probability=float(np.clip(blended_cal, 0.001, 0.999)),
            raw_probability=float(np.clip(blended_raw, 0.001, 0.999)),
            confidence=confidence,
            lead_time_hours=lead_time_hours,
            sources_used=sources,
            details=" | ".join(details_parts),
        )
