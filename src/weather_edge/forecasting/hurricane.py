"""Hurricane forecast model — STUB.

Basic climatological prior + NOAA alert pass-through.
Not investing heavily until temperature/precipitation models are validated.
"""

from __future__ import annotations

from datetime import datetime, timezone

from weather_edge.forecasting.base import ProbabilityEstimate
from weather_edge.forecasting.calibration import confidence_from_lead_time
from weather_edge.markets.models import MarketParams
from weather_edge.weather.models import EnsembleForecast, NOAAForecast

# Rough climatological base rates for US hurricane landfall
# Source: NOAA historical records, averaged over modern era
_CLIMO_ANNUAL_US_LANDFALL = 0.40  # ~40% chance any given year
_CLIMO_MONTHLY: dict[int, float] = {
    1: 0.001, 2: 0.001, 3: 0.001, 4: 0.002, 5: 0.005,
    6: 0.03, 7: 0.05, 8: 0.12, 9: 0.15, 10: 0.08,
    11: 0.02, 12: 0.002,
}


class HurricaneModel:
    """Stub hurricane forecast model.

    Uses climatological base rates adjusted by NOAA alerts.
    A proper implementation would ingest NHC track forecasts,
    SST data, wind shear analysis, etc.
    """

    async def estimate(
        self,
        params: MarketParams,
        gfs: EnsembleForecast | None,
        ecmwf: EnsembleForecast | None,
        noaa: NOAAForecast | None,
    ) -> ProbabilityEstimate:
        """Produce a stub hurricane probability estimate."""
        now = datetime.now(timezone.utc)
        target_time = params.target_date or now
        lead_time_hours = max(0, (target_time - now).total_seconds() / 3600)

        # Start with climatological base rate
        month = target_time.month
        base_prob = _CLIMO_MONTHLY.get(month, 0.01)

        sources = ["climatology"]
        details_parts = [f"Climo base rate for month {month}: {base_prob:.3f}"]

        # Adjust based on NOAA alerts
        if noaa is not None and noaa.alerts:
            hurricane_alerts = [
                a for a in noaa.alerts
                if any(kw in a.event.lower() for kw in ("hurricane", "tropical"))
            ]
            if hurricane_alerts:
                # Active hurricane/tropical storm alerts significantly increase probability
                max_severity = max(
                    ({"extreme": 4, "severe": 3, "moderate": 2, "minor": 1}.get(a.severity.lower(), 0)
                     for a in hurricane_alerts),
                    default=0,
                )
                alert_boost = {4: 0.6, 3: 0.4, 2: 0.2, 1: 0.1}.get(max_severity, 0.05)
                base_prob = min(0.95, base_prob + alert_boost)
                sources.append("NOAA alerts")
                details_parts.append(
                    f"{len(hurricane_alerts)} active alert(s), "
                    f"max severity boost: +{alert_boost:.2f}"
                )

        # Low confidence — this is a stub
        confidence = confidence_from_lead_time(lead_time_hours) * 0.5

        return ProbabilityEstimate(
            probability=base_prob,
            raw_probability=base_prob,
            confidence=confidence,
            lead_time_hours=lead_time_hours,
            sources_used=sources,
            details=" | ".join(details_parts) + " [STUB MODEL]",
        )
