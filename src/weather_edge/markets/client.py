"""Polymarket Gamma + CLOB API client (read-only)."""

from __future__ import annotations

import logging
from datetime import datetime

from weather_edge.common.http import HttpClient
from weather_edge.config import get_settings
from weather_edge.markets.models import WeatherMarket

logger = logging.getLogger(__name__)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


async def fetch_all_active_markets() -> list[dict]:
    """Fetch all active markets from the Gamma API.

    Uses cursor-based pagination to get all markets.
    Returns raw market dicts for classification/filtering.
    """
    settings = get_settings()
    all_markets: list[dict] = []
    offset = 0
    limit = 100

    async with HttpClient(base_url=settings.gamma_api_url) as client:
        while True:
            resp = await client.get(
                "/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                    "offset": offset,
                },
            )
            markets = resp.json()
            if not markets:
                break
            all_markets.extend(markets)
            if len(markets) < limit:
                break
            offset += limit

    return all_markets


def raw_to_weather_market(raw: dict) -> WeatherMarket:
    """Convert a raw Gamma API market dict to a WeatherMarket.

    Prices come from the CLOB tokens embedded in the Gamma response,
    or default to 0.5 if not available.
    """
    # Extract YES/NO prices from CLOB token data if available
    tokens = raw.get("clobTokenIds", "")
    yes_price = 0.5
    no_price = 0.5

    # outcomePrices is a JSON string like "[\"0.55\",\"0.45\"]"
    market_id = str(raw.get("id", ""))
    outcome_prices = raw.get("outcomePrices", "")
    if outcome_prices:
        try:
            import json
            prices = json.loads(outcome_prices)
            if len(prices) >= 2:
                yes_price = float(prices[0])
                no_price = float(prices[1])
        except (json.JSONDecodeError, ValueError, IndexError):
            logger.debug("Failed to parse outcomePrices for market %s", market_id)

    # Best bid/ask from bestAsk field if available
    best_ask = raw.get("bestAsk")
    if best_ask is not None:
        try:
            yes_price = float(best_ask)
            no_price = 1.0 - yes_price
        except (ValueError, TypeError):
            logger.debug("Failed to parse bestAsk for market %s", market_id)

    # Validate prices are in [0, 1] and sum to ~1.0
    if not (0.0 <= yes_price <= 1.0 and 0.0 <= no_price <= 1.0):
        logger.warning(
            "Market %s has out-of-range prices: YES=%.4f, NO=%.4f — defaulting to 0.5",
            market_id, yes_price, no_price,
        )
        yes_price, no_price = 0.5, 0.5
    elif abs(yes_price + no_price - 1.0) > 0.05:
        logger.warning(
            "Market %s prices don't sum to ~1.0: YES=%.4f + NO=%.4f = %.4f",
            market_id, yes_price, no_price, yes_price + no_price,
        )

    return WeatherMarket(
        market_id=market_id,
        condition_id=raw.get("conditionId", raw.get("condition_id", "")),
        question=raw.get("question", ""),
        description=raw.get("description", ""),
        outcome_yes_price=yes_price,
        outcome_no_price=no_price,
        end_date=_parse_iso(raw.get("endDate") or raw.get("end_date_iso")),
        volume=float(raw.get("volume", 0) or 0),
        liquidity=float(raw.get("liquidity", 0) or 0),
        slug=raw.get("slug", ""),
        active=raw.get("active", True),
    )
