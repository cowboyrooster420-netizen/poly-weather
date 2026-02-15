"""Tests for Polymarket Gamma API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weather_edge.markets.client import fetch_all_active_markets, raw_to_weather_market


@pytest.mark.asyncio
async def test_fetch_all_active_markets(gamma_api_response):
    """Test fetching and paginating active markets."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = gamma_api_response
    mock_resp.raise_for_status = MagicMock()

    # Second call returns empty (end of pagination)
    mock_resp_empty = MagicMock()
    mock_resp_empty.json.return_value = []
    mock_resp_empty.raise_for_status = MagicMock()

    with patch("weather_edge.markets.client.HttpClient") as MockClient:
        instance = AsyncMock()
        instance.get = AsyncMock(side_effect=[mock_resp, mock_resp_empty])
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        markets = await fetch_all_active_markets()

    assert len(markets) == 4
    assert markets[0]["id"] == "market-weather-1"


@pytest.mark.asyncio
async def test_fetch_pagination():
    """Test that pagination stops when response has fewer items than limit."""
    # Create exactly 100 items to trigger a second page fetch
    page1 = [{"id": f"m-{i}", "question": f"Q{i}"} for i in range(100)]
    page2 = [{"id": "m-100", "question": "Q100"}]

    mock_resp1 = MagicMock()
    mock_resp1.json.return_value = page1
    mock_resp2 = MagicMock()
    mock_resp2.json.return_value = page2

    with patch("weather_edge.markets.client.HttpClient") as MockClient:
        instance = AsyncMock()
        instance.get = AsyncMock(side_effect=[mock_resp1, mock_resp2])
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        markets = await fetch_all_active_markets()

    assert len(markets) == 101


def test_raw_to_weather_market_with_prices():
    """Test conversion of raw market data with outcome prices."""
    raw = {
        "id": "test-123",
        "conditionId": "cond-123",
        "question": "Will it rain?",
        "description": "Rain market",
        "outcomePrices": '["0.65","0.35"]',
        "active": True,
        "volume": "50000",
        "liquidity": "10000",
        "slug": "rain-market",
        "endDate": "2025-07-01T00:00:00Z",
    }

    market = raw_to_weather_market(raw)

    assert market.market_id == "test-123"
    assert market.condition_id == "cond-123"
    assert market.question == "Will it rain?"
    assert abs(market.outcome_yes_price - 0.65) < 0.001
    assert abs(market.outcome_no_price - 0.35) < 0.001
    assert market.volume == 50000.0
    assert market.slug == "rain-market"
    assert market.end_date is not None
    assert market.market_prob == 0.65


def test_raw_to_weather_market_no_prices():
    """Test conversion when no price data is available."""
    raw = {
        "id": "test-456",
        "question": "Market question",
        "description": "",
    }

    market = raw_to_weather_market(raw)

    assert market.outcome_yes_price == 0.5
    assert market.outcome_no_price == 0.5


def test_raw_to_weather_market_malformed_prices():
    """Test handling of malformed price JSON."""
    raw = {
        "id": "test-789",
        "question": "Market question",
        "description": "",
        "outcomePrices": "not-valid-json",
    }

    market = raw_to_weather_market(raw)

    assert market.outcome_yes_price == 0.5  # Falls back to default


def test_raw_to_weather_market_best_ask():
    """Test that bestAsk field overrides outcomePrices."""
    raw = {
        "id": "test-ask",
        "question": "Market question",
        "description": "",
        "outcomePrices": '["0.40","0.60"]',
        "bestAsk": 0.55,
    }

    market = raw_to_weather_market(raw)

    assert abs(market.outcome_yes_price - 0.55) < 0.001
    assert abs(market.outcome_no_price - 0.45) < 0.001
