"""Tests for two-stage market parameter parser."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from weather_edge.markets.models import Comparison, MarketType
from weather_edge.markets.parser import parse_market, _regex_parse


# --- Stage 1: Regex fast path tests ---


class TestRegexParse:
    def test_temperature_above(self):
        params = _regex_parse("Will the temperature in Phoenix exceed 120F on July 4, 2025?")
        assert params is not None
        assert params.market_type == MarketType.TEMPERATURE
        assert params.threshold == 120.0
        assert params.unit == "F"
        assert params.comparison == Comparison.ABOVE
        assert "Phoenix" in params.location

    def test_temperature_below(self):
        params = _regex_parse("Will the temperature in Chicago drop below 0F this winter?")
        assert params is not None
        assert params.market_type == MarketType.TEMPERATURE
        assert params.threshold == 0.0
        assert params.comparison == Comparison.BELOW

    def test_temperature_celsius(self):
        params = _regex_parse("Will the temperature in London exceed 40C this summer?")
        assert params is not None
        assert params.threshold == 40.0
        assert params.unit == "C"

    def test_precipitation_inches(self):
        params = _regex_parse("Will rainfall in Houston exceed 5 inches this week?")
        assert params is not None
        assert params.market_type == MarketType.PRECIPITATION
        assert params.threshold == 5.0
        assert params.unit == "in"

    def test_precipitation_mm(self):
        params = _regex_parse("Will precipitation in Tokyo exceed 100 mm?")
        assert params is not None
        assert params.market_type == MarketType.PRECIPITATION
        assert params.threshold == 100.0

    def test_hurricane(self):
        params = _regex_parse("Will a hurricane make landfall in Florida this season?")
        assert params is not None
        assert params.market_type == MarketType.HURRICANE

    def test_tropical_storm(self):
        params = _regex_parse("Will a tropical storm hit the Gulf Coast?")
        assert params is not None
        assert params.market_type == MarketType.HURRICANE

    def test_bare_degrees(self):
        """Test extraction from bare degree notation without 'temperature' keyword."""
        params = _regex_parse("Will the high in Miami reach 105F?")
        assert params is not None
        assert params.market_type == MarketType.TEMPERATURE
        assert params.threshold == 105.0

    def test_non_weather_returns_none(self):
        params = _regex_parse("Will Bitcoin reach $100,000?")
        assert params is None

    def test_date_extraction(self):
        params = _regex_parse("Will the temperature in Denver exceed 100F on August 15, 2025?")
        assert params is not None
        assert params.target_date is not None
        assert params.target_date.month == 8
        assert params.target_date.day == 15

    def test_date_extraction_ordinal(self):
        params = _regex_parse("Will the temperature in Denver exceed 100F on August 3rd, 2025?")
        assert params is not None
        assert params.target_date is not None
        assert params.target_date.day == 3

    def test_location_with_state(self):
        params = _regex_parse("Will the temperature in Miami, Florida exceed 100F?")
        assert params is not None
        assert "Miami, Florida" in params.location

    def test_snowfall(self):
        params = _regex_parse("Will snowfall in Denver exceed 12 inches this month?")
        assert params is not None
        assert params.market_type == MarketType.PRECIPITATION


# --- Stage 2: LLM fallback tests ---


@pytest.mark.asyncio
async def test_parse_market_llm_fallback():
    """Test LLM fallback for creative market phrasings."""
    llm_response = json.dumps({
        "market_type": "temperature",
        "location": "New York, NY",
        "threshold": 100.0,
        "comparison": "above",
        "unit": "F",
        "target_date": "2025-07-04",
        "target_date_str": "Independence Day",
    })

    mock_latlon = (40.7128, -74.0060)

    with patch("weather_edge.markets.parser.ask_haiku", new_callable=AsyncMock) as mock_haiku, \
         patch("weather_edge.markets.parser.geocode", new_callable=AsyncMock) as mock_geo:
        mock_haiku.return_value = llm_response
        mock_geo.return_value = mock_latlon

        params = await parse_market(
            "Will the Big Apple see triple digits on Independence Day?"
        )

    assert params is not None
    assert params.market_type == MarketType.TEMPERATURE
    assert params.location == "New York, NY"
    assert params.threshold == 100.0
    assert params.lat_lon == mock_latlon


@pytest.mark.asyncio
async def test_parse_market_llm_failure():
    """Test that LLM parse failure returns None."""
    import anthropic

    with patch("weather_edge.markets.parser.ask_haiku", new_callable=AsyncMock) as mock_haiku, \
         patch("weather_edge.markets.parser.geocode", new_callable=AsyncMock) as mock_geo:
        mock_haiku.side_effect = anthropic.APIError(
            message="API error",
            request=None,
            body=None,
        )
        mock_geo.return_value = None

        params = await parse_market("Some completely unparseable market")

    assert params is None


@pytest.mark.asyncio
async def test_parse_market_regex_path_with_geocoding():
    """Test that regex-parsed markets get geocoded."""
    mock_latlon = (33.4484, -112.074)

    with patch("weather_edge.markets.parser.geocode", new_callable=AsyncMock) as mock_geo:
        mock_geo.return_value = mock_latlon

        params = await parse_market(
            "Will the temperature in Phoenix exceed 120F?"
        )

    assert params is not None
    assert params.lat_lon == mock_latlon
    mock_geo.assert_called_once()


@pytest.mark.asyncio
async def test_parse_market_geocoding_failure():
    """Test handling when geocoding fails."""
    with patch("weather_edge.markets.parser.geocode", new_callable=AsyncMock) as mock_geo:
        mock_geo.return_value = None

        params = await parse_market(
            "Will the temperature in Phoenix exceed 120F?"
        )

    assert params is not None
    assert params.lat_lon is None
    assert params.location == "Phoenix"
