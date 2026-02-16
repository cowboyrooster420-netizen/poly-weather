"""Tests for two-stage market parameter parser."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from weather_edge.markets.models import Comparison, MarketType
from weather_edge.markets.parser import parse_market, _regex_parse, _resolve_period


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

    def test_precipitation_between(self):
        """BETWEEN pattern extracts lower/upper thresholds."""
        params = _regex_parse(
            "Will Seattle have between 5 and 6 inches of rain in February?"
        )
        assert params is not None
        assert params.market_type == MarketType.PRECIPITATION
        assert params.comparison == Comparison.BETWEEN
        assert params.threshold == 5.0
        assert params.threshold_upper == 6.0
        assert params.unit == "in"

    def test_precipitation_period_month(self):
        """'in February' should populate period_start/period_end."""
        params = _regex_parse(
            "Will rainfall in Seattle exceed 5 inches in February?"
        )
        assert params is not None
        assert params.period_start is not None
        assert params.period_end is not None
        assert params.period_start.month == 2
        assert params.period_start.day == 1
        assert params.period_end.month == 2
        assert params.period_end.day == 28 or params.period_end.day == 29

    def test_temperature_between(self):
        """BETWEEN pattern for temperature bucket markets."""
        params = _regex_parse(
            "Highest temperature in London between 45F and 50F on February 16?"
        )
        assert params is not None
        assert params.market_type == MarketType.TEMPERATURE
        assert params.comparison == Comparison.BETWEEN
        assert params.threshold == 45.0
        assert params.threshold_upper == 50.0
        assert params.unit == "F"

    def test_temperature_between_degrees(self):
        """BETWEEN with degree symbols and dash separator."""
        params = _regex_parse(
            "Will the high in London be between 7°C and 10°C?"
        )
        assert params is not None
        assert params.comparison == Comparison.BETWEEN
        assert params.threshold == 7.0
        assert params.threshold_upper == 10.0
        assert params.unit == "C"

    def test_temperature_bucket_exact(self):
        """Polymarket single-degree bucket: 'be 7°C on' → BETWEEN 6.5 and 7.5."""
        params = _regex_parse(
            "Will the highest temperature in London be 7°C on February 16?"
        )
        assert params is not None
        assert params.market_type == MarketType.TEMPERATURE
        assert params.comparison == Comparison.BETWEEN
        assert params.threshold == 6.5
        assert params.threshold_upper == 7.5
        assert params.unit == "C"

    def test_temperature_bucket_or_below(self):
        """Polymarket edge bucket: 'be 4°C or below'."""
        params = _regex_parse(
            "Will the highest temperature in London be 4°C or below on February 16?"
        )
        assert params is not None
        assert params.market_type == MarketType.TEMPERATURE
        assert params.comparison == Comparison.BELOW
        assert params.threshold == 4.0
        assert params.unit == "C"

    def test_temperature_bucket_or_higher(self):
        """Polymarket edge bucket: 'be 12°C or higher'."""
        params = _regex_parse(
            "Will the highest temperature in London be 12°C or higher on February 16?"
        )
        assert params is not None
        assert params.market_type == MarketType.TEMPERATURE
        assert params.comparison == Comparison.ABOVE
        assert params.threshold == 12.0
        assert params.unit == "C"

    def test_temperature_no_period(self):
        """Temperature markets should never get period_start/period_end."""
        params = _regex_parse("Will the temperature in Phoenix exceed 120F in July?")
        assert params is not None
        assert params.market_type == MarketType.TEMPERATURE
        assert params.period_start is None
        assert params.period_end is None

    # --- Polymarket F-range bucket tests ---

    def test_f_range_bucket(self):
        """Polymarket F-range: '32-33°F' → BETWEEN."""
        params = _regex_parse(
            "Will the highest temperature in New York City be 32-33°F on February 16?"
        )
        assert params is not None
        assert params.market_type == MarketType.TEMPERATURE
        assert params.comparison == Comparison.BETWEEN
        assert params.threshold == 32.0
        assert params.threshold_upper == 33.0
        assert params.unit == "F"

    def test_f_range_bucket_higher_range(self):
        """Polymarket F-range: '38-39°F'."""
        params = _regex_parse(
            "Will the highest temperature in Atlanta be 38-39°F on February 16?"
        )
        assert params is not None
        assert params.comparison == Comparison.BETWEEN
        assert params.threshold == 38.0
        assert params.threshold_upper == 39.0
        assert params.unit == "F"

    # --- Negative temperature tests (Toronto) ---

    def test_negative_celsius_bucket(self):
        """Polymarket: 'be -2°C on' → BETWEEN -2.5 and -1.5."""
        params = _regex_parse(
            "Will the highest temperature in Toronto be -2°C on February 17?"
        )
        assert params is not None
        assert params.market_type == MarketType.TEMPERATURE
        assert params.comparison == Comparison.BETWEEN
        assert params.threshold == -2.5
        assert params.threshold_upper == -1.5
        assert params.unit == "C"

    def test_negative_celsius_or_below(self):
        """Polymarket: 'be -5°C or below'."""
        params = _regex_parse(
            "Will the highest temperature in Toronto be -5°C or below on February 17?"
        )
        assert params is not None
        assert params.comparison == Comparison.BELOW
        assert params.threshold == -5.0
        assert params.unit == "C"

    def test_negative_f_range(self):
        """Negative F-range: '-2--1°F'."""
        params = _regex_parse(
            "Will the highest temperature in Chicago be -2-0°F on January 20?"
        )
        assert params is not None
        assert params.comparison == Comparison.BETWEEN
        assert params.threshold == -2.0
        assert params.threshold_upper == 0.0

    # --- Precipitation inch-mark tests ---

    def test_precip_inch_mark_range(self):
        """Polymarket: '3-4"' → BETWEEN 3 and 4 inches."""
        params = _regex_parse(
            'Will total rainfall in Seattle, WA in February 2026 be 3-4"?'
        )
        assert params is not None
        assert params.market_type == MarketType.PRECIPITATION
        assert params.comparison == Comparison.BETWEEN
        assert params.threshold == 3.0
        assert params.threshold_upper == 4.0
        assert params.unit == "in"

    def test_precip_inch_mark_above(self):
        """Polymarket: '>8"' → ABOVE 8 inches."""
        params = _regex_parse(
            'Will total rainfall in Seattle, WA in February 2026 be >8"?'
        )
        assert params is not None
        assert params.market_type == MarketType.PRECIPITATION
        assert params.comparison == Comparison.ABOVE
        assert params.threshold == 8.0
        assert params.unit == "in"

    def test_precip_inch_mark_below(self):
        """Polymarket: '<3"' → BELOW 3 inches."""
        params = _regex_parse(
            'Will total rainfall in Seattle, WA in February 2026 be <3"?'
        )
        assert params is not None
        assert params.market_type == MarketType.PRECIPITATION
        assert params.comparison == Comparison.BELOW
        assert params.threshold == 3.0
        assert params.unit == "in"

    def test_precip_period_month_with_year(self):
        """'in February 2026' should populate period_start/period_end."""
        params = _regex_parse(
            'Will total rainfall in Seattle, WA in February 2026 be 3-4"?'
        )
        assert params is not None
        assert params.period_start is not None
        assert params.period_end is not None
        assert params.period_start.year == 2026
        assert params.period_start.month == 2
        assert params.period_start.day == 1
        assert params.period_end.month == 2
        assert params.period_end.day == 28

    def test_date_extraction_no_year(self):
        """'on February 16' (no year) still extracts a date."""
        params = _regex_parse(
            "Will the highest temperature in London be 7°C on February 16?"
        )
        assert params is not None
        assert params.target_date is not None
        assert params.target_date.month == 2
        assert params.target_date.day == 16


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
