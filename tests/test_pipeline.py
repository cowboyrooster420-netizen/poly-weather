"""Tests for the full pipeline with mocked dependencies."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from weather_edge.markets.models import Comparison, MarketParams, MarketType, WeatherMarket
from weather_edge.pipeline import (
    fetch_weather_data,
    generate_signals,
    run_forecasts,
    run_pipeline,
    scan_markets,
)
from weather_edge.weather.models import EnsembleForecast


@pytest.fixture
def mock_raw_markets(gamma_api_response):
    return gamma_api_response


@pytest.fixture
def mock_weather_markets(now):
    """Pre-built weather markets with params and lat/lon."""
    return [
        WeatherMarket(
            market_id="m1",
            condition_id="c1",
            question="Will Phoenix exceed 100F?",
            description="Temp market",
            outcome_yes_price=0.35,
            outcome_no_price=0.65,
            params=MarketParams(
                market_type=MarketType.TEMPERATURE,
                location="Phoenix, AZ",
                lat_lon=(33.45, -112.07),
                threshold=100.0,
                comparison=Comparison.ABOVE,
                unit="F",
                target_date=now + timedelta(hours=24),
            ),
        ),
        WeatherMarket(
            market_id="m2",
            condition_id="c2",
            question="Will Houston get 5 inches of rain?",
            description="Precip market",
            outcome_yes_price=0.40,
            outcome_no_price=0.60,
            params=MarketParams(
                market_type=MarketType.PRECIPITATION,
                location="Houston, TX",
                lat_lon=(29.76, -95.37),
                threshold=5.0,
                comparison=Comparison.ABOVE,
                unit="in",
                target_date=now + timedelta(hours=48),
            ),
        ),
    ]


def _make_ensemble(lat, lon, source, n_members, now):
    """Create a synthetic ensemble for testing."""
    np.random.seed(42 if source == "ecmwf" else 43)
    times = [now + timedelta(hours=i) for i in range(48)]
    n = len(times)
    return EnsembleForecast(
        source=source,
        lat=lat,
        lon=lon,
        times=times,
        temperature_2m=np.random.normal(35, 3, (n, n_members)),
        precipitation=np.maximum(0, np.random.exponential(2, (n, n_members)) - 1),
    )


@pytest.mark.asyncio
async def test_scan_markets_filters_weather(gamma_api_response):
    """scan_markets should filter to only weather markets."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = gamma_api_response
    mock_resp_empty = MagicMock()
    mock_resp_empty.json.return_value = []

    with patch("weather_edge.pipeline.fetch_all_active_markets", new_callable=AsyncMock) as mock_fetch, \
         patch("weather_edge.pipeline.classify_market", new_callable=AsyncMock) as mock_classify, \
         patch("weather_edge.pipeline.parse_market", new_callable=AsyncMock) as mock_parse:

        mock_fetch.return_value = gamma_api_response

        # Classify: first 2 and last 1 are weather, third is politics
        async def classify_side_effect(question, description=""):
            if "election" in question.lower() or "candidate" in question.lower():
                return (False, 0.95)
            return (True, 0.90)

        mock_classify.side_effect = classify_side_effect
        mock_parse.return_value = MarketParams(
            market_type=MarketType.TEMPERATURE,
            location="Test",
        )

        markets = await scan_markets()

    assert len(markets) == 3  # 4 total - 1 politics = 3 weather


@pytest.mark.asyncio
async def test_fetch_weather_data_groups_by_location(mock_weather_markets, now):
    """Weather data should be fetched once per unique location."""
    fetch_count = 0

    async def mock_fetch_both(lat, lon):
        nonlocal fetch_count
        fetch_count += 1
        ecmwf = _make_ensemble(lat, lon, "ecmwf", 51, now)
        gfs = _make_ensemble(lat, lon, "gfs", 31, now)
        return gfs, ecmwf

    async def mock_fetch_noaa(lat, lon):
        return None

    with patch("weather_edge.pipeline.fetch_both_ensembles", side_effect=mock_fetch_both), \
         patch("weather_edge.pipeline.fetch_noaa_forecast", side_effect=mock_fetch_noaa):
        weather_data = await fetch_weather_data(mock_weather_markets)

    # Two different locations → two fetches
    assert fetch_count == 2
    assert len(weather_data) == 2


@pytest.mark.asyncio
async def test_fetch_weather_data_deduplicates(now):
    """Markets at the same location should share one fetch."""
    markets = [
        WeatherMarket(
            market_id=f"m{i}",
            condition_id=f"c{i}",
            question=f"Market {i}",
            description="",
            outcome_yes_price=0.5,
            outcome_no_price=0.5,
            params=MarketParams(
                market_type=MarketType.TEMPERATURE,
                location="Phoenix, AZ",
                lat_lon=(33.45, -112.07),
                threshold=100.0,
            ),
        )
        for i in range(5)
    ]

    fetch_count = 0

    async def mock_fetch_both(lat, lon):
        nonlocal fetch_count
        fetch_count += 1
        ecmwf = _make_ensemble(lat, lon, "ecmwf", 51, now)
        gfs = _make_ensemble(lat, lon, "gfs", 31, now)
        return gfs, ecmwf

    async def mock_fetch_noaa(lat, lon):
        return None

    with patch("weather_edge.pipeline.fetch_both_ensembles", side_effect=mock_fetch_both), \
         patch("weather_edge.pipeline.fetch_noaa_forecast", side_effect=mock_fetch_noaa):
        weather_data = await fetch_weather_data(markets)

    # All 5 markets share one location → only one fetch
    assert fetch_count == 1
    assert len(weather_data) == 1


@pytest.mark.asyncio
async def test_fetch_weather_data_handles_errors(mock_weather_markets, now):
    """Fetch errors should be caught, not crash pipeline."""

    async def mock_fetch_both(lat, lon):
        raise Exception("Open-Meteo down")

    async def mock_fetch_noaa(lat, lon):
        return None

    with patch("weather_edge.pipeline.fetch_both_ensembles", side_effect=mock_fetch_both), \
         patch("weather_edge.pipeline.fetch_noaa_forecast", side_effect=mock_fetch_noaa):
        weather_data = await fetch_weather_data(mock_weather_markets)

    # Should still have entries (with None values) not crash
    assert len(weather_data) == 2
    for key, val in weather_data.items():
        assert val == (None, None, None)


@pytest.mark.asyncio
async def test_run_forecasts(mock_weather_markets, now):
    """Forecast models should produce estimates for each market."""
    lat, lon = 33.45, -112.07
    key = (round(lat, 2), round(lon, 2))
    ecmwf = _make_ensemble(lat, lon, "ecmwf", 51, now)
    gfs = _make_ensemble(lat, lon, "gfs", 31, now)

    lat2, lon2 = 29.76, -95.37
    key2 = (round(lat2, 2), round(lon2, 2))
    ecmwf2 = _make_ensemble(lat2, lon2, "ecmwf", 51, now)
    gfs2 = _make_ensemble(lat2, lon2, "gfs", 31, now)

    weather_data = {
        key: (gfs, ecmwf, None),
        key2: (gfs2, ecmwf2, None),
    }

    results = await run_forecasts(mock_weather_markets, weather_data)

    assert len(results) == 2
    for market, estimate in results:
        assert 0 < estimate.probability < 1
        assert estimate.confidence > 0


@pytest.mark.asyncio
async def test_generate_signals_filters_by_edge(now):
    """Only signals with sufficient edge should be generated."""
    from weather_edge.forecasting.base import ProbabilityEstimate

    market1 = WeatherMarket(
        market_id="m1", condition_id="c1",
        question="Q1", description="",
        outcome_yes_price=0.50, outcome_no_price=0.50,
        params=MarketParams(market_type=MarketType.TEMPERATURE, location="A"),
    )
    market2 = WeatherMarket(
        market_id="m2", condition_id="c2",
        question="Q2", description="",
        outcome_yes_price=0.30, outcome_no_price=0.70,
        params=MarketParams(market_type=MarketType.TEMPERATURE, location="B"),
    )

    # 2% edge — below 5% threshold
    est1 = ProbabilityEstimate(
        probability=0.52, raw_probability=0.52,
        confidence=0.8, lead_time_hours=24,
    )
    # 20% edge — above threshold
    est2 = ProbabilityEstimate(
        probability=0.50, raw_probability=0.50,
        confidence=0.8, lead_time_hours=24,
    )

    results = [(market1, est1), (market2, est2)]
    signals = await generate_signals(results)

    assert len(signals) == 1
    assert signals[0].market_id == "m2"


@pytest.mark.asyncio
async def test_run_pipeline_end_to_end(gamma_api_response, now):
    """Full pipeline end-to-end with all dependencies mocked."""

    async def mock_classify(question, description=""):
        if "temperature" in question.lower() or "rainfall" in question.lower() or "hurricane" in question.lower():
            return (True, 0.90)
        return (False, 0.95)

    async def mock_parse(question, description=""):
        return MarketParams(
            market_type=MarketType.TEMPERATURE,
            location="Phoenix, AZ",
            lat_lon=(33.45, -112.07),
            threshold=100.0,
            comparison=Comparison.ABOVE,
            unit="F",
            target_date=now + timedelta(hours=24),
        )

    ecmwf = _make_ensemble(33.45, -112.07, "ecmwf", 51, now)
    gfs = _make_ensemble(33.45, -112.07, "gfs", 31, now)

    with patch("weather_edge.pipeline.fetch_all_active_markets", new_callable=AsyncMock) as mock_fetch_markets, \
         patch("weather_edge.pipeline.classify_market", side_effect=mock_classify), \
         patch("weather_edge.pipeline.parse_market", side_effect=mock_parse), \
         patch("weather_edge.pipeline.fetch_both_ensembles", new_callable=AsyncMock) as mock_fetch_ens, \
         patch("weather_edge.pipeline.fetch_noaa_forecast", new_callable=AsyncMock) as mock_fetch_noaa, \
         patch("weather_edge.signals.tracker.SignalTracker.log_signals", new_callable=AsyncMock) as mock_log:

        mock_fetch_markets.return_value = gamma_api_response
        mock_fetch_ens.return_value = (gfs, ecmwf)
        mock_fetch_noaa.return_value = None
        mock_log.return_value = [1, 2, 3]

        signals = await run_pipeline()

    # Should have found weather markets and produced some signals (or not, depending on edge)
    assert isinstance(signals, list)
    # All signals should have valid fields
    for s in signals:
        assert 0 <= s.model_prob <= 1
        assert 0 <= s.market_prob <= 1
        assert s.direction in ("YES", "NO")
        assert s.kelly_fraction >= 0
