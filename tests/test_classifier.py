"""Tests for two-stage weather market classifier."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from weather_edge.markets.classifier import classify_market


# --- Stage 1: Regex fast path tests ---


@pytest.mark.asyncio
async def test_classify_obvious_weather_temperature():
    """Obvious temperature market — regex catches it."""
    is_weather, conf = await classify_market(
        "Will the temperature in Phoenix exceed 120F on July 4?"
    )
    assert is_weather is True
    assert conf >= 0.85


@pytest.mark.asyncio
async def test_classify_obvious_weather_hurricane():
    """Obvious hurricane market — regex catches it."""
    is_weather, conf = await classify_market(
        "Will a hurricane make landfall in Florida this season?"
    )
    assert is_weather is True
    assert conf >= 0.85


@pytest.mark.asyncio
async def test_classify_obvious_weather_precipitation():
    """Obvious precipitation market — regex catches it."""
    is_weather, conf = await classify_market(
        "Will rainfall in Houston exceed 5 inches this week?"
    )
    assert is_weather is True
    assert conf >= 0.85


@pytest.mark.asyncio
async def test_classify_obvious_weather_snow():
    """Obvious snow market — regex catches it."""
    is_weather, conf = await classify_market(
        "Will snowfall in Denver exceed 12 inches?"
    )
    assert is_weather is True
    assert conf >= 0.85


@pytest.mark.asyncio
async def test_classify_multiple_keywords_high_confidence():
    """Multiple weather keywords → higher confidence."""
    is_weather, conf = await classify_market(
        "Will the temperature reach record high levels and cause a heat wave?"
    )
    assert is_weather is True
    assert conf >= 0.90


@pytest.mark.asyncio
async def test_classify_obvious_non_weather():
    """Obvious non-weather market — regex rejects it."""
    is_weather, conf = await classify_market(
        "Will the candidate win the 2025 presidential election?"
    )
    assert is_weather is False
    assert conf >= 0.80


@pytest.mark.asyncio
async def test_classify_anti_pattern_political_storm():
    """Figurative 'storm' usage — anti-pattern catches it."""
    is_weather, conf = await classify_market(
        "Will the politician weather the storm of controversy?"
    )
    assert is_weather is False
    assert conf >= 0.90


@pytest.mark.asyncio
async def test_classify_anti_pattern_heated_debate():
    """Figurative 'heated' usage — anti-pattern catches it."""
    is_weather, conf = await classify_market(
        "Will there be a heated debate at the summit?"
    )
    assert is_weather is False
    assert conf >= 0.90


@pytest.mark.asyncio
async def test_classify_anti_pattern_cold_war():
    """Figurative 'cold war' usage — anti-pattern catches it."""
    is_weather, conf = await classify_market(
        "Will the cold war between the two nations escalate?"
    )
    assert is_weather is False
    assert conf >= 0.90


@pytest.mark.asyncio
async def test_classify_no_keywords_returns_false():
    """Market with no weather keywords returns False from regex path."""
    is_weather, conf = await classify_market(
        "Will the heat continue?"  # "heat" alone doesn't match heat\s*wave
    )
    assert is_weather is False
    assert conf >= 0.80


@pytest.mark.asyncio
async def test_classify_crypto_storm_not_weather():
    """Crypto token 'Storm' should not match weather patterns."""
    is_weather, conf = await classify_market(
        "Will Storm token reach $1?",
        description="Cryptocurrency market for Storm (STORM) token.",
    )
    assert is_weather is False
