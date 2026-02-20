"""Two-stage weather market classifier.

Stage 1: Regex fast path catches obvious weather markets (free, instant).
Stage 2: Claude Haiku LLM fallback classifies ambiguous ones (~$0.001/call).
"""

from __future__ import annotations

import json
import logging
import re

import anthropic

from weather_edge.common.llm import ask_haiku
from weather_edge.config import get_settings

logger = logging.getLogger(__name__)

# Strong indicators: specific meteorological terms that almost always mean weather
_STRONG_KEYWORDS = re.compile(
    r"("
    r"\btemperature\b|\bfahrenheit\b|\bcelsius\b"
    r"|°[FfCc]|\d+\s*°\s*[FfCc]"
    r"|\brain(?:fall)?\b|\bprecipitation\b|\bprecip\b|\bsnow(?:fall)?\b"
    r"|\binches?\s+of\s+(?:rain|snow)\b"
    r"|\bhurricane\b|\btropical\s+storm\b|\bcyclone\b|\btyphoon\b"
    r"|\bheat\s*wave\b|\bcold\s*snap\b|\bfreeze\b|\bfrost\b"
    r"|\bhigh\s+of\b|\blow\s+of\b|\brecord\s+high\b|\brecord\s+low\b"
    r"|\bwind\s*(?:speed|chill|gust)\b"
    r"|\btornado\b|\bblizzard\b|\bice\s+storm\b"
    r"|\bhighest\s+temperature\b|\blowest\s+temperature\b"
    r"|\btotal\s+rainfall\b|\btotal\s+precipitation\b"
    r")",
    re.IGNORECASE,
)

# Weak indicators: only count these if combined with strong ones
_WEAK_KEYWORDS = re.compile(
    r"\b(temp|degrees?|weather|forecast)\b",
    re.IGNORECASE,
)

# Anti-patterns: markets that mention weather words but aren't weather predictions
_ANTI_PATTERNS = re.compile(
    r"\b("
    r"political\s+storm|brainstorm|weather\s+the\s+storm"
    r"|heated\s+debate|hot\s+take|cold\s+war"
    r"|temperature\s+of\s+(the\s+)?(debate|conversation|discussion)"
    r")\b",
    re.IGNORECASE,
)

_LLM_SYSTEM = """You are a classifier that determines whether a prediction market question
is about a real-world weather event or meteorological outcome.

Respond with ONLY a JSON object:
{"is_weather": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}

Weather markets ask about measurable meteorological outcomes: temperature thresholds,
precipitation amounts, hurricane landfalls, severe weather events, etc.

NOT weather: political metaphors ("storm of controversy"), sports ("heat vs thunder"),
crypto ("weather token"), or figurative language."""


async def classify_market(question: str, description: str = "") -> tuple[bool, float]:
    """Classify whether a market is weather-related.

    Returns:
        (is_weather, confidence) tuple
    """
    # Check anti-patterns first (search both question + description)
    text = f"{question} {description}"
    if _ANTI_PATTERNS.search(text):
        return False, 0.95

    # Stage 1: Regex fast path — only search the QUESTION for keywords.
    # Descriptions often contain tangential weather references in non-weather markets.
    strong_matches = _STRONG_KEYWORDS.findall(question)
    if len(strong_matches) >= 1:
        return True, 0.95

    # Weak keyword alone in question isn't enough — check description too for backup
    weak_matches = _WEAK_KEYWORDS.findall(question)
    if weak_matches:
        # Weak keyword in question + strong keyword in description → likely weather
        if _STRONG_KEYWORDS.search(description):
            return True, 0.80
        # Weak keyword only → ambiguous, try LLM if available
        if get_settings().anthropic_api_key:
            return await _llm_classify(question, description)
        return False, 0.70

    # No keywords at all → probably not weather, but try LLM if description
    # contains any weather-adjacent language
    if description and _WEAK_KEYWORDS.search(description) and get_settings().anthropic_api_key:
        return await _llm_classify(question, description)

    return False, 0.90


async def _llm_classify(question: str, description: str) -> tuple[bool, float]:
    """Use Claude Haiku to classify an ambiguous market."""
    user_prompt = f"Market question: {question}"
    if description:
        user_prompt += f"\nDescription: {description[:500]}"

    try:
        response = await ask_haiku(_LLM_SYSTEM, user_prompt, max_tokens=200)
        # Parse JSON response
        # Find JSON in response (handle markdown code blocks)
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            result = json.loads(json_match.group())
            return bool(result.get("is_weather", False)), float(result.get("confidence", 0.5))
        logger.debug("LLM classifier returned no JSON for: %s", question[:80])
    except json.JSONDecodeError as exc:
        logger.warning("LLM classifier returned invalid JSON: %s", exc)
    except anthropic.APIError as exc:
        logger.warning("LLM classifier API error: %s", exc)
    except (ValueError, TypeError) as exc:
        logger.warning("LLM classifier parse error: %s", exc)

    # If LLM fails, be conservative
    return False, 0.3
