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

logger = logging.getLogger(__name__)

# Regex patterns for obvious weather markets
_WEATHER_KEYWORDS = re.compile(
    r"\b("
    r"temperature|temp|degrees?|fahrenheit|celsius"
    r"|rain(fall)?|precipitation|precip|snow(fall)?|inches?\s+of\s+(rain|snow)"
    r"|hurricane|tropical\s+storm|cyclone|typhoon"
    r"|heat\s*wave|cold\s*snap|freeze|frost"
    r"|high\s+of|low\s+of|record\s+high|record\s+low"
    r"|weather|forecast"
    r"|wind\s*(speed|chill|gust)"
    r"|tornado|blizzard|ice\s+storm"
    r")\b",
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
    # Stage 1: Regex fast path
    text = f"{question} {description}"

    # Check anti-patterns first
    if _ANTI_PATTERNS.search(text):
        return False, 0.95

    # Check weather keywords
    matches = _WEATHER_KEYWORDS.findall(text)
    if len(matches) >= 2:
        return True, 0.95
    if len(matches) == 1:
        return True, 0.85
    if len(matches) == 0:
        return False, 0.85

    # Stage 2: LLM fallback for ambiguous cases
    return await _llm_classify(question, description)


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
