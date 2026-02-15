"""Thin wrapper around Anthropic SDK for Claude Haiku calls."""

from __future__ import annotations

import logging

import anthropic

from weather_edge.config import get_settings

logger = logging.getLogger(__name__)


async def ask_haiku(system: str, user: str, max_tokens: int = 1024) -> str:
    """Send a prompt to Claude Haiku and return the text response.

    Used for market classification and parameter extraction fallback.
    Cost is ~$0.001 per call.
    """
    settings = get_settings()
    async with anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key) as client:
        message = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if not message.content:
            raise ValueError("Claude returned empty content")
        return message.content[0].text
