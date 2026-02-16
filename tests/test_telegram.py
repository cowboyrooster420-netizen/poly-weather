"""Tests for Telegram notification formatters and notifier."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from weather_edge.signals.formatters import format_telegram_signal, format_telegram_summary
from weather_edge.signals.models import Signal


def _make_signal(
    direction: str = "YES",
    edge: float = 0.123,
    kelly: float = 0.032,
    model_prob: float = 0.673,
    market_prob: float = 0.55,
    confidence: float = 0.82,
    question: str = "Will Phoenix temperature exceed 115F by July 15?",
    location: str = "Phoenix, AZ",
    market_type: str = "temperature",
    lead_time: float = 72.0,
    sources: list[str] | None = None,
) -> Signal:
    return Signal(
        market_id="abc123",
        question=question,
        market_type=market_type,
        location=location,
        model_prob=model_prob,
        market_prob=market_prob,
        edge=edge,
        kelly_fraction=kelly,
        confidence=confidence,
        direction=direction,
        lead_time_hours=lead_time,
        sources=sources or ["ECMWF (51 members)", "GFS (31 members)"],
        details="test details",
        timestamp=datetime(2026, 2, 15, 18, 30, 0),
    )


# ---- Formatter tests ----


class TestFormatTelegramSignal:
    def test_contains_high_edge_header(self):
        signal = _make_signal()
        text = format_telegram_signal(signal)
        assert "*HIGH EDGE SIGNAL*" in text

    def test_contains_direction(self):
        signal = _make_signal(direction="YES")
        text = format_telegram_signal(signal)
        assert "*YES*" in text

    def test_contains_edge_percentage(self):
        signal = _make_signal(edge=0.123)
        text = format_telegram_signal(signal)
        assert "+12.3%" in text

    def test_negative_edge(self):
        signal = _make_signal(direction="NO", edge=-0.081)
        text = format_telegram_signal(signal)
        assert "*NO*" in text
        assert "-8.1%" in text

    def test_contains_kelly(self):
        signal = _make_signal(kelly=0.032)
        text = format_telegram_signal(signal)
        assert "3.2%" in text
        assert "bankroll" in text

    def test_contains_question(self):
        signal = _make_signal(question="Will Phoenix temperature exceed 115F by July 15?")
        text = format_telegram_signal(signal)
        assert "Phoenix temperature exceed 115F" in text

    def test_contains_location_and_type(self):
        signal = _make_signal(location="Phoenix, AZ", market_type="temperature")
        text = format_telegram_signal(signal)
        assert "Phoenix, AZ" in text
        assert "temperature" in text

    def test_contains_lead_time_and_confidence(self):
        signal = _make_signal(lead_time=72.0, confidence=0.82)
        text = format_telegram_signal(signal)
        assert "72.0h" in text
        assert "0.82" in text

    def test_contains_model_and_market_probs(self):
        signal = _make_signal(model_prob=0.673, market_prob=0.55)
        text = format_telegram_signal(signal)
        assert "67.3%" in text
        assert "55.0%" in text

    def test_contains_sources(self):
        signal = _make_signal(sources=["ECMWF (51 members)", "GFS (31 members)"])
        text = format_telegram_signal(signal)
        assert "ECMWF (51 members)" in text
        assert "GFS (31 members)" in text

    def test_long_question_truncated(self):
        long_q = "A" * 200
        signal = _make_signal(question=long_q)
        text = format_telegram_signal(signal)
        # Question should be truncated to 120 chars
        assert "A" * 120 in text
        assert "A" * 121 not in text


class TestFormatTelegramSummary:
    def test_contains_header(self):
        text = format_telegram_summary([])
        assert "*Weather Edge Scan Complete*" in text

    def test_zero_signals(self):
        text = format_telegram_summary([])
        assert "0 signal(s) generated" in text

    def test_signal_count(self):
        signals = [_make_signal(), _make_signal(edge=0.05), _make_signal(edge=-0.081)]
        text = format_telegram_summary(signals)
        assert "3 signal(s) generated" in text

    def test_contains_table_rows(self):
        signals = [
            _make_signal(direction="YES", edge=0.123, kelly=0.032),
            _make_signal(direction="NO", edge=-0.081, kelly=0.015),
        ]
        text = format_telegram_summary(signals)
        assert "| YES |" in text
        assert "| NO |" in text
        assert "+12.3%" in text
        assert "-8.1%" in text

    def test_sorted_by_absolute_edge(self):
        signals = [
            _make_signal(edge=0.05),
            _make_signal(edge=-0.15),
            _make_signal(edge=0.10),
        ]
        text = format_telegram_summary(signals)
        lines = text.split("\n")
        table_lines = [l for l in lines if l.startswith("| ") and "Dir" not in l]
        assert len(table_lines) == 3
        # First row should be highest absolute edge (-15%)
        assert "-15.0%" in table_lines[0]

    def test_contains_timestamp(self):
        text = format_telegram_summary([])
        # Should contain a UTC timestamp
        assert "UTC" in text

    def test_truncates_to_fit_limit(self):
        """Many signals should be truncated to respect max_chars."""
        signals = [_make_signal(edge=0.01 * i) for i in range(1, 101)]
        text = format_telegram_summary(signals, max_chars=1000)
        assert len(text) <= 1000
        assert "100 signal(s) generated" in text
        assert "... and" in text
        assert "more" in text


# ---- Notifier tests ----


class TestTelegramNotifier:
    @pytest.mark.asyncio
    async def test_no_op_when_not_configured(self):
        """Notifier should be a no-op when bot_token/chat_id are empty."""
        from weather_edge.notifications.telegram import TelegramNotifier

        notifier = TelegramNotifier(bot_token="", chat_id="")
        result = await notifier.send_message("test")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_posts_to_correct_endpoint(self):
        """send_message should POST to /bot{token}/sendMessage."""
        from weather_edge.notifications.telegram import TelegramNotifier

        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="999")

        mock_response = httpx.Response(200, json={"ok": True})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        notifier._client = mock_client

        result = await notifier.send_message("Hello")
        assert result is True

        mock_client.post.assert_called_once_with(
            "/bot123:ABC/sendMessage",
            json={
                "chat_id": "999",
                "text": "Hello",
                "parse_mode": "Markdown",
            },
        )

    @pytest.mark.asyncio
    async def test_send_message_catches_exceptions(self):
        """send_message should catch and log errors, never raise."""
        from weather_edge.notifications.telegram import TelegramNotifier

        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="999")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.HTTPStatusError(
            "Unauthorized", request=httpx.Request("POST", "https://example.com"), response=httpx.Response(401),
        ))
        notifier._client = mock_client

        result = await notifier.send_message("Hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_notify_sends_high_edge_alerts_and_summary(self):
        """notify should send individual alerts for high-edge signals + summary."""
        from weather_edge.notifications.telegram import TelegramNotifier

        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="999")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=httpx.Response(200, json={"ok": True}))
        mock_client.close = AsyncMock()
        notifier._client = mock_client

        signals = [
            _make_signal(edge=0.12),   # above 10% threshold
            _make_signal(edge=0.05),   # below threshold
            _make_signal(edge=-0.11),  # above threshold (absolute)
        ]

        await notifier.notify(signals, high_edge_threshold=0.10)

        # 2 high-edge alerts + 1 summary = 3 calls
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_notify_no_op_when_disabled(self):
        """notify should do nothing when not configured."""
        from weather_edge.notifications.telegram import TelegramNotifier

        notifier = TelegramNotifier(bot_token="", chat_id="")

        signals = [_make_signal(edge=0.15)]
        await notifier.notify(signals)
        # No exception raised, no calls made

    @pytest.mark.asyncio
    async def test_notify_signal_formats_and_sends(self):
        """notify_signal should format the signal and send it."""
        from weather_edge.notifications.telegram import TelegramNotifier

        notifier = TelegramNotifier(bot_token="123:ABC", chat_id="999")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=httpx.Response(200, json={"ok": True}))
        notifier._client = mock_client

        signal = _make_signal()
        result = await notifier.notify_signal(signal)
        assert result is True

        call_args = mock_client.post.call_args
        sent_text = call_args.kwargs["json"]["text"]
        assert "*HIGH EDGE SIGNAL*" in sent_text
