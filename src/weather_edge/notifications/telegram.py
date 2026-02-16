"""Telegram Bot API notifications for high-edge signals and run summaries."""

from __future__ import annotations

import logging

from weather_edge.common.http import HttpClient
from weather_edge.config import get_settings
from weather_edge.signals.formatters import format_telegram_signal, format_telegram_summary
from weather_edge.signals.models import Signal

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send trading signal alerts via Telegram Bot API.

    All errors are logged but never raised — notifications must not break
    the pipeline.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        settings = get_settings()
        self._bot_token = bot_token or settings.telegram_bot_token
        self._chat_id = chat_id or settings.telegram_chat_id
        self._client: HttpClient | None = None

    @property
    def _enabled(self) -> bool:
        return bool(self._bot_token and self._chat_id)

    async def _get_client(self) -> HttpClient:
        if self._client is None:
            self._client = HttpClient(base_url="https://api.telegram.org")
        return self._client

    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a message via the Telegram Bot API.

        Returns True if the message was sent successfully, False otherwise.
        """
        if not self._enabled:
            logger.debug("Telegram not configured, skipping message")
            return False

        try:
            client = await self._get_client()
            await client.post(
                f"/bot{self._bot_token}/sendMessage",
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
            )
            return True
        except Exception:
            logger.warning("Failed to send Telegram message", exc_info=True)
            return False

    async def notify_signal(self, signal: Signal) -> bool:
        """Format and send a single high-edge signal alert."""
        text = format_telegram_signal(signal)
        return await self.send_message(text)

    async def notify_summary(self, signals: list[Signal]) -> bool:
        """Format and send a run summary."""
        text = format_telegram_summary(signals)
        return await self.send_message(text)

    async def notify(
        self,
        signals: list[Signal],
        high_edge_threshold: float | None = None,
    ) -> None:
        """Orchestrate notifications: per-signal alerts for high-edge, then summary.

        Args:
            signals: All signals from the pipeline run.
            high_edge_threshold: Edge threshold for individual alerts.
                Defaults to settings.telegram_high_edge.
        """
        if not self._enabled:
            return

        if high_edge_threshold is None:
            high_edge_threshold = get_settings().telegram_high_edge

        high_edge_count = 0
        for signal in signals:
            if abs(signal.edge) >= high_edge_threshold and signal.confidence > 0:
                await self.notify_signal(signal)
                high_edge_count += 1

        await self.notify_summary(signals)

        if self._client is not None:
            await self._client.close()
            self._client = None

        logger.info(
            "Telegram: sent %d high-edge alert(s) + summary (%d signals)",
            high_edge_count,
            len(signals),
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
