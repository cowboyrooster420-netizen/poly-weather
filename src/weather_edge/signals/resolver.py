"""Auto-resolve market outcomes from the Polymarket Gamma API."""

from __future__ import annotations

import json
import logging

from weather_edge.config import get_settings
from weather_edge.markets.client import fetch_market_by_id
from weather_edge.notifications.telegram import TelegramNotifier
from weather_edge.signals.tracker import SignalTracker

logger = logging.getLogger(__name__)

_RESOLUTION_THRESHOLD = 0.99


def _parse_outcome(raw: dict) -> int | None:
    """Parse outcome from a closed market's outcomePrices.

    Returns 1 if YES won, 0 if NO won, or None if ambiguous.
    """
    outcome_prices = raw.get("outcomePrices", "")
    if not outcome_prices:
        return None

    try:
        if isinstance(outcome_prices, str):
            prices = json.loads(outcome_prices)
        else:
            prices = outcome_prices
        if len(prices) < 2:
            return None
        yes_price = float(prices[0])
        no_price = float(prices[1])
    except (json.JSONDecodeError, ValueError, IndexError, TypeError):
        return None

    if yes_price >= _RESOLUTION_THRESHOLD:
        return 1
    if no_price >= _RESOLUTION_THRESHOLD:
        return 0
    return None


async def resolve_pending_signals() -> list[dict]:
    """Check unresolved signals against the Gamma API and backfill outcomes.

    Returns a list of dicts for each newly resolved market:
        {market_id, question, outcome, direction, correct}
    """
    tracker = SignalTracker()
    pending = await tracker.get_unresolved_market_ids()

    if not pending:
        logger.debug("No unresolved signals to check")
        return []

    resolved: list[dict] = []

    for market_id, question in pending:
        raw = await fetch_market_by_id(market_id)
        if raw is None:
            continue

        if not raw.get("closed", False):
            continue

        outcome = _parse_outcome(raw)
        if outcome is None:
            logger.warning(
                "Ambiguous resolution for market %s (%s) — skipping",
                market_id, question,
            )
            continue

        closed_time = raw.get("closedTime") or raw.get("endDate")
        await tracker.backfill_outcome(market_id, outcome, closed_time)

        # Determine if our direction was correct
        direction = await tracker.get_signal_direction(market_id)

        correct = None
        if direction is not None:
            correct = (direction == "YES" and outcome == 1) or (
                direction == "NO" and outcome == 0
            )

        resolved.append({
            "market_id": market_id,
            "question": question,
            "outcome": outcome,
            "direction": direction,
            "correct": correct,
        })

    # Send Telegram scorecard if any markets were resolved
    if resolved:
        settings = get_settings()
        if settings.telegram_enabled:
            await _send_scorecard(resolved, tracker)

    return resolved


async def _send_scorecard(
    resolved: list[dict], tracker: SignalTracker,
) -> None:
    """Send a Telegram scorecard summarizing resolved markets."""
    lines = [f"Resolved {len(resolved)} market(s):"]
    for r in resolved:
        outcome_str = "YES" if r["outcome"] == 1 else "NO"
        direction = r["direction"] or "?"
        mark = "\u2713" if r["correct"] else "\u2717"
        q = r["question"] or r["market_id"]
        # Truncate long questions
        if len(q) > 50:
            q = q[:47] + "..."
        lines.append(
            f"  {q} \u2192 {outcome_str} won \u2192 We said {direction} {mark}"
        )

    summary = await tracker.get_performance_summary()
    win_rate = summary.get("win_rate")
    brier = summary.get("brier_score")

    stats_parts = []
    if win_rate is not None:
        wins = summary.get("wins", 0)
        total = summary.get("resolved", 0)
        stats_parts.append(f"Record: {wins}/{total} ({win_rate:.0%})")
    if brier is not None:
        stats_parts.append(f"Brier: {brier:.3f}")
    if stats_parts:
        lines.append("")
        lines.append(" | ".join(stats_parts))

    notifier = TelegramNotifier()
    await notifier.send_message("\n".join(lines))
    await notifier.close()
