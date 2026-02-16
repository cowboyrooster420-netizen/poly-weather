"""Signal output formatters: Rich table, JSON, CSV."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime

from rich.console import Console
from rich.table import Table

from weather_edge.signals.models import Signal


def format_table(signals: list[Signal], console: Console | None = None) -> None:
    """Print signals as a Rich table sorted by absolute edge (descending)."""
    if console is None:
        console = Console()

    if not signals:
        console.print("[yellow]No signals generated (no markets with sufficient edge).[/yellow]")
        return

    sorted_signals = sorted(signals, key=lambda s: abs(s.edge), reverse=True)

    table = Table(
        title="Weather Edge Signals",
        caption=f"Generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        show_lines=True,
    )

    table.add_column("Direction", style="bold", width=5)
    table.add_column("Edge", justify="right", width=7)
    table.add_column("Kelly %", justify="right", width=7)
    table.add_column("Model P", justify="right", width=7)
    table.add_column("Market P", justify="right", width=7)
    table.add_column("Conf", justify="right", width=5)
    table.add_column("Type", width=8)
    table.add_column("Location", width=16)
    table.add_column("Lead (h)", justify="right", width=8)
    table.add_column("Question", width=40, no_wrap=False)

    for s in sorted_signals:
        edge_color = "green" if s.edge > 0 else "red"
        dir_color = "green" if s.direction == "YES" else "red"

        table.add_row(
            f"[{dir_color}]{s.direction}[/{dir_color}]",
            f"[{edge_color}]{s.edge:+.1%}[/{edge_color}]",
            f"{s.kelly_fraction:.1%}",
            f"{s.model_prob:.1%}",
            f"{s.market_prob:.1%}",
            f"{s.confidence:.0%}",
            s.market_type,
            (s.location or "")[:16],
            f"{s.lead_time_hours:.0f}",
            s.question[:80],
        )

    console.print(table)
    console.print(f"\n[dim]{len(signals)} signal(s) total[/dim]")


def format_json(signals: list[Signal]) -> str:
    """Format signals as a JSON string."""
    sorted_signals = sorted(signals, key=lambda s: abs(s.edge), reverse=True)
    return json.dumps(
        [
            {
                "market_id": s.market_id,
                "question": s.question,
                "direction": s.direction,
                "edge": s.edge,
                "kelly_fraction": s.kelly_fraction,
                "model_prob": s.model_prob,
                "market_prob": s.market_prob,
                "confidence": s.confidence,
                "market_type": s.market_type,
                "location": s.location,
                "lead_time_hours": s.lead_time_hours,
                "sources": s.sources,
                "details": s.details,
                "timestamp": s.timestamp.isoformat(),
            }
            for s in sorted_signals
        ],
        indent=2,
    )


def format_telegram_signal(signal: Signal) -> str:
    """Format a single signal for Telegram (Markdown)."""
    arrow = "\U0001f4c8" if signal.direction == "YES" else "\U0001f4c9"
    lines = [
        "\U0001f514 *HIGH EDGE SIGNAL*",
        "",
        f"{arrow} *{signal.direction}* \u2014 Edge: {signal.edge:+.1%}",
        f"\U0001f4b0 Kelly: {signal.kelly_fraction:.1%} of bankroll",
        "",
        f"\U0001f4cb {signal.question[:120]}",
        f"\U0001f4cd {signal.location} | \U0001f321 {signal.market_type}",
        f"\u23f0 Lead time: {signal.lead_time_hours:.1f}h | Confidence: {signal.confidence:.2f}",
        "",
        f"Model: {signal.model_prob:.1%} | Market: {signal.market_prob:.1%}",
        f"Sources: {', '.join(signal.sources)}",
    ]
    return "\n".join(lines)


def format_telegram_summary(signals: list[Signal]) -> str:
    """Format a run summary for Telegram (Markdown)."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    count = len(signals)

    lines = [
        "\U0001f4ca *Weather Edge Scan Complete*",
        "",
        f"\U0001f550 {now}",
        f"\U0001f4c8 {count} signal(s) generated",
    ]

    if signals:
        sorted_signals = sorted(signals, key=lambda s: abs(s.edge), reverse=True)
        lines.append("")
        lines.append("| Dir | Edge | Kelly | Market |")
        for s in sorted_signals:
            question_short = s.question[:30].split("?")[0].strip()
            lines.append(
                f"| {s.direction} | {s.edge:+.1%} | {s.kelly_fraction:.1%} | {question_short} |"
            )

    return "\n".join(lines)


def format_csv(signals: list[Signal]) -> str:
    """Format signals as CSV."""
    sorted_signals = sorted(signals, key=lambda s: abs(s.edge), reverse=True)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "market_id", "question", "direction", "edge", "kelly_fraction",
        "model_prob", "market_prob", "confidence", "market_type", "location",
        "lead_time_hours", "sources", "timestamp",
    ])
    for s in sorted_signals:
        writer.writerow([
            s.market_id, s.question, s.direction, s.edge, s.kelly_fraction,
            s.model_prob, s.market_prob, s.confidence, s.market_type, s.location,
            s.lead_time_hours, "|".join(s.sources), s.timestamp.isoformat(),
        ])
    return output.getvalue()
