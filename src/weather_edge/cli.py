"""Typer CLI: weather-edge scan, inspect, list-markets."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="weather-edge",
    help="Polymarket weather prediction market signal generator",
    no_args_is_help=True,
)
console = Console()


@app.command()
def scan(
    output: str = typer.Option(
        "table", "--output", "-o",
        help="Output format: table, json, csv",
    ),
    min_edge: Optional[float] = typer.Option(
        None, "--min-edge",
        help="Override minimum edge threshold (e.g. 0.05 for 5%)",
    ),
    notify: bool = typer.Option(
        False, "--notify", "-n",
        help="Send Telegram notifications",
    ),
) -> None:
    """Scan Polymarket for weather markets and generate trading signals."""
    from weather_edge.signals.formatters import format_csv, format_json, format_table

    # Override settings if CLI flags provided
    if min_edge is not None or notify:
        import weather_edge.config as cfg
        original = cfg.get_settings
        def patched() -> cfg.Settings:
            s = original()
            if min_edge is not None:
                s.min_edge = min_edge  # type: ignore[misc]
            if notify:
                s.telegram_enabled = True  # type: ignore[misc]
            return s
        cfg.get_settings = patched

    async def _run() -> None:
        from weather_edge.pipeline import run_pipeline
        signals = await run_pipeline()

        if output == "json":
            console.print(format_json(signals))
        elif output == "csv":
            console.print(format_csv(signals))
        else:
            format_table(signals, console)

    asyncio.run(_run())


@app.command(name="list-markets")
def list_markets() -> None:
    """List detected weather markets from Polymarket (no forecasting)."""

    async def _run() -> None:
        from weather_edge.pipeline import scan_markets

        markets = await scan_markets()

        if not markets:
            console.print("[yellow]No weather markets found.[/yellow]")
            return

        table = Table(title="Detected Weather Markets", show_lines=True)
        table.add_column("ID", width=10)
        table.add_column("Type", width=10)
        table.add_column("Location", width=20)
        table.add_column("Threshold", width=12)
        table.add_column("YES Price", justify="right", width=8)
        table.add_column("Volume", justify="right", width=10)
        table.add_column("Question", width=50, no_wrap=False)

        for m in markets:
            market_type = m.params.market_type.value if m.params else "?"
            location = m.params.location if m.params else "?"
            threshold = ""
            if m.params and m.params.threshold is not None:
                threshold = f"{m.params.comparison.value} {m.params.threshold}{m.params.unit}"

            table.add_row(
                m.market_id[:10],
                market_type,
                location[:20] if location else "?",
                threshold,
                f"{m.outcome_yes_price:.2f}",
                f"${m.volume:,.0f}",
                m.question[:80],
            )

        console.print(table)

    asyncio.run(_run())


@app.command()
def inspect(
    market_id: str = typer.Argument(help="Market ID or slug to inspect"),
) -> None:
    """Inspect a specific weather market with detailed forecast breakdown."""

    async def _run() -> None:
        from weather_edge.forecasting.registry import get_model
        from weather_edge.markets.client import fetch_all_active_markets, raw_to_weather_market
        from weather_edge.markets.classifier import classify_market
        from weather_edge.markets.parser import parse_market
        from weather_edge.signals.analyzer import generate_signal
        from weather_edge.weather.noaa import fetch_noaa_forecast
        from weather_edge.weather.openmeteo import fetch_both_ensembles

        console.print(f"[bold]Inspecting market: {market_id}[/bold]")

        # Find the market
        raw_markets = await fetch_all_active_markets()
        raw = None
        for m in raw_markets:
            if str(m.get("id", "")).startswith(market_id) or m.get("slug", "") == market_id:
                raw = m
                break

        if raw is None:
            console.print(f"[red]Market '{market_id}' not found[/red]")
            return

        market = raw_to_weather_market(raw)
        question = raw.get("question", "")
        description = raw.get("description", "")

        # Classify
        is_weather, conf = await classify_market(question, description)
        console.print(f"  Weather classification: {is_weather} (confidence: {conf:.0%})")

        if not is_weather:
            console.print("[yellow]This market was not classified as weather-related.[/yellow]")
            return

        # Parse
        params = await parse_market(question, description)
        market.params = params

        if params:
            console.print(f"  Type: {params.market_type.value}")
            console.print(f"  Location: {params.location}")
            console.print(f"  Lat/Lon: {params.lat_lon}")
            console.print(f"  Threshold: {params.comparison.value} {params.threshold} {params.unit}")
            console.print(f"  Target date: {params.target_date_str} ({params.target_date})")
        else:
            console.print("[yellow]Could not parse market parameters.[/yellow]")
            return

        # Fetch weather data
        if params.lat_lon:
            lat, lon = params.lat_lon
            console.print(f"\n[bold]Fetching weather data for ({lat:.2f}, {lon:.2f})...[/bold]")

            gfs, ecmwf = await fetch_both_ensembles(lat, lon)
            console.print(f"  GFS: {gfs.n_members} members, {gfs.n_times} time steps")
            console.print(f"  ECMWF: {ecmwf.n_members} members, {ecmwf.n_times} time steps")

            noaa = await fetch_noaa_forecast(lat, lon)
            if noaa:
                console.print(f"  NOAA: {len(noaa.periods)} periods, {len(noaa.alerts)} alerts")
            else:
                console.print("  NOAA: not available (non-US or API error)")

            # Run forecast
            model = get_model(params.market_type)
            if model:
                estimate = await model.estimate(params, gfs, ecmwf, noaa)
                console.print(f"\n[bold]Forecast Result:[/bold]")
                console.print(f"  Model probability: [bold]{estimate.probability:.1%}[/bold]")
                console.print(f"  Raw probability:   {estimate.raw_probability:.1%}")
                console.print(f"  Market probability: {market.market_prob:.1%}")
                console.print(
                    f"  Edge: [{'green' if estimate.probability > market.market_prob else 'red'}]"
                    f"{estimate.probability - market.market_prob:+.1%}[/]"
                )
                console.print(f"  Confidence: {estimate.confidence:.0%}")
                console.print(f"  Lead time: {estimate.lead_time_hours:.0f}h")
                console.print(f"  Sources: {', '.join(estimate.sources_used)}")
                console.print(f"  Details: {estimate.details}")

                signal = generate_signal(market, estimate)
                if signal:
                    console.print(
                        f"\n  [bold green]SIGNAL: {signal.direction} "
                        f"(Kelly: {signal.kelly_fraction:.1%})[/bold green]"
                    )
                else:
                    console.print("\n  [dim]No signal (edge below threshold)[/dim]")

    asyncio.run(_run())


@app.command()
def stats() -> None:
    """Show historical signal performance statistics."""

    async def _run() -> None:
        from weather_edge.signals.tracker import SignalTracker

        tracker = SignalTracker()
        summary = await tracker.get_performance_summary()

        console.print("[bold]Signal Performance Summary[/bold]")
        console.print(f"  Total signals logged: {summary['total_signals']}")
        console.print(f"  Resolved outcomes:    {summary['resolved']}")
        if summary['win_rate'] is not None:
            console.print(f"  Win rate:             {summary['win_rate']:.1%}")
        else:
            console.print("  Win rate:             N/A (no resolved outcomes)")
        if summary['avg_abs_edge'] is not None:
            console.print(f"  Avg |edge|:           {summary['avg_abs_edge']:.1%}")

    asyncio.run(_run())


if __name__ == "__main__":
    app()
