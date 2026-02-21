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
def resolve() -> None:
    """Resolve pending market outcomes from Polymarket."""

    async def _run() -> None:
        from weather_edge.signals.resolver import resolve_pending_signals
        from weather_edge.signals.tracker import SignalTracker

        resolved = await resolve_pending_signals()

        if not resolved:
            console.print("[dim]No markets newly resolved.[/dim]")
        else:
            table = Table(title="Resolved Markets", show_lines=True)
            table.add_column("Question", width=50, no_wrap=False)
            table.add_column("Outcome", width=8)
            table.add_column("Our Call", width=8)
            table.add_column("Correct", width=8)

            for r in resolved:
                outcome_str = "YES" if r["outcome"] == 1 else "NO"
                direction = r["direction"] or "?"
                mark = "\u2713" if r["correct"] else "\u2717"
                table.add_row(
                    r["question"] or r["market_id"],
                    outcome_str,
                    direction,
                    mark,
                )
            console.print(table)

        # Show updated stats
        tracker = SignalTracker()
        summary = await tracker.get_performance_summary()
        console.print("\n[bold]Updated Stats[/bold]")
        if summary["win_rate"] is not None:
            console.print(f"  Win rate:    {summary['win_rate']:.1%}")
        else:
            console.print("  Win rate:    N/A")
        if summary.get("brier_score") is not None:
            console.print(f"  Brier score: {summary['brier_score']:.3f}")
        else:
            console.print("  Brier score: N/A")

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
        if summary.get('brier_score') is not None:
            console.print(f"  Brier score:          {summary['brier_score']:.3f}")
        else:
            console.print("  Brier score:          N/A (no resolved outcomes)")

    asyncio.run(_run())


@app.command()
def scorecard(
    market_type: Optional[str] = typer.Option(
        None, "--type", "-t",
        help="Filter by market type (e.g. temperature, precipitation)",
    ),
) -> None:
    """Show detailed scorecard of all resolved markets with Brier scores."""

    async def _run() -> None:
        from collections import defaultdict

        from weather_edge.signals.tracker import SignalTracker

        tracker = SignalTracker()
        rows = await tracker.get_resolved_signals(market_type=market_type)

        if not rows:
            if market_type:
                console.print(f"[yellow]No resolved {market_type} signals found.[/yellow]")
            else:
                console.print("[yellow]No resolved signals found.[/yellow]")
            return

        # Per-signal detail table
        table = Table(
            title="Resolved Signals Scorecard",
            show_lines=True,
        )
        table.add_column("Question", width=40, no_wrap=False)
        table.add_column("Location", width=14)
        table.add_column("Type", width=8)
        table.add_column("Our Call", justify="center", width=8)
        table.add_column("Model P", justify="right", width=8)
        table.add_column("Market P", justify="right", width=8)
        table.add_column("Edge", justify="right", width=8)
        table.add_column("Outcome", justify="center", width=8)
        table.add_column("Result", justify="center", width=6)

        # Track per-type stats
        type_stats: dict[str, list[tuple[float, int]]] = defaultdict(list)
        total_wins = 0
        total_resolved = 0

        for row in rows:
            outcome_str = "YES" if row["outcome"] == 1 else "NO"
            direction = row["direction"] or "?"
            correct = (
                (direction == "YES" and row["outcome"] == 1)
                or (direction == "NO" and row["outcome"] == 0)
            )
            mark = "[green]W[/green]" if correct else "[red]L[/red]"

            mtype = row["market_type"] or "unknown"
            type_stats[mtype].append((row["model_prob"], row["outcome"]))
            total_resolved += 1
            if correct:
                total_wins += 1

            edge_color = "green" if row["edge"] > 0 else "red"
            table.add_row(
                (row["question"] or "")[:60],
                (row["location"] or "")[:14],
                mtype,
                f"[bold]{direction}[/bold]",
                f"{row['model_prob']:.1%}",
                f"{row['market_prob']:.1%}",
                f"[{edge_color}]{row['edge']:+.1%}[/{edge_color}]",
                outcome_str,
                mark,
            )

        console.print(table)

        # Brier score summary by type
        def _brier(pairs: list[tuple[float, int]]) -> float:
            return sum((p - o) ** 2 for p, o in pairs) / len(pairs)

        def _win_rate(pairs: list[tuple[float, int]], rows_for_type: list[dict]) -> float:
            # Need direction info, which isn't in the pairs. Recompute.
            wins = 0
            for p, o in pairs:
                # Approximate: model_prob > 0.5 means we'd call YES
                predicted_yes = p > 0.5
                if (predicted_yes and o == 1) or (not predicted_yes and o == 0):
                    wins += 1
            return wins / len(pairs)

        summary_table = Table(title="Brier Score by Market Type", show_lines=True)
        summary_table.add_column("Market Type", width=16)
        summary_table.add_column("Resolved", justify="right", width=10)
        summary_table.add_column("Win Rate", justify="right", width=10)
        summary_table.add_column("Brier Score", justify="right", width=12)

        # Sort types alphabetically, compute stats
        all_pairs: list[tuple[float, int]] = []
        for mtype in sorted(type_stats.keys()):
            pairs = type_stats[mtype]
            all_pairs.extend(pairs)
            brier = _brier(pairs)
            wins = sum(
                1 for row in rows
                if (row["market_type"] or "unknown") == mtype
                and (
                    (row["direction"] == "YES" and row["outcome"] == 1)
                    or (row["direction"] == "NO" and row["outcome"] == 0)
                )
            )
            wr = wins / len(pairs)
            summary_table.add_row(
                mtype,
                str(len(pairs)),
                f"{wr:.1%}",
                f"{brier:.3f}",
            )

        # Total row
        if all_pairs:
            total_brier = _brier(all_pairs)
            total_wr = total_wins / total_resolved if total_resolved else 0
            summary_table.add_row(
                "[bold]ALL[/bold]",
                f"[bold]{total_resolved}[/bold]",
                f"[bold]{total_wr:.1%}[/bold]",
                f"[bold]{total_brier:.3f}[/bold]",
            )

        console.print()
        console.print(summary_table)

    asyncio.run(_run())


@app.command()
def calibrate(
    station: Optional[str] = typer.Option(
        None, "--station", "-s",
        help="Calibrate a single station (WU station ID)",
    ),
    days: int = typer.Option(
        90, "--days", "-d",
        help="Training window in days",
    ),
) -> None:
    """Compute per-station bias correction from WU history vs ERA5 reanalysis.

    For each station, scrapes WU daily high/low, fetches matching ERA5
    reanalysis, and computes the systematic bias. Results are saved to
    ~/.weather-edge/station_biases.json.

    Can be cron'd (e.g. weekly) to keep biases fresh.
    """

    async def _run() -> None:
        from weather_edge.calibration.openmeteo_history import (
            fetch_openmeteo_history_v2,
            training_window,
        )
        from weather_edge.calibration.station_bias import (
            StationBiasV2,
            compute_station_bias_stratified,
            save_biases,
        )
        from weather_edge.weather.stations import STATIONS
        from weather_edge.weather.wunderground import fetch_wu_history

        start, end = training_window(days)
        console.print(
            f"[bold]Calibrating station biases[/bold]  "
            f"window: {start} to {end} ({days} days)"
        )

        # Determine which stations to calibrate
        if station:
            if station not in STATIONS:
                console.print(f"[red]Unknown station: {station}[/red]")
                console.print(f"Known stations: {', '.join(STATIONS.keys())}")
                return
            targets = {station: STATIONS[station]}
        else:
            targets = STATIONS

        results: dict[str, StationBiasV2] = {}

        for wu_id, stn in targets.items():
            console.print(f"\n  [bold]{wu_id}[/bold] ({stn.city})")

            # Fetch WU history
            console.print(f"    Scraping WU history...", end="")
            wu_obs = await fetch_wu_history(wu_id, start, end)
            console.print(f" {len(wu_obs)} days")

            if not wu_obs:
                console.print("    [yellow]No WU data, skipping[/yellow]")
                results[wu_id] = StationBiasV2(
                    station_id=wu_id, city=stn.city,
                    high_bias_c=0.0, low_bias_c=0.0, mean_bias_c=0.0,
                    n_days=0,
                )
                continue

            # Fetch ERA5 reanalysis (v2 with cloud cover)
            console.print(f"    Fetching ERA5 reanalysis...", end="")
            lat, lon = stn.lat_lon
            om_obs = await fetch_openmeteo_history_v2(
                lat, lon, start, end, timezone=stn.timezone,
            )
            console.print(f" {len(om_obs)} days")

            if not om_obs:
                console.print("    [yellow]No ERA5 data, skipping[/yellow]")
                results[wu_id] = StationBiasV2(
                    station_id=wu_id, city=stn.city,
                    high_bias_c=0.0, low_bias_c=0.0, mean_bias_c=0.0,
                    n_days=0,
                )
                continue

            # Match days present in both datasets
            om_by_date = {o.obs_date: o for o in om_obs}
            wu_highs: list[float] = []
            wu_lows: list[float] = []
            om_maxs: list[float] = []
            om_mins: list[float] = []
            cloud_covers: list[float | None] = []

            for wu in wu_obs:
                om = om_by_date.get(wu.date)
                if om is not None:
                    wu_highs.append(wu.high_temp_c)
                    wu_lows.append(wu.low_temp_c)
                    om_maxs.append(om.max_temp_c)
                    om_mins.append(om.min_temp_c)
                    cloud_covers.append(om.cloud_cover_mean)

            if not wu_highs:
                console.print("    [yellow]No matching days, skipping[/yellow]")
                results[wu_id] = StationBiasV2(
                    station_id=wu_id, city=stn.city,
                    high_bias_c=0.0, low_bias_c=0.0, mean_bias_c=0.0,
                    n_days=0,
                )
                continue

            bias = compute_station_bias_stratified(
                wu_highs, wu_lows, om_maxs, om_mins, cloud_covers,
                station_id=wu_id, city=stn.city,
            )
            results[wu_id] = bias
            console.print(
                f"    Bias: high={bias.high_bias_c:+.2f}C, "
                f"low={bias.low_bias_c:+.2f}C, "
                f"mean={bias.mean_bias_c:+.2f}C "
                f"(n={bias.n_days})"
            )
            # Show per-condition breakdown
            for cb in bias.condition_biases:
                if cb.n_days > 0:
                    console.print(
                        f"      {cb.condition.value:>8}: "
                        f"high={cb.high_bias_c:+.2f}C, "
                        f"low={cb.low_bias_c:+.2f}C "
                        f"(n={cb.n_days})"
                    )

        # Save results
        path = save_biases(results, training_days=days)
        console.print(f"\n[bold green]Saved biases to {path}[/bold green]")

        # Display summary table
        table = Table(title="Station Bias Summary", show_lines=True)
        table.add_column("Station", width=16)
        table.add_column("City", width=14)
        table.add_column("High Bias", justify="right", width=10)
        table.add_column("Low Bias", justify="right", width=10)
        table.add_column("Mean Bias", justify="right", width=10)
        table.add_column("Std (H/L)", justify="right", width=12)
        table.add_column("Days", justify="right", width=6)
        table.add_column("Clear", justify="right", width=12)
        table.add_column("Partly", justify="right", width=12)
        table.add_column("Overcast", justify="right", width=12)

        for wu_id, b in results.items():
            # Format per-condition high bias summaries
            cond_strs: dict[str, str] = {}
            for cb in b.condition_biases:
                if cb.n_days > 0:
                    cond_strs[cb.condition.value] = f"{cb.high_bias_c:+.1f} ({cb.n_days}d)"
                else:
                    cond_strs[cb.condition.value] = "-"

            table.add_row(
                wu_id,
                b.city,
                f"{b.high_bias_c:+.2f}C",
                f"{b.low_bias_c:+.2f}C",
                f"{b.mean_bias_c:+.2f}C",
                f"{b.high_std_c:.2f}/{b.low_std_c:.2f}",
                str(b.n_days),
                cond_strs.get("clear", "-"),
                cond_strs.get("partly", "-"),
                cond_strs.get("overcast", "-"),
            )

        console.print(table)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
