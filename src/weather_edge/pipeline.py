"""Top-level pipeline orchestrator.

Wires together: market scanning → weather fetching → forecasting → signal generation.
Uses asyncio.gather for concurrent API calls, grouped by location.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

import httpx
from rich.console import Console

from weather_edge.common.types import LatLon
from weather_edge.forecasting.base import ProbabilityEstimate
from weather_edge.forecasting.registry import get_model
from weather_edge.markets.classifier import classify_market
from weather_edge.markets.client import fetch_all_active_markets, raw_to_weather_market
from weather_edge.markets.models import WeatherMarket
from weather_edge.markets.parser import parse_market
from weather_edge.config import get_settings
from weather_edge.notifications.telegram import TelegramNotifier
from weather_edge.signals.analyzer import generate_signal
from weather_edge.signals.models import Signal
from weather_edge.signals.tracker import SignalTracker
from weather_edge.weather.models import EnsembleForecast, NOAAForecast
from weather_edge.weather.noaa import fetch_noaa_forecast
from weather_edge.weather.openmeteo import fetch_both_ensembles

logger = logging.getLogger(__name__)
console = Console()


async def scan_markets() -> list[WeatherMarket]:
    """Scan Polymarket for weather prediction markets.

    1. Fetch all active markets from Gamma API
    2. Classify each as weather/non-weather (two-stage)
    3. Parse weather markets into structured params
    """
    console.print("[bold]Scanning Polymarket for weather markets...[/bold]")
    raw_markets = await fetch_all_active_markets()
    console.print(f"  Found {len(raw_markets)} active markets total")

    weather_markets: list[WeatherMarket] = []
    llm_calls = 0

    filtered_count = 0
    parse_fail_count = 0

    for raw in raw_markets:
        question = raw.get("question", "")
        description = raw.get("description", "")

        is_weather, confidence = await classify_market(question, description)
        if not is_weather:
            filtered_count += 1
            continue

        market = raw_to_weather_market(raw)

        # Parse market parameters
        params = await parse_market(question, description)
        if params is not None:
            market.params = params
        else:
            parse_fail_count += 1
            logger.info("Failed to parse params for market %s: %s", market.market_id, question[:80])

        weather_markets.append(market)

    logger.debug("Filtered %d non-weather markets, %d parse failures", filtered_count, parse_fail_count)
    console.print(
        f"  Identified [green]{len(weather_markets)}[/green] weather market(s)"
    )
    return weather_markets


async def _fetch_weather_for_location(
    lat: float, lon: float,
) -> tuple[EnsembleForecast | None, EnsembleForecast | None, NOAAForecast | None]:
    """Fetch all weather data for a single location."""
    gfs: EnsembleForecast | None = None
    ecmwf: EnsembleForecast | None = None
    noaa: NOAAForecast | None = None

    try:
        gfs, ecmwf = await fetch_both_ensembles(lat, lon)
    except httpx.HTTPStatusError as exc:
        logger.warning("Open-Meteo HTTP %d for (%.2f, %.2f)", exc.response.status_code, lat, lon)
        console.print(f"  [yellow]Open-Meteo HTTP {exc.response.status_code} for ({lat:.2f}, {lon:.2f})[/yellow]")
    except httpx.TimeoutException:
        logger.warning("Open-Meteo timeout for (%.2f, %.2f)", lat, lon)
        console.print(f"  [yellow]Open-Meteo timeout for ({lat:.2f}, {lon:.2f})[/yellow]")
    except (ValueError, KeyError) as exc:
        logger.warning("Open-Meteo parse error for (%.2f, %.2f): %s", lat, lon, exc)
        console.print(f"  [yellow]Open-Meteo parse error for ({lat:.2f}, {lon:.2f}): {exc}[/yellow]")

    try:
        noaa = await fetch_noaa_forecast(lat, lon)
    except httpx.HTTPStatusError:
        logger.debug("NWS unavailable for (%.2f, %.2f) — likely non-US", lat, lon)
    except httpx.TimeoutException:
        logger.info("NWS timeout for (%.2f, %.2f)", lat, lon)

    return gfs, ecmwf, noaa


async def fetch_weather_data(
    markets: list[WeatherMarket],
) -> dict[LatLon, tuple[EnsembleForecast | None, EnsembleForecast | None, NOAAForecast | None]]:
    """Fetch weather data grouped by location (deduped).

    Markets at the same location share one set of API calls.
    """
    # Group markets by location (rounded to 2 decimal places)
    locations: dict[LatLon, list[WeatherMarket]] = defaultdict(list)
    skipped_no_location = 0
    for market in markets:
        if market.params and market.params.lat_lon:
            key = (
                round(market.params.lat_lon[0], 2),
                round(market.params.lat_lon[1], 2),
            )
            locations[key].append(market)
        else:
            skipped_no_location += 1

    if skipped_no_location:
        logger.info(
            "Skipped %d market(s) without geocoded location for weather fetch",
            skipped_no_location,
        )

    console.print(
        f"[bold]Fetching weather data for {len(locations)} unique location(s)...[/bold]"
    )

    results: dict[LatLon, tuple] = {}

    # Rate-limit concurrent fetches to avoid Open-Meteo 429s.
    # Each location = 2 API calls (GFS + ECMWF), so 3 concurrent locations = 6 requests.
    sem = asyncio.Semaphore(3)

    async def _throttled_fetch(lat: float, lon: float) -> tuple:
        async with sem:
            result = await _fetch_weather_for_location(lat, lon)
            # Small delay between batches to stay under rate limits
            await asyncio.sleep(1.0)
            return result

    latlon_list = list(locations.keys())
    coros = [_throttled_fetch(ll[0], ll[1]) for ll in latlon_list]
    fetched = await asyncio.gather(*coros, return_exceptions=True)

    for latlon, result in zip(latlon_list, fetched):
        if isinstance(result, BaseException):
            console.print(f"  [red]Error fetching ({latlon[0]:.2f}, {latlon[1]:.2f}): {result}[/red]")
            results[latlon] = (None, None, None)
        else:
            results[latlon] = result

    return results


async def run_forecasts(
    markets: list[WeatherMarket],
    weather_data: dict[LatLon, tuple],
) -> list[tuple[WeatherMarket, ProbabilityEstimate]]:
    """Run forecast models for each market.

    Dispatches to the correct model by MarketType.
    """
    console.print("[bold]Running forecast models...[/bold]")
    results: list[tuple[WeatherMarket, ProbabilityEstimate]] = []

    skipped_no_params = 0
    skipped_no_model = 0

    for market in markets:
        if not market.params:
            skipped_no_params += 1
            continue

        model = get_model(market.params.market_type)
        if model is None:
            skipped_no_model += 1
            logger.info("No model for market type %s (market %s)", market.params.market_type.value, market.market_id)
            continue

        # Get weather data for this market's location
        gfs, ecmwf, noaa = None, None, None
        if market.params.lat_lon:
            key = (
                round(market.params.lat_lon[0], 2),
                round(market.params.lat_lon[1], 2),
            )
            weather = weather_data.get(key)
            if weather:
                gfs, ecmwf, noaa = weather

        try:
            estimate = await model.estimate(market.params, gfs, ecmwf, noaa)
            results.append((market, estimate))
            console.print(
                f"  {market.params.market_type.value}: "
                f"model={estimate.probability:.1%} "
                f"market={market.market_prob:.1%} "
                f"edge={estimate.probability - market.market_prob:+.1%}"
            )
        except (ValueError, TypeError, IndexError) as exc:
            logger.warning("Model error for market %s: %s", market.market_id, exc)
            console.print(f"  [red]Model error for {market.market_id}: {exc}[/red]")

    if skipped_no_params or skipped_no_model:
        logger.info("Forecast skips: %d no params, %d no model", skipped_no_params, skipped_no_model)

    return results


async def generate_signals(
    forecast_results: list[tuple[WeatherMarket, ProbabilityEstimate]],
) -> list[Signal]:
    """Generate trading signals from forecast results.

    Filters by minimum edge threshold and computes Kelly sizing.
    """
    signals: list[Signal] = []

    for market, estimate in forecast_results:
        signal = generate_signal(market, estimate)
        if signal is not None:
            signals.append(signal)

    console.print(
        f"[bold]Generated [green]{len(signals)}[/green] signal(s) "
        f"(from {len(forecast_results)} forecast(s))[/bold]"
    )
    return signals


async def run_pipeline() -> list[Signal]:
    """Run the full pipeline: scan → fetch → forecast → signal → log.

    Returns list of generated signals.
    """
    # Step 1: Scan markets
    markets = await scan_markets()
    if not markets:
        console.print("[yellow]No weather markets found.[/yellow]")
        return []

    # Step 2: Fetch weather data (grouped by location, concurrent)
    weather_data = await fetch_weather_data(markets)

    # Step 3: Run forecast models
    forecast_results = await run_forecasts(markets, weather_data)
    if not forecast_results:
        console.print("[yellow]No forecasts produced.[/yellow]")
        return []

    # Step 4: Generate signals
    signals = await generate_signals(forecast_results)

    # Step 5: Log signals to SQLite
    if signals:
        tracker = SignalTracker()
        ids = await tracker.log_signals(signals)
        console.print(f"[dim]Logged {len(ids)} signal(s) to database[/dim]")

    # Step 6: Send Telegram notifications
    settings = get_settings()
    if settings.telegram_enabled:
        notifier = TelegramNotifier()
        await notifier.notify(signals)

    return signals
