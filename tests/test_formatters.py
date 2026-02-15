"""Tests for signal output formatters: Rich table, JSON, CSV."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

import pytest

from weather_edge.signals.formatters import format_csv, format_json, format_table
from weather_edge.signals.models import Signal


@pytest.fixture
def multiple_signals(now):
    """Multiple signals for testing sort order and formatting."""
    return [
        Signal(
            market_id="m1",
            question="Will Phoenix exceed 120F?",
            market_type="temperature",
            location="Phoenix, AZ",
            model_prob=0.55,
            market_prob=0.35,
            edge=0.20,
            kelly_fraction=0.08,
            confidence=0.85,
            direction="YES",
            lead_time_hours=24.0,
            sources=["ECMWF", "GFS"],
            details="Test 1",
            timestamp=now,
        ),
        Signal(
            market_id="m2",
            question="Will it rain in Houston?",
            market_type="precipitation",
            location="Houston, TX",
            model_prob=0.20,
            market_prob=0.45,
            edge=-0.25,
            kelly_fraction=0.10,
            confidence=0.70,
            direction="NO",
            lead_time_hours=48.0,
            sources=["ECMWF"],
            details="Test 2",
            timestamp=now,
        ),
        Signal(
            market_id="m3",
            question="Will NYC freeze?",
            market_type="temperature",
            location="New York, NY",
            model_prob=0.40,
            market_prob=0.30,
            edge=0.10,
            kelly_fraction=0.03,
            confidence=0.90,
            direction="YES",
            lead_time_hours=12.0,
            sources=["ECMWF", "GFS", "NOAA/NWS"],
            details="Test 3",
            timestamp=now,
        ),
    ]


class TestFormatJson:
    def test_json_valid(self, sample_signal):
        result = format_json([sample_signal])
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_json_fields(self, sample_signal):
        result = format_json([sample_signal])
        data = json.loads(result)
        entry = data[0]
        assert entry["market_id"] == "test-market-001"
        assert entry["direction"] == "YES"
        assert entry["edge"] == 0.20
        assert entry["kelly_fraction"] == 0.08
        assert entry["model_prob"] == 0.55
        assert entry["market_prob"] == 0.35
        assert entry["confidence"] == 0.85
        assert entry["market_type"] == "temperature"
        assert entry["location"] == "Phoenix, AZ"
        assert entry["lead_time_hours"] == 24.0
        assert any("ECMWF" in s for s in entry["sources"])

    def test_json_sorted_by_edge(self, multiple_signals):
        result = format_json(multiple_signals)
        data = json.loads(result)
        edges = [abs(d["edge"]) for d in data]
        assert edges == sorted(edges, reverse=True)

    def test_json_empty_list(self):
        result = format_json([])
        data = json.loads(result)
        assert data == []

    def test_json_timestamp_iso(self, sample_signal):
        result = format_json([sample_signal])
        data = json.loads(result)
        # Should be valid ISO format
        datetime.fromisoformat(data[0]["timestamp"])


class TestFormatCsv:
    def test_csv_valid(self, sample_signal):
        result = format_csv([sample_signal])
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 2  # Header + 1 data row

    def test_csv_header(self, sample_signal):
        result = format_csv([sample_signal])
        reader = csv.reader(io.StringIO(result))
        header = next(reader)
        assert "market_id" in header
        assert "direction" in header
        assert "edge" in header
        assert "kelly_fraction" in header

    def test_csv_data(self, sample_signal):
        result = format_csv([sample_signal])
        reader = csv.reader(io.StringIO(result))
        next(reader)  # Skip header
        row = next(reader)
        assert "test-market-001" in row
        assert "YES" in row

    def test_csv_multiple_rows(self, multiple_signals):
        result = format_csv(multiple_signals)
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 4  # Header + 3 data rows

    def test_csv_sorted_by_edge(self, multiple_signals):
        result = format_csv(multiple_signals)
        reader = csv.reader(io.StringIO(result))
        header = next(reader)
        edge_idx = header.index("edge")
        edges = [abs(float(row[edge_idx])) for row in reader]
        assert edges == sorted(edges, reverse=True)

    def test_csv_sources_pipe_separated(self, sample_signal):
        result = format_csv([sample_signal])
        # Sources should be pipe-separated in CSV
        assert "ECMWF (51 members)|GFS (31 members)" in result

    def test_csv_empty_list(self):
        result = format_csv([])
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 1  # Header only


class TestFormatTable:
    def test_table_no_error(self, sample_signal):
        """Rich table rendering should not raise."""
        from rich.console import Console
        console = Console(file=io.StringIO(), width=200)
        format_table([sample_signal], console)
        output = console.file.getvalue()
        assert "Weather Edge Signals" in output

    def test_table_empty_signals(self):
        """Empty signal list should show 'no signals' message."""
        from rich.console import Console
        console = Console(file=io.StringIO())
        format_table([], console)
        output = console.file.getvalue()
        assert "No signals" in output

    def test_table_contains_data(self, multiple_signals):
        """Table should contain key data from signals."""
        from rich.console import Console
        console = Console(file=io.StringIO(), width=200)
        format_table(multiple_signals, console)
        output = console.file.getvalue()
        assert "YES" in output
        assert "NO" in output
        assert "3 signal(s) total" in output

    def test_table_signal_count(self, multiple_signals):
        """Table footer should show correct signal count."""
        from rich.console import Console
        console = Console(file=io.StringIO(), width=200)
        format_table(multiple_signals, console)
        output = console.file.getvalue()
        assert "3 signal(s)" in output
