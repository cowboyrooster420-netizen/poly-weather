"""Application configuration via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Anthropic API key for Claude Haiku calls
    anthropic_api_key: str = ""

    # Kelly criterion fraction (0.25 = quarter-Kelly)
    kelly_fraction: float = 0.25

    # Minimum edge to generate a signal
    min_edge: float = 0.05

    # SQLite database path for signal tracking
    db_path: Path = Path.home() / ".weather-edge" / "signals.db"

    # NOAA/NWS user agent string
    nws_user_agent: str = "weather-edge (weather-edge@example.com)"

    # Polymarket Gamma API base URL
    gamma_api_url: str = "https://gamma-api.polymarket.com"

    # CLOB API base URL
    clob_api_url: str = "https://clob.polymarket.com"

    # Open-Meteo base URL
    openmeteo_api_url: str = "https://ensemble-api.open-meteo.com/v1"

    # NOAA/NWS base URL
    nws_api_url: str = "https://api.weather.gov"

    # ECMWF weight for blending (GFS gets 1 - this)
    ecmwf_weight: float = 0.6

    # HTTP request timeout seconds
    http_timeout: float = 30.0

    # Max concurrent API requests
    max_concurrency: int = 10

    # Telegram notifications
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_enabled: bool = False
    telegram_high_edge: float = 0.10

    @field_validator("ecmwf_weight")
    @classmethod
    def _ecmwf_weight_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"ecmwf_weight must be in [0, 1], got {v}")
        return v

    @field_validator("kelly_fraction")
    @classmethod
    def _kelly_fraction_in_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"kelly_fraction must be in (0, 1], got {v}")
        return v

    @field_validator("min_edge")
    @classmethod
    def _min_edge_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"min_edge must be >= 0, got {v}")
        return v


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
