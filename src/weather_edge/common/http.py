"""Shared async HTTP client with retry and caching."""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from weather_edge.config import get_settings


def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient HTTP errors and timeouts."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


_retry_decorator = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


class HttpClient:
    """Async HTTP client with retry logic."""

    def __init__(self, base_url: str = "", headers: dict[str, str] | None = None) -> None:
        settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers or {},
            timeout=httpx.Timeout(settings.http_timeout),
        )

    @_retry_decorator
    async def get(self, url: str, params: dict | None = None) -> httpx.Response:
        resp = await self._client.get(url, params=params)
        resp.raise_for_status()
        return resp

    @_retry_decorator
    async def post(self, url: str, json: dict | None = None) -> httpx.Response:
        resp = await self._client.post(url, json=json)
        resp.raise_for_status()
        return resp

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
