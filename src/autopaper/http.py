from __future__ import annotations

import json
import logging
import time

import httpx

from autopaper.config import Settings
from autopaper.errors import SourceError

log = logging.getLogger(__name__)


class HttpClient:
    """带间隔和有限重试的 HTTP 客户端。各数据源用 bucket 分开限速。"""

    def __init__(self, *, timeout: float, user_agent: str, delays: dict[str, float]):
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )
        self._delays = delays
        self._next_ok: dict[str, float] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> HttpClient:
        return cls(
            timeout=settings.timeout_seconds,
            user_agent=settings.user_agent,
            delays=settings.http_delays(),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get_text(self, url: str, *, params=None, headers=None, bucket: str = "default") -> str:
        return self._request(url, params=params, headers=headers, bucket=bucket).text

    def get_json(self, url: str, *, params=None, headers=None, bucket: str = "default"):
        response = self._request(url, params=params, headers=headers, bucket=bucket)
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise SourceError(bucket, "响应不是 JSON") from exc

    def _throttle(self, bucket: str) -> None:
        delay = self._delays.get(bucket, self._delays.get("default", 0.0))
        if delay <= 0:
            return
        now = time.monotonic()
        wait = self._next_ok.get(bucket, 0.0) - now
        if wait > 0:
            time.sleep(wait)
        self._next_ok[bucket] = time.monotonic() + delay

    def _request(self, url: str, *, params, headers, bucket: str) -> httpx.Response:
        last_error = "请求失败"
        for attempt in range(4):
            self._throttle(bucket)
            try:
                response = self._client.get(url, params=params, headers=headers)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                time.sleep(1.0 * (attempt + 1))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                if attempt >= 3:
                    break
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    time.sleep(min(int(retry_after), 20))
                elif response.status_code == 429:
                    time.sleep(min(20, 3 * (attempt + 1)))
                else:
                    time.sleep(1.0 * (attempt + 1))
                continue
            if response.status_code >= 400:
                detail = response.text[:300].replace("\n", " ")
                raise SourceError(bucket, f"HTTP {response.status_code} {detail}")
            return response
        raise SourceError(bucket, last_error)
