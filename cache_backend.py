from __future__ import annotations

from collections import OrderedDict
import json
import logging
import threading
import time
from typing import Any, Callable, TypeVar

from .db import (
    REDIS_CACHE_CONNECT_TIMEOUT_SECONDS,
    REDIS_CACHE_NAMESPACE,
    REDIS_CACHE_TTL_SECONDS,
    REDIS_URL,
    USE_REDIS_CACHE,
)
from .observability import record_cache_lookup

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_SERIALIZER = Callable[[Any], str]
_PARSER = Callable[[str], _T]

_redis_client: Any | None = None
_redis_lock = threading.Lock()
_redis_next_retry_at = 0.0
_REDIS_RETRY_BACKOFF_SECONDS = 30.0


def _mark_redis_client_unavailable(client: Any | None) -> None:
    global _redis_client, _redis_next_retry_at

    client_to_close = None
    with _redis_lock:
        if client is not None and _redis_client is not None and _redis_client is not client:
            return
        client_to_close = _redis_client if _redis_client is not None else client
        _redis_client = None
        _redis_next_retry_at = time.monotonic() + _REDIS_RETRY_BACKOFF_SECONDS

    close = getattr(client_to_close, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            logger.warning("Redis client close failed during backoff transition", exc_info=True)


def _get_redis_client() -> Any | None:
    global _redis_client, _redis_next_retry_at
    if not USE_REDIS_CACHE or not REDIS_URL.strip():
        return None
    if _redis_client is not None:
        return _redis_client

    now = time.monotonic()
    if now < _redis_next_retry_at:
        return None

    with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        if time.monotonic() < _redis_next_retry_at:
            return None
        try:
            import redis  # type: ignore
        except Exception:  # noqa: BLE001
            _redis_next_retry_at = time.monotonic() + _REDIS_RETRY_BACKOFF_SECONDS
            logger.warning(
                "Redis cache backend unavailable (redis package missing), fallback to in-memory cache only"
            )
            return None

        try:
            candidate = redis.Redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=REDIS_CACHE_CONNECT_TIMEOUT_SECONDS,
                socket_timeout=REDIS_CACHE_CONNECT_TIMEOUT_SECONDS,
            )
            candidate.ping()
        except Exception:  # noqa: BLE001
            _redis_next_retry_at = time.monotonic() + _REDIS_RETRY_BACKOFF_SECONDS
            logger.warning(
                "Redis cache backend unreachable, fallback to in-memory cache only",
                exc_info=True,
            )
            return None

        _redis_client = candidate
        _redis_next_retry_at = 0.0
        logger.info("Redis cache backend connected")
        return _redis_client


class TwoTierCache:
    def __init__(
        self,
        *,
        name: str,
        max_entries: int,
        ttl_seconds: int | None = None,
    ) -> None:
        self.name = name
        self.max_entries = max(int(max_entries), 1)
        self.ttl_seconds = max(int(ttl_seconds or REDIS_CACHE_TTL_SECONDS), 1)
        self._l1: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._prefix = f"{REDIS_CACHE_NAMESPACE}:{self.name}"

    def get(self, key: str, *, parser: _PARSER[_T] | None = None) -> _T | None:
        cache_key = str(key)
        now = time.monotonic()
        with self._lock:
            local = self._l1.get(cache_key)
            if local is not None:
                expires_at, value = local
                if expires_at > now:
                    self._l1.move_to_end(cache_key)
                    record_cache_lookup(cache=self.name, tier="l1", result="hit")
                    return value
                del self._l1[cache_key]
                record_cache_lookup(cache=self.name, tier="l1", result="expired")
            else:
                record_cache_lookup(cache=self.name, tier="l1", result="miss")

        client = _get_redis_client()
        if client is None:
            record_cache_lookup(cache=self.name, tier="l2", result="unavailable")
            return None
        try:
            raw = client.get(self._redis_key(cache_key))
        except Exception:  # noqa: BLE001
            _mark_redis_client_unavailable(client)
            logger.warning("Redis get failed for cache '%s'", self.name, exc_info=True)
            record_cache_lookup(cache=self.name, tier="l2", result="error")
            return None
        if raw is None:
            record_cache_lookup(cache=self.name, tier="l2", result="miss")
            return None

        try:
            parsed: Any = parser(raw) if parser is not None else raw
        except Exception:  # noqa: BLE001
            logger.warning("Redis parser failed for cache '%s'", self.name, exc_info=True)
            record_cache_lookup(cache=self.name, tier="l2", result="parse_error")
            return None

        self._set_l1(cache_key, parsed)
        record_cache_lookup(cache=self.name, tier="l2", result="hit")
        return parsed

    def set(self, key: str, value: Any, *, serializer: _SERIALIZER | None = None) -> None:
        cache_key = str(key)
        self._set_l1(cache_key, value)

        client = _get_redis_client()
        if client is None:
            return

        try:
            encoded = serializer(value) if serializer is not None else str(value)
        except Exception:  # noqa: BLE001
            logger.warning("Redis serializer failed for cache '%s'", self.name, exc_info=True)
            return

        try:
            client.set(self._redis_key(cache_key), encoded, ex=self.ttl_seconds)
        except Exception:  # noqa: BLE001
            _mark_redis_client_unavailable(client)
            logger.warning("Redis set failed for cache '%s'", self.name, exc_info=True)

    def clear(self) -> None:
        with self._lock:
            self._l1.clear()

        client = _get_redis_client()
        if client is None:
            return
        pattern = f"{self._prefix}:*"
        try:
            keys = list(client.scan_iter(match=pattern, count=512))
            if keys:
                client.delete(*keys)
        except Exception:  # noqa: BLE001
            _mark_redis_client_unavailable(client)
            logger.warning("Redis clear failed for cache '%s'", self.name, exc_info=True)

    def _set_l1(self, cache_key: str, value: Any) -> None:
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._l1[cache_key] = (expires_at, value)
            self._l1.move_to_end(cache_key)
            while len(self._l1) > self.max_entries:
                self._l1.popitem(last=False)

    def _redis_key(self, key: str) -> str:
        return f"{self._prefix}:{key}"


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def loads_json(raw: str) -> Any:
    return json.loads(raw)
