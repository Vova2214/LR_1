from __future__ import annotations

import threading
import time
from typing import Any, Callable


class CircuitOpenError(RuntimeError):
    def __init__(self, provider: str, retry_after_seconds: float) -> None:
        super().__init__(f"Circuit open for provider '{provider}'")
        self.provider = provider
        self.retry_after_seconds = max(float(retry_after_seconds), 0.0)


class _SimpleCircuitBreaker:
    def __init__(self, provider: str, *, fail_max: int, reset_timeout: float) -> None:
        self.provider = provider
        self.fail_max = max(int(fail_max), 1)
        self.reset_timeout = max(float(reset_timeout), 1.0)
        self._lock = threading.Lock()
        self._state = "closed"
        self._failure_count = 0
        self._opened_at = 0.0
        self._probe_started_at = 0.0
        self._probe_in_flight = False

    def _reset_probe_state(self) -> None:
        self._probe_started_at = 0.0
        self._probe_in_flight = False

    def _acquire_half_open_probe(self, now: float) -> None:
        if self._probe_in_flight:
            elapsed = now - self._probe_started_at
            remaining = self.reset_timeout - elapsed
            if remaining > 0:
                raise CircuitOpenError(self.provider, remaining)

        self._probe_in_flight = True
        self._probe_started_at = now

    def ensure_available(self) -> None:
        with self._lock:
            if self._state == "closed":
                return
            now = time.monotonic()
            if self._state == "open":
                elapsed = now - self._opened_at
                remaining = self.reset_timeout - elapsed
                if remaining > 0:
                    raise CircuitOpenError(self.provider, remaining)
                self._state = "half_open"
                self._reset_probe_state()

            self._acquire_half_open_probe(now)

    def mark_success(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._opened_at = 0.0
            self._reset_probe_state()

    def mark_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._state == "half_open" or self._failure_count >= self.fail_max:
                self._state = "open"
                self._opened_at = time.monotonic()
                self._reset_probe_state()

    def call(self, func: Callable[[], Any]) -> Any:
        self.ensure_available()
        try:
            value = func()
        except Exception:
            self.mark_failure()
            raise
        self.mark_success()
        return value


_breakers: dict[str, _SimpleCircuitBreaker] = {}
_breakers_lock = threading.Lock()
_DEFAULT_FAIL_MAX = 5
_DEFAULT_RESET_TIMEOUT_SECONDS = 30.0


def _get_breaker(
    provider: str,
    *,
    fail_max: int,
    reset_timeout_seconds: float,
) -> _SimpleCircuitBreaker:
    with _breakers_lock:
        breaker = _breakers.get(provider)
        if breaker is None:
            breaker = _SimpleCircuitBreaker(
                provider=provider,
                fail_max=fail_max,
                reset_timeout=reset_timeout_seconds,
            )
            _breakers[provider] = breaker
    return breaker


def ensure_provider_available(
    provider: str,
    *,
    fail_max: int = _DEFAULT_FAIL_MAX,
    reset_timeout_seconds: float = _DEFAULT_RESET_TIMEOUT_SECONDS,
) -> None:
    _get_breaker(
        provider,
        fail_max=fail_max,
        reset_timeout_seconds=reset_timeout_seconds,
    ).ensure_available()


def record_provider_success(
    provider: str,
    *,
    fail_max: int = _DEFAULT_FAIL_MAX,
    reset_timeout_seconds: float = _DEFAULT_RESET_TIMEOUT_SECONDS,
) -> None:
    _get_breaker(
        provider,
        fail_max=fail_max,
        reset_timeout_seconds=reset_timeout_seconds,
    ).mark_success()


def record_provider_failure(
    provider: str,
    *,
    fail_max: int = _DEFAULT_FAIL_MAX,
    reset_timeout_seconds: float = _DEFAULT_RESET_TIMEOUT_SECONDS,
) -> None:
    _get_breaker(
        provider,
        fail_max=fail_max,
        reset_timeout_seconds=reset_timeout_seconds,
    ).mark_failure()


def call_with_circuit_breaker(
    provider: str,
    func: Callable[[], Any],
    *,
    fail_max: int = _DEFAULT_FAIL_MAX,
    reset_timeout_seconds: float = _DEFAULT_RESET_TIMEOUT_SECONDS,
) -> Any:
    return _get_breaker(
        provider,
        fail_max=fail_max,
        reset_timeout_seconds=reset_timeout_seconds,
    ).call(func)
