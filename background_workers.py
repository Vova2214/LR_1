from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_workers_lock = threading.Lock()
_workers: set[threading.Thread] = set()
_shutdown_requested = threading.Event()


def _cleanup_finished_workers_locked() -> None:
    finished = [worker for worker in _workers if not worker.is_alive()]
    for worker in finished:
        _workers.discard(worker)


def allow_new_workers() -> None:
    _shutdown_requested.clear()


def begin_shutdown() -> None:
    _shutdown_requested.set()


def shutdown_requested() -> bool:
    return _shutdown_requested.is_set()


def wait_for_shutdown(timeout_seconds: float) -> bool:
    return _shutdown_requested.wait(timeout=max(float(timeout_seconds), 0.0))


def start_background_worker(
    *,
    target: Callable[..., Any],
    kwargs: dict[str, Any],
    name: str,
) -> threading.Thread | None:
    payload = dict(kwargs)

    with _workers_lock:
        _cleanup_finished_workers_locked()
        if _shutdown_requested.is_set():
            logger.info("Background worker skipped during shutdown: %s", name)
            return None

        def _runner() -> None:
            try:
                target(**payload)
            finally:
                with _workers_lock:
                    _workers.discard(threading.current_thread())

        worker = threading.Thread(
            target=_runner,
            daemon=True,
            name=name,
        )
        _workers.add(worker)

    worker.start()
    return worker


def wait_for_background_workers(timeout_seconds: float = 8.0) -> int:
    deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
    while True:
        with _workers_lock:
            _cleanup_finished_workers_locked()
            active_workers = list(_workers)
        if not active_workers:
            return 0

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return len(active_workers)

        join_slice = min(remaining, 0.2)
        for worker in active_workers:
            worker.join(timeout=join_slice)


def _reset_for_tests() -> None:
    with _workers_lock:
        _workers.clear()
        _shutdown_requested.clear()
