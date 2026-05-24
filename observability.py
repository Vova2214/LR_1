from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import contextvars
import json
import logging
import threading
import time
from typing import Any, Iterator


_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
_logging_configured = False
_logging_lock = threading.Lock()

logger = logging.getLogger(__name__)


def configure_structured_logging() -> None:
    global _logging_configured
    if _logging_configured:
        return
    with _logging_lock:
        if _logging_configured:
            return
        try:
            import structlog  # type: ignore
        except Exception:
            _logging_configured = True
            return

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            cache_logger_on_first_use=True,
        )
        _logging_configured = True


def set_trace_id(trace_id: str) -> contextvars.Token[str | None]:
    token = _trace_id_var.set(trace_id)
    try:
        import structlog  # type: ignore
    except Exception:
        return token
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    return token


def reset_trace_id(token: contextvars.Token[str | None]) -> None:
    _trace_id_var.reset(token)
    try:
        import structlog  # type: ignore
    except Exception:
        return
    structlog.contextvars.unbind_contextvars("trace_id")


def get_trace_id() -> str | None:
    return _trace_id_var.get()


def trace_extra(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(extra or {})
    trace_id = get_trace_id()
    if trace_id:
        payload.setdefault("trace_id", trace_id)
    return payload


class _Counter:
    def __init__(self, name: str, label_names: tuple[str, ...] = ()) -> None:
        self.name = name
        self.label_names = label_names
        self._values: defaultdict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount <= 0:
            return
        key = _normalize_labels(self.label_names, labels)
        with self._lock:
            self._values[key] += float(amount)

    def collect(self) -> dict[tuple[tuple[str, str], ...], float]:
        with self._lock:
            return dict(self._values)


class _Histogram:
    def __init__(
        self,
        name: str,
        *,
        buckets: tuple[float, ...],
        label_names: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.label_names = label_names
        self.buckets = tuple(sorted(set(buckets)))
        self._bucket_counts: defaultdict[tuple[tuple[str, str], ...], list[int]] = defaultdict(
            lambda: [0 for _ in self.buckets]
        )
        self._sum: defaultdict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._count: defaultdict[tuple[tuple[str, str], ...], int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        if value < 0:
            value = 0.0
        key = _normalize_labels(self.label_names, labels)
        with self._lock:
            # Internal storage is per-boundary hit counts (non-cumulative).
            # Prometheus cumulative buckets are assembled in _render_histogram.
            buckets = self._bucket_counts[key]
            for idx, boundary in enumerate(self.buckets):
                if value <= boundary:
                    buckets[idx] += 1
                    break
            self._sum[key] += float(value)
            self._count[key] += 1

    def collect(self) -> tuple[
        dict[tuple[tuple[str, str], ...], list[int]],
        dict[tuple[tuple[str, str], ...], float],
        dict[tuple[tuple[str, str], ...], int],
    ]:
        with self._lock:
            return (
                {k: list(v) for k, v in self._bucket_counts.items()},
                dict(self._sum),
                dict(self._count),
            )


def _normalize_labels(
    label_names: tuple[str, ...],
    labels: dict[str, str],
) -> tuple[tuple[str, str], ...]:
    if not label_names:
        return ()
    normalized: list[tuple[str, str]] = []
    for name in label_names:
        normalized.append((name, str(labels.get(name, ""))))
    return tuple(normalized)


_HISTOGRAM_BUCKETS_SECONDS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, 120.0)

turns_total = _Counter("rpg_turns_total", ("status",))
turn_duration_seconds = _Histogram(
    "rpg_turn_duration_seconds",
    buckets=_HISTOGRAM_BUCKETS_SECONDS,
    label_names=("status",),
)
llm_calls_total = _Counter("rpg_llm_calls_total", ("provider",))
llm_tokens_total = _Counter("rpg_llm_tokens_total", ("provider", "type"))
cache_lookups_total = _Counter("rpg_cache_lookups_total", ("cache", "tier", "result"))
canon_repairs_total = _Counter("rpg_canon_repairs_total", ("result",))
callback_decisions_total = _Counter("rpg_callback_decisions_total", ("result",))
memory_relevant_surfaced_total = _Counter("rpg_memory_relevant_surfaced_total")
memory_relevant_used_total = _Counter("rpg_memory_relevant_used_total")
memory_callback_surfaced_total = _Counter("rpg_memory_callback_surfaced_total")
memory_callback_useful_total = _Counter("rpg_memory_callback_useful_total")
memory_callback_false_resurfacing_total = _Counter("rpg_memory_callback_false_resurfacing_total")
memory_bundle_surfaced_total = _Counter("rpg_memory_bundle_surfaced_total")
memory_bundle_used_total = _Counter("rpg_memory_bundle_used_total")
memory_continuity_expected_total = _Counter("rpg_memory_continuity_expected_total")
memory_continuity_missed_total = _Counter("rpg_memory_continuity_missed_total")
memory_transition_ambiguity_total = _Counter("rpg_memory_transition_ambiguity_total")
memory_bundle_pressure_total = _Counter("rpg_memory_bundle_pressure_total")
memory_review_runs_total = _Counter("rpg_memory_review_runs_total")
memory_review_findings_total = _Counter("rpg_memory_review_findings_total", ("kind",))
memory_benchmark_runs_total = _Counter("rpg_memory_benchmark_runs_total", ("benchmark",))
memory_benchmark_score = _Histogram(
    "rpg_memory_benchmark_score_ratio",
    buckets=(0.2, 0.4, 0.6, 0.8, 1.0),
    label_names=("benchmark",),
)


def record_turn(status: str, duration_seconds: float) -> None:
    turns_total.inc(status=status)
    turn_duration_seconds.observe(duration_seconds, status=status)


def record_llm_call(
    *,
    provider: str,
    duration_seconds: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
) -> None:
    llm_calls_total.inc(provider=provider)
    turn_duration_seconds.observe(duration_seconds, status=f"llm:{provider}")
    if prompt_tokens > 0:
        llm_tokens_total.inc(float(prompt_tokens), provider=provider, type="prompt")
    if completion_tokens > 0:
        llm_tokens_total.inc(float(completion_tokens), provider=provider, type="completion")
    if total_tokens > 0:
        llm_tokens_total.inc(float(total_tokens), provider=provider, type="total")


def record_cache_lookup(*, cache: str, tier: str, result: str) -> None:
    cache_lookups_total.inc(cache=cache, tier=tier, result=result)


def record_canon_repair(result: str) -> None:
    canon_repairs_total.inc(result=result)


def record_callback_decision(result: str) -> None:
    callback_decisions_total.inc(result=result)


def record_memory_surface(
    *,
    relevant_count: int = 0,
    callback_count: int = 0,
    bundle_count: int = 0,
) -> None:
    if relevant_count > 0:
        memory_relevant_surfaced_total.inc(float(relevant_count))
    if callback_count > 0:
        memory_callback_surfaced_total.inc(float(callback_count))
    if bundle_count > 0:
        memory_bundle_surfaced_total.inc(float(bundle_count))


def record_memory_usage(
    *,
    relevant_used: int = 0,
    callback_used: int = 0,
    bundle_used: int = 0,
) -> None:
    if relevant_used > 0:
        memory_relevant_used_total.inc(float(relevant_used))
    if callback_used > 0:
        memory_callback_useful_total.inc(float(callback_used))
    if bundle_used > 0:
        memory_bundle_used_total.inc(float(bundle_used))


def record_memory_false_resurfacing(count: int = 1) -> None:
    if count > 0:
        memory_callback_false_resurfacing_total.inc(float(count))


def record_memory_continuity_miss(*, expected_count: int = 0, missed_count: int = 0) -> None:
    if expected_count > 0:
        memory_continuity_expected_total.inc(float(expected_count))
    if missed_count > 0:
        memory_continuity_missed_total.inc(float(missed_count))


def record_memory_transition_ambiguity(count: int = 1) -> None:
    if count > 0:
        memory_transition_ambiguity_total.inc(float(count))


def record_memory_bundle_pressure(count: int = 1) -> None:
    if count > 0:
        memory_bundle_pressure_total.inc(float(count))


def record_memory_review_run() -> None:
    memory_review_runs_total.inc()


def record_memory_review_findings(kind: str, count: int = 1) -> None:
    if count > 0:
        memory_review_findings_total.inc(float(count), kind=str(kind or "unknown"))


def record_memory_benchmark_result(*, benchmark: str, score: float) -> None:
    normalized_benchmark = str(benchmark or "unknown")
    memory_benchmark_runs_total.inc(benchmark=normalized_benchmark)
    memory_benchmark_score.observe(max(min(float(score), 1.0), 0.0), benchmark=normalized_benchmark)


@contextmanager
def time_block() -> Iterator[float]:
    started = time.perf_counter()
    yield started


def render_prometheus(extra_gauges: dict[str, float] | None = None) -> str:
    lines: list[str] = []

    _render_counter(lines, turns_total)
    _render_counter(lines, llm_calls_total)
    _render_counter(lines, llm_tokens_total)
    _render_counter(lines, cache_lookups_total)
    _render_counter(lines, canon_repairs_total)
    _render_counter(lines, callback_decisions_total)
    _render_counter(lines, memory_relevant_surfaced_total)
    _render_counter(lines, memory_relevant_used_total)
    _render_counter(lines, memory_callback_surfaced_total)
    _render_counter(lines, memory_callback_useful_total)
    _render_counter(lines, memory_callback_false_resurfacing_total)
    _render_counter(lines, memory_bundle_surfaced_total)
    _render_counter(lines, memory_bundle_used_total)
    _render_counter(lines, memory_continuity_expected_total)
    _render_counter(lines, memory_continuity_missed_total)
    _render_counter(lines, memory_transition_ambiguity_total)
    _render_counter(lines, memory_bundle_pressure_total)
    _render_counter(lines, memory_review_runs_total)
    _render_counter(lines, memory_review_findings_total)
    _render_counter(lines, memory_benchmark_runs_total)
    _render_histogram(lines, turn_duration_seconds)
    _render_histogram(lines, memory_benchmark_score)

    gauges = dict(extra_gauges or {})
    gauges.setdefault(
        "rpg_memory_recall_hit_rate",
        _safe_ratio(_counter_total(memory_relevant_used_total), _counter_total(memory_relevant_surfaced_total)),
    )
    gauges.setdefault(
        "rpg_memory_callback_usefulness_rate",
        _safe_ratio(_counter_total(memory_callback_useful_total), _counter_total(memory_callback_surfaced_total)),
    )
    gauges.setdefault(
        "rpg_memory_false_resurfacing_rate",
        _safe_ratio(_counter_total(memory_callback_false_resurfacing_total), _counter_total(memory_callback_surfaced_total)),
    )
    gauges.setdefault(
        "rpg_memory_continuity_miss_rate",
        _safe_ratio(_counter_total(memory_continuity_missed_total), _counter_total(memory_continuity_expected_total)),
    )
    for name, value in sorted(gauges.items()):
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {float(value)}")

    return "\n".join(lines) + "\n"


def _render_counter(lines: list[str], metric: _Counter) -> None:
    lines.append(f"# TYPE {metric.name} counter")
    rows = metric.collect()
    if not rows:
        lines.append(f"{metric.name} 0.0")
        return
    for labels, value in sorted(rows.items()):
        lines.append(f"{metric.name}{_labels_to_text(labels)} {value}")


def _counter_total(metric: _Counter) -> float:
    return float(sum(metric.collect().values()))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _render_histogram(lines: list[str], metric: _Histogram) -> None:
    lines.append(f"# TYPE {metric.name} histogram")
    buckets_by_label, sums, counts = metric.collect()
    if not buckets_by_label:
        for boundary in metric.buckets:
            lines.append(f"{metric.name}_bucket{{le=\"{_format_bucket(boundary)}\"}} 0")
        lines.append(f"{metric.name}_bucket{{le=\"+Inf\"}} 0")
        lines.append(f"{metric.name}_sum 0")
        lines.append(f"{metric.name}_count 0")
        return

    for labels, bucket_counts in sorted(buckets_by_label.items()):
        cumulative = 0
        for idx, boundary in enumerate(metric.buckets):
            cumulative += int(bucket_counts[idx])
            bucket_labels = labels + (("le", _format_bucket(boundary)),)
            lines.append(f"{metric.name}_bucket{_labels_to_text(bucket_labels)} {cumulative}")
        inf_labels = labels + (("le", "+Inf"),)
        lines.append(f"{metric.name}_bucket{_labels_to_text(inf_labels)} {counts.get(labels, 0)}")
        lines.append(f"{metric.name}_sum{_labels_to_text(labels)} {sums.get(labels, 0.0)}")
        lines.append(f"{metric.name}_count{_labels_to_text(labels)} {counts.get(labels, 0)}")


def _format_bucket(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


def _labels_to_text(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    encoded = ",".join(f'{name}={json.dumps(value)}' for name, value in labels)
    return "{" + encoded + "}"
