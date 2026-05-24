from __future__ import annotations

from typing import Any

from .db import LLM_MODEL_PRICING_JSON, LLM_PRICING_REVISION


_RATE_KEYS = ("prompt_cents_per_1k", "completion_cents_per_1k")


def _to_non_negative_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
    else:
        return None

    if parsed < 0:
        return None
    return parsed


def _extract_rates(payload: Any) -> tuple[float, float] | None:
    if not isinstance(payload, dict):
        return None

    prompt_rate = _to_non_negative_float(payload.get(_RATE_KEYS[0]))
    completion_rate = _to_non_negative_float(payload.get(_RATE_KEYS[1]))
    if prompt_rate is None or completion_rate is None:
        return None
    return (prompt_rate, completion_rate)


def model_pricing_rates(model_name: str) -> tuple[float, float] | None:
    model_key = str(model_name or "").strip()
    if not model_key:
        return None

    source = LLM_MODEL_PRICING_JSON
    if not isinstance(source, dict):
        return None

    rates = _extract_rates(source.get(model_key))
    if rates is not None:
        return rates

    for fallback_key in ("*", "default", "__default__"):
        rates = _extract_rates(source.get(fallback_key))
        if rates is not None:
            return rates
    return None


def estimate_cost_cents(
    *,
    model_name: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    rates = model_pricing_rates(model_name)
    if rates is None:
        return None
    prompt_rate, completion_rate = rates

    prompt = max(int(prompt_tokens or 0), 0)
    completion = max(int(completion_tokens or 0), 0)
    cost = (prompt / 1000.0) * prompt_rate + (completion / 1000.0) * completion_rate
    if cost <= 0:
        return 0.0
    return round(cost, 4)


def pricing_revision() -> str:
    return str(LLM_PRICING_REVISION or "").strip() or "unknown"


__all__ = [
    "estimate_cost_cents",
    "model_pricing_rates",
    "pricing_revision",
]
