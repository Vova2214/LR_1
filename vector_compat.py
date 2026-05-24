from __future__ import annotations

import json
from typing import Any, Iterable

from sqlalchemy import cast, literal
from sqlalchemy.types import UserDefinedType

try:
    from pgvector.sqlalchemy import Vector as Vector  # type: ignore[import-untyped]
except ImportError:
    # Fallback type to keep app/migrations runnable in offline environments.
    # In production use the real `pgvector` package from requirements.txt.

    def _format_vector(value: Iterable[float]) -> str:
        return "[" + ",".join(str(float(x)) for x in value) + "]"

    def _parse_vector(value: Any) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [float(x) for x in value]
        if isinstance(value, tuple):
            return [float(x) for x in value]
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = bytes(value).decode("utf-8")
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid vector text value: {value!r}") from exc
            if not isinstance(parsed, list):
                raise ValueError(f"Vector value must decode to list, got {type(parsed).__name__}")
            return [float(x) for x in parsed]
        raise TypeError(f"Unsupported vector value type: {type(value).__name__}")

    class _VectorComparator(UserDefinedType.Comparator):
        def _coerce_other(self, other: Any) -> Any:
            if isinstance(other, (list, tuple)):
                return cast(literal(_format_vector(other)), self.expr.type)
            return other

        def cosine_distance(self, other: Any) -> Any:
            return self.expr.op("<=>")(self._coerce_other(other))

        def l2_distance(self, other: Any) -> Any:
            return self.expr.op("<->")(self._coerce_other(other))

    class _FallbackVector(UserDefinedType):
        cache_ok = True
        comparator_factory = _VectorComparator

        def __init__(self, dim: int):
            self.dim = dim

        def get_col_spec(self, **kw: Any) -> str:
            return f"VECTOR({self.dim})"

        def bind_processor(self, dialect: Any):
            def process(value: Any) -> str | None:
                if value is None:
                    return None
                if isinstance(value, str):
                    return value
                return _format_vector(value)

            return process

        def result_processor(self, dialect: Any, coltype: Any):
            def process(value: Any) -> list[float] | None:
                return _parse_vector(value)

            return process

    Vector = _FallbackVector
