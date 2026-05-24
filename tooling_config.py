from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditToolingConfig:
    trace_index_default_limit: int = 25
    trace_index_max_limit: int = 200


audit_tooling_config = AuditToolingConfig()
