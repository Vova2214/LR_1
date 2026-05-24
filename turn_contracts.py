from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from .. import schemas


@dataclass(slots=True)
class TurnPlanResult:
    narration: str
    choices: list[schemas.NarratorChoice]
    zone_scope: uuid.UUID | None
    parsed_ops: list[schemas.PatchOp]
    validator_status: Literal["ok", "reject", "uncertain"]
    validator_reasons: list[str]
    raw_response: dict[str, Any]
    librarian_used: bool
    memory_candidates: list[schemas.MemoryCandidate] = field(default_factory=list)
    semantic_events: list[schemas.SemanticEvent] = field(default_factory=list)
    scene_entities: list[schemas.SceneEntity] = field(default_factory=list)
    memory_trace: schemas.MemoryTrace | None = None
    llm_usage: dict[str, Any] = field(default_factory=dict)
    consequence_seeds: list[schemas.ConsequenceSeed] = field(default_factory=list)
    consequence_intents: list[schemas.ConsequenceIntent] = field(default_factory=list)
    resolved_consequence_ids: list[str] = field(default_factory=list)
    planner_contract_version: int = 1


@dataclass(slots=True, frozen=True)
class TurnApplyExternalRequest:
    kind: Literal["embedding", "dedup_arbiter", "profile_text"]
    instruction: str | None = None
    texts: tuple[str, ...] = ()
    dedup_payload: dict[str, Any] | None = None
    profile_cache_key: str | None = None
    profile_object_type: str | None = None
    profile_name: str | None = None
    profile_data: dict[str, Any] | None = None
    profile_fallback_text: str | None = None


@dataclass(slots=True)
class TurnApplyExternalArtifacts:
    embedding_vectors: dict[tuple[str | None, str], list[float]] = field(default_factory=dict)
    dedup_arbiter_decisions: dict[str, bool] = field(default_factory=dict)
    profile_texts: dict[str, str] = field(default_factory=dict)


class TurnApplyExternalPreparationRequired(RuntimeError):
    def __init__(self, request: TurnApplyExternalRequest) -> None:
        super().__init__(f"turn apply external preparation required: {request.kind}")
        self.request = request
