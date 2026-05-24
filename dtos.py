from __future__ import annotations

from dataclasses import dataclass, field
import uuid
from typing import Any

from .turn_contracts import TurnPlanResult


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_uuid(value: Any) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return uuid.UUID(text)
    except (TypeError, ValueError, AttributeError):
        return None


@dataclass(slots=True)
class TurnAllocation:
    new_turn: int
    in_game_day: int
    in_game_minute: int
    previous_day: int
    previous_minute: int
    turn_kind: str = "player"
    actor_object_id: uuid.UUID | None = None
    triggered_by_turn_index: int | None = None
    root_turn_index: int | None = None
    chain_depth: int = 0
    source_window_keys: list[str] = field(default_factory=list)
    agenda_key: str | None = None
    budget: int = 2
    planning_user_input: str | None = None


@dataclass(slots=True)
class TurnPlanningEnvelope:
    allocation: TurnAllocation
    plan: TurnPlanResult
    context_pack: dict[str, Any] | None


@dataclass(slots=True)
class TurnValidatedPlan:
    allocation: TurnAllocation
    plan: TurnPlanResult
    context_pack: dict[str, Any] | None


@dataclass(slots=True)
class TurnPatchApplication:
    allocation: TurnAllocation
    plan: TurnPlanResult
    context_pack: dict[str, Any] | None
    session_row: Any
    turn_row: Any
    state_payload: dict[str, Any]
    ttl_cleaned: int
    ref_map: dict[str, str]
    applied_ops: list[dict[str, Any]]
    resolved_zone_scope: uuid.UUID | None
    player_object_id: uuid.UUID | None
    memory_anchor_object_ids: list[str]
    effective_memory_candidates: list[Any]
    effective_durable_facts: list[Any]
    committed_events: list[dict[str, Any]] = field(default_factory=list)
    structural_signals: list[dict[str, Any]] = field(default_factory=list)
    consequence_windows_opened: list[dict[str, Any]] = field(default_factory=list)
    shown_consequence_ids: list[str] = field(default_factory=list)
    resolved_consequence_ids: list[str] = field(default_factory=list)
    consequence_validation: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TurnMemoryPersistence:
    stage: TurnPatchApplication
    memory_candidates_stored: dict[str, Any]


@dataclass(slots=True)
class TurnObservabilityPayload:
    stage: TurnMemoryPersistence
    ai_json: dict[str, Any]
    transition_ambiguity_count: int
    memory_debug: dict[str, Any]


@dataclass(slots=True)
class TurnApplyResult:
    turn_row: Any
    applied_ops: list[dict[str, Any]]
    resolved_zone_scope: uuid.UUID | None
    resolved_plan: TurnPlanResult
    context_pack: dict[str, Any] | None


@dataclass(slots=True)
class MemoryRetrievalRow:
    object_id: str
    prompt_id: str | None = None
    fact_key: str | None = None
    bundle_key: str | None = None
    obligation_key: str | None = None
    view_key: str | None = None
    kind: str | None = None
    layer: str | None = None
    lane: str | None = None
    memory_class: str | None = None
    why_surfaced: list[str] = field(default_factory=list)
    source_bundle_keys: list[str] = field(default_factory=list)
    source_obligation_keys: list[str] = field(default_factory=list)
    narrative_chain_keys: list[str] = field(default_factory=list)
    expectation_salience_score: float = 0.0
    expectation_debt_score: float = 0.0
    obligation_pressure_score: float = 0.0
    player_salience_score: float = 0.0
    epistemic_pressure_score: float = 0.0
    chain_influence_score: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MemoryRetrievalRow":
        row_payload = dict(payload or {})
        return cls(
            object_id=str(row_payload.get("object_id") or "").strip(),
            prompt_id=str(row_payload.get("prompt_id") or "").strip() or None,
            fact_key=str(row_payload.get("fact_key") or "").strip() or None,
            bundle_key=str(row_payload.get("bundle_key") or "").strip() or None,
            obligation_key=str(row_payload.get("obligation_key") or "").strip() or None,
            view_key=str(row_payload.get("view_key") or "").strip() or None,
            kind=str(row_payload.get("kind") or "").strip() or None,
            layer=str(row_payload.get("layer") or "").strip() or None,
            lane=str(row_payload.get("lane") or "").strip() or None,
            memory_class=str(row_payload.get("memory_class") or "").strip() or None,
            why_surfaced=[str(value).strip() for value in list(row_payload.get("why_surfaced") or []) if str(value).strip()],
            source_bundle_keys=[
                str(value).strip()
                for value in list(row_payload.get("source_bundle_keys") or [])
                if str(value).strip()
            ],
            source_obligation_keys=[
                str(value).strip()
                for value in list(row_payload.get("source_obligation_keys") or [])
                if str(value).strip()
            ],
            narrative_chain_keys=[
                str(value).strip()
                for value in list(row_payload.get("narrative_chain_keys") or [])
                if str(value).strip()
            ],
            expectation_salience_score=_safe_float(row_payload.get("expectation_salience_score")),
            expectation_debt_score=_safe_float(row_payload.get("expectation_debt_score")),
            obligation_pressure_score=_safe_float(row_payload.get("obligation_pressure_score")),
            player_salience_score=_safe_float(row_payload.get("player_salience_score")),
            epistemic_pressure_score=_safe_float(row_payload.get("epistemic_pressure_score")),
            chain_influence_score=_safe_float(row_payload.get("chain_influence_score")),
            payload=row_payload,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = dict(self.payload)
        payload["object_id"] = self.object_id
        payload["prompt_id"] = self.prompt_id
        payload["fact_key"] = self.fact_key
        payload["bundle_key"] = self.bundle_key
        payload["obligation_key"] = self.obligation_key
        payload["view_key"] = self.view_key
        payload["kind"] = self.kind
        payload["layer"] = self.layer
        payload["lane"] = self.lane
        payload["memory_class"] = self.memory_class
        payload["why_surfaced"] = list(self.why_surfaced)
        payload["source_bundle_keys"] = list(self.source_bundle_keys)
        payload["source_obligation_keys"] = list(self.source_obligation_keys)
        payload["narrative_chain_keys"] = list(self.narrative_chain_keys)
        payload["expectation_salience_score"] = self.expectation_salience_score
        payload["expectation_debt_score"] = self.expectation_debt_score
        payload["obligation_pressure_score"] = self.obligation_pressure_score
        payload["player_salience_score"] = self.player_salience_score
        payload["epistemic_pressure_score"] = self.epistemic_pressure_score
        payload["chain_influence_score"] = self.chain_influence_score
        return payload

    def to_observability_payload(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id or "",
            "object_id": self.object_id,
            "fact_key": self.fact_key,
            "bundle_key": self.bundle_key,
            "obligation_key": self.obligation_key,
            "view_key": self.view_key,
            "layer": self.layer or "",
            "memory_class": self.memory_class or "",
            "lane": self.lane or "",
            "lane_family_key": str(self.payload.get("lane_family_key") or "").strip() or None,
            "why_surfaced": list(self.why_surfaced),
            "expectation_salience_score": self.expectation_salience_score,
            "expectation_debt_score": self.expectation_debt_score,
            "obligation_pressure_score": self.obligation_pressure_score,
            "player_salience_score": self.player_salience_score,
            "epistemic_pressure_score": self.epistemic_pressure_score,
            "chain_influence_score": self.chain_influence_score,
            "narrative_chain_keys": list(self.narrative_chain_keys),
        }


@dataclass(slots=True)
class CallbackRow:
    fact_key: str
    kind: str
    text: str
    priority: str
    source_turn: int
    anchor_hits: int
    anchor_object_ids: list[str]
    relevance: float
    scene_mode: str
    callback_strength: str
    confidence: float
    durability: float
    emotional_weight: float
    obligation_weight: float
    sentimental_weight: float
    routine_weight: float
    zone_resurfacing_boost: float
    npc_revisit_boost: float
    player_emotional_boost: float
    player_sentimental_boost: float
    prompt_id: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CallbackRow":
        row_payload = dict(payload or {})
        return cls(
            fact_key=str(row_payload.get("fact_key") or "").strip(),
            kind=str(row_payload.get("kind") or "").strip(),
            text=str(row_payload.get("text") or "").strip(),
            priority=str(row_payload.get("priority") or "med").strip(),
            source_turn=_safe_int(row_payload.get("source_turn")),
            anchor_hits=_safe_int(row_payload.get("anchor_hits")),
            anchor_object_ids=[str(value) for value in list(row_payload.get("anchor_object_ids") or []) if str(value)],
            relevance=_safe_float(row_payload.get("relevance")),
            scene_mode=str(row_payload.get("scene_mode") or "").strip(),
            callback_strength=str(row_payload.get("callback_strength") or "soft").strip(),
            confidence=_safe_float(row_payload.get("confidence")),
            durability=_safe_float(row_payload.get("durability")),
            emotional_weight=_safe_float(row_payload.get("emotional_weight")),
            obligation_weight=_safe_float(row_payload.get("obligation_weight")),
            sentimental_weight=_safe_float(row_payload.get("sentimental_weight")),
            routine_weight=_safe_float(row_payload.get("routine_weight")),
            zone_resurfacing_boost=_safe_float(row_payload.get("zone_resurfacing_boost")),
            npc_revisit_boost=_safe_float(row_payload.get("npc_revisit_boost")),
            player_emotional_boost=_safe_float(row_payload.get("player_emotional_boost")),
            player_sentimental_boost=_safe_float(row_payload.get("player_sentimental_boost")),
            prompt_id=str(row_payload.get("prompt_id") or "").strip() or None,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "fact_key": self.fact_key,
            "kind": self.kind,
            "text": self.text,
            "priority": self.priority,
            "source_turn": self.source_turn,
            "anchor_hits": self.anchor_hits,
            "anchor_object_ids": list(self.anchor_object_ids),
            "relevance": self.relevance,
            "scene_mode": self.scene_mode,
            "callback_strength": self.callback_strength,
            "confidence": self.confidence,
            "durability": self.durability,
            "emotional_weight": self.emotional_weight,
            "obligation_weight": self.obligation_weight,
            "sentimental_weight": self.sentimental_weight,
            "routine_weight": self.routine_weight,
            "zone_resurfacing_boost": self.zone_resurfacing_boost,
            "npc_revisit_boost": self.npc_revisit_boost,
            "player_emotional_boost": self.player_emotional_boost,
            "player_sentimental_boost": self.player_sentimental_boost,
        }

    def to_observability_payload(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id or "",
            "fact_key": self.fact_key,
        }


@dataclass(slots=True)
class BundlePacket:
    object_id: str
    bundle_key: str
    prompt_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BundlePacket":
        row_payload = dict(payload or {})
        return cls(
            object_id=str(row_payload.get("object_id") or "").strip(),
            bundle_key=str(row_payload.get("bundle_key") or "").strip(),
            prompt_id=str(row_payload.get("prompt_id") or "").strip() or None,
            payload=row_payload,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = dict(self.payload)
        payload["object_id"] = self.object_id
        payload["bundle_key"] = self.bundle_key
        payload["prompt_id"] = self.prompt_id
        return payload

    def to_observability_payload(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id or "",
            "object_id": self.object_id,
            "bundle_key": self.bundle_key,
            "lane": str(self.payload.get("lane") or "narrative_bundles").strip(),
            "lane_family_key": str(self.payload.get("lane_family_key") or "").strip() or None,
        }


@dataclass(slots=True)
class ObligationNode:
    object_id: str
    obligation_key: str
    kind: str
    prompt_id: str | None = None
    why_surfaced: list[str] = field(default_factory=list)
    source_bundle_keys: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ObligationNode":
        row_payload = dict(payload or {})
        return cls(
            object_id=str(row_payload.get("object_id") or "").strip(),
            obligation_key=str(row_payload.get("obligation_key") or "").strip(),
            kind=str(row_payload.get("kind") or "").strip(),
            prompt_id=str(row_payload.get("prompt_id") or "").strip() or None,
            why_surfaced=[str(value).strip() for value in list(row_payload.get("why_surfaced") or []) if str(value).strip()],
            source_bundle_keys=[
                str(value).strip()
                for value in list(row_payload.get("source_bundle_keys") or [])
                if str(value).strip()
            ],
            payload=row_payload,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = dict(self.payload)
        payload["object_id"] = self.object_id
        payload["obligation_key"] = self.obligation_key
        payload["kind"] = self.kind
        payload["prompt_id"] = self.prompt_id
        payload["why_surfaced"] = list(self.why_surfaced)
        payload["source_bundle_keys"] = list(self.source_bundle_keys)
        return payload

    def to_observability_payload(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id or "",
            "object_id": self.object_id,
            "obligation_key": self.obligation_key,
            "kind": self.kind,
            "lane": str(self.payload.get("lane") or "obligations").strip(),
            "lane_family_key": str(self.payload.get("lane_family_key") or "").strip() or None,
            "why_surfaced": list(self.why_surfaced),
            "source_bundle_keys": list(self.source_bundle_keys),
        }


@dataclass(slots=True)
class ConflictEdge:
    edge_key: str
    source_node_key: str
    target_node_key: str
    relation: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ConflictEdge":
        row_payload = dict(payload or {})
        return cls(
            edge_key=str(row_payload.get("edge_key") or "").strip(),
            source_node_key=str(row_payload.get("source_node_key") or "").strip(),
            target_node_key=str(row_payload.get("target_node_key") or "").strip(),
            relation=str(row_payload.get("relation") or "").strip(),
            payload=row_payload,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = dict(self.payload)
        payload["edge_key"] = self.edge_key
        payload["source_node_key"] = self.source_node_key
        payload["target_node_key"] = self.target_node_key
        payload["relation"] = self.relation
        return payload


@dataclass(slots=True)
class ReviewReport:
    session_id: str
    generated_at: str | None
    session_memory_profile: str | None
    session_memory_profile_override: str | None
    session_narrative_mode: str | None
    session_narrative_mode_override: str | None
    session_narrative_mode_source: str | None
    memory_policy_state: dict[str, Any]
    economy_state: dict[str, Any]
    story_obligations: list[ObligationNode]
    memory_conflict_edges: list[ConflictEdge]
    narrative_graph_edges: list[dict[str, Any]]
    narrative_chains: list[dict[str, Any]]
    actor_memory_views: list[dict[str, Any]]
    tuning_report: dict[str, Any]
    saturation_diagnostics: dict[str, Any]
    trace_corpus: dict[str, Any]
    benchmark_report: dict[str, Any]
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ReviewReport":
        report_payload = dict(payload or {})
        return cls(
            session_id=str(report_payload.get("session_id") or "").strip(),
            generated_at=report_payload.get("generated_at"),
            session_memory_profile=report_payload.get("session_memory_profile"),
            session_memory_profile_override=report_payload.get("session_memory_profile_override"),
            session_narrative_mode=report_payload.get("session_narrative_mode"),
            session_narrative_mode_override=report_payload.get("session_narrative_mode_override"),
            session_narrative_mode_source=report_payload.get("session_narrative_mode_source"),
            memory_policy_state=dict(report_payload.get("memory_policy_state") or {}),
            economy_state=dict(report_payload.get("economy_state") or {}),
            story_obligations=[
                ObligationNode.from_payload(item)
                for item in list(report_payload.get("story_obligations") or [])
                if isinstance(item, dict)
            ],
            memory_conflict_edges=[
                ConflictEdge.from_payload(item)
                for item in list(report_payload.get("memory_conflict_edges") or [])
                if isinstance(item, dict)
            ],
            narrative_graph_edges=[
                dict(item)
                for item in list(report_payload.get("narrative_graph_edges") or [])
                if isinstance(item, dict)
            ],
            narrative_chains=[
                dict(item)
                for item in list(report_payload.get("narrative_chains") or [])
                if isinstance(item, dict)
            ],
            actor_memory_views=[
                dict(item)
                for item in list(report_payload.get("actor_memory_views") or [])
                if isinstance(item, dict)
            ],
            tuning_report=dict(report_payload.get("tuning_report") or {}),
            saturation_diagnostics=dict(report_payload.get("saturation_diagnostics") or {}),
            trace_corpus=dict(report_payload.get("trace_corpus") or {}),
            benchmark_report=dict(report_payload.get("benchmark_report") or {}),
            payload=report_payload,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = dict(self.payload)
        payload["session_id"] = self.session_id
        payload["generated_at"] = self.generated_at
        payload["session_memory_profile"] = self.session_memory_profile
        payload["session_memory_profile_override"] = self.session_memory_profile_override
        payload["session_narrative_mode"] = self.session_narrative_mode
        payload["session_narrative_mode_override"] = self.session_narrative_mode_override
        payload["session_narrative_mode_source"] = self.session_narrative_mode_source
        payload["memory_policy_state"] = dict(self.memory_policy_state)
        payload["economy_state"] = dict(self.economy_state)
        payload["story_obligations"] = [item.to_payload() for item in self.story_obligations]
        payload["memory_conflict_edges"] = [item.to_payload() for item in self.memory_conflict_edges]
        payload["narrative_graph_edges"] = [dict(item) for item in self.narrative_graph_edges]
        payload["narrative_chains"] = [dict(item) for item in self.narrative_chains]
        payload["actor_memory_views"] = [dict(item) for item in self.actor_memory_views]
        payload["tuning_report"] = dict(self.tuning_report)
        payload["saturation_diagnostics"] = dict(self.saturation_diagnostics)
        payload["trace_corpus"] = dict(self.trace_corpus)
        payload["benchmark_report"] = dict(self.benchmark_report)
        return payload


@dataclass(slots=True)
class TurnTraceSummary:
    turn_index: int
    status: str | None
    turn_intent: str | None
    scene_mode: str | None
    narration_summary: str
    surfaced_relevant_count: int
    surfaced_callback_count: int
    surfaced_bundle_count: int
    transition_ambiguity_count: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TurnTraceSummary":
        row_payload = dict(payload or {})
        return cls(
            turn_index=_safe_int(row_payload.get("turn_index")),
            status=str(row_payload.get("status") or "").strip() or None,
            turn_intent=str(row_payload.get("turn_intent") or "").strip() or None,
            scene_mode=str(row_payload.get("scene_mode") or "").strip() or None,
            narration_summary=str(row_payload.get("narration_summary") or ""),
            surfaced_relevant_count=_safe_int(row_payload.get("surfaced_relevant_count")),
            surfaced_callback_count=_safe_int(row_payload.get("surfaced_callback_count")),
            surfaced_bundle_count=_safe_int(row_payload.get("surfaced_bundle_count")),
            transition_ambiguity_count=_safe_int(row_payload.get("transition_ambiguity_count")),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "status": self.status,
            "turn_intent": self.turn_intent,
            "scene_mode": self.scene_mode,
            "narration_summary": self.narration_summary,
            "surfaced_relevant_count": self.surfaced_relevant_count,
            "surfaced_callback_count": self.surfaced_callback_count,
            "surfaced_bundle_count": self.surfaced_bundle_count,
            "transition_ambiguity_count": self.transition_ambiguity_count,
        }


@dataclass(slots=True)
class AuditSnapshot:
    session_id: str
    generated_at: str | None
    session_memory_profile: str | None
    session_memory_profile_override: str | None
    session_narrative_mode: str | None
    session_narrative_mode_override: str | None
    session_narrative_mode_source: str | None
    memory_policy_state: dict[str, Any]
    economy_state: dict[str, Any]
    session_memory_health_score: float
    finding_counts: dict[str, Any]
    feedback_summary: dict[str, Any]
    operational_alerts: list[dict[str, Any]]
    operational_guardrails: dict[str, Any]
    story_obligations: list[ObligationNode]
    memory_conflict_edges: list[ConflictEdge]
    narrative_graph_edges: list[dict[str, Any]]
    narrative_chains: list[dict[str, Any]]
    actor_memory_views: list[dict[str, Any]]
    tuning_report: dict[str, Any]
    saturation_diagnostics: dict[str, Any]
    trace_corpus: dict[str, Any]
    benchmark_report: dict[str, Any]
    stable_report: dict[str, Any]

    @classmethod
    def from_report(cls, report: ReviewReport) -> "AuditSnapshot":
        payload = report.payload
        return cls(
            session_id=report.session_id,
            generated_at=report.generated_at,
            session_memory_profile=report.session_memory_profile,
            session_memory_profile_override=report.session_memory_profile_override,
            session_narrative_mode=report.session_narrative_mode,
            session_narrative_mode_override=report.session_narrative_mode_override,
            session_narrative_mode_source=report.session_narrative_mode_source,
            memory_policy_state=dict(report.memory_policy_state),
            economy_state=dict(report.economy_state),
            session_memory_health_score=_safe_float(payload.get("session_memory_health_score")),
            finding_counts=dict(payload.get("finding_counts") or {}),
            feedback_summary=dict(payload.get("feedback_summary") or {}),
            operational_alerts=[
                dict(item)
                for item in list(payload.get("operational_alerts") or [])
                if isinstance(item, dict)
            ],
            operational_guardrails=dict(payload.get("operational_guardrails") or {}),
            story_obligations=[ObligationNode.from_payload(item.to_payload()) for item in report.story_obligations],
            memory_conflict_edges=[ConflictEdge.from_payload(item.to_payload()) for item in report.memory_conflict_edges],
            narrative_graph_edges=[dict(item) for item in report.narrative_graph_edges],
            narrative_chains=[dict(item) for item in report.narrative_chains],
            actor_memory_views=[dict(item) for item in report.actor_memory_views],
            tuning_report=dict(report.tuning_report),
            saturation_diagnostics=dict(report.saturation_diagnostics),
            trace_corpus=dict(report.trace_corpus),
            benchmark_report=dict(report.benchmark_report),
            stable_report=dict(payload.get("stable_report") or {}),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "generated_at": self.generated_at,
            "session_memory_profile": self.session_memory_profile,
            "session_memory_profile_override": self.session_memory_profile_override,
            "session_narrative_mode": self.session_narrative_mode,
            "session_narrative_mode_override": self.session_narrative_mode_override,
            "session_narrative_mode_source": self.session_narrative_mode_source,
            "memory_policy_state": dict(self.memory_policy_state),
            "economy_state": dict(self.economy_state),
            "session_memory_health_score": self.session_memory_health_score,
            "finding_counts": dict(self.finding_counts),
            "feedback_summary": dict(self.feedback_summary),
            "operational_alerts": [dict(item) for item in self.operational_alerts],
            "operational_guardrails": dict(self.operational_guardrails),
            "story_obligations": [item.to_payload() for item in self.story_obligations],
            "memory_conflict_edges": [item.to_payload() for item in self.memory_conflict_edges],
            "narrative_graph_edges": [dict(item) for item in self.narrative_graph_edges],
            "narrative_chains": [dict(item) for item in self.narrative_chains],
            "actor_memory_views": [dict(item) for item in self.actor_memory_views],
            "tuning_report": dict(self.tuning_report),
            "saturation_diagnostics": dict(self.saturation_diagnostics),
            "trace_corpus": dict(self.trace_corpus),
            "benchmark_report": dict(self.benchmark_report),
            "stable_report": dict(self.stable_report),
        }
