from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy.orm import Session

from .. import background_workers, models
from ..db import MEMORY_REVIEW_INTERVAL_SECONDS, SessionLocal
from ..llm_telemetry import telemetry_context
from ..observability import (
    record_memory_benchmark_result,
    record_memory_review_findings,
    record_memory_review_run,
    reset_trace_id,
    set_trace_id,
)
from ..persistence.memory_repository import memory_projection_repository
from ..application.dtos import AuditSnapshot, ReviewReport
from .economy_services import economy_service
from .. import memory_evaluation as evaluation_ops
from .. import memory_review as review_ops

logger = logging.getLogger(__name__)


class MemoryReviewService:
    def build_review_report(self, db: Session, *, session_id: uuid.UUID) -> ReviewReport:
        session_row = db.get(models.SessionModel, session_id)
        session_state = dict(getattr(session_row, "state_json", {}) or {})
        turn_rows = memory_projection_repository.list_turn_rows_for_review(db, session_id=session_id)
        fact_rows = memory_projection_repository.list_fact_payloads(db, session_id=session_id)
        bundle_packets = memory_projection_repository.list_bundle_packets(db, session_id=session_id)
        bundle_rows = [packet.to_payload() for packet in bundle_packets]
        event_rows = memory_projection_repository.list_event_payloads(db, session_id=session_id)
        feedback_maps = review_ops._review_feedback_maps(
            db,
            session_id=session_id,
            turn_rows=turn_rows,
            fact_rows=fact_rows,
            bundle_rows=bundle_rows,
            event_rows=event_rows,
        )
        current_turn = max((int(row.get("turn_index") or 0) for row in turn_rows), default=0)
        base_conflict_edges = review_ops.derive_conflict_edge_payloads(
            fact_rows=fact_rows,
            event_rows=event_rows,
            bundle_rows=bundle_rows,
            current_turn=current_turn,
        )
        obligations = review_ops.derive_story_obligation_payloads(
            fact_rows=fact_rows,
            event_rows=event_rows,
            bundle_rows=bundle_rows,
            conflict_edges=base_conflict_edges,
            current_turn=current_turn,
        )
        conflict_edges = review_ops.derive_conflict_edge_payloads(
            fact_rows=fact_rows,
            event_rows=event_rows,
            bundle_rows=bundle_rows,
            obligation_rows=obligations,
            current_turn=current_turn,
        )
        narrative_graph_edges = review_ops.derive_narrative_graph_edges(
            conflict_edges=conflict_edges,
            bundle_rows=bundle_rows,
            obligation_rows=obligations,
        )
        narrative_chains = review_ops.derive_narrative_chains(
            narrative_graph_edges=narrative_graph_edges,
            obligation_rows=obligations,
            bundle_rows=bundle_rows,
        )
        obligations = review_ops.annotate_rows_with_narrative_chains(
            obligations,
            narrative_chains=narrative_chains,
        )
        initial_policy_state = review_ops.derive_memory_policy_state(
            turn_rows=turn_rows,
            session_state=session_state,
            obligation_rows=obligations,
        )
        actor_memory_views = review_ops.derive_actor_memory_views(
            fact_rows=fact_rows,
            event_rows=event_rows,
            bundle_rows=bundle_rows,
            obligation_rows=obligations,
            player_object_id=session_state.get("player_object_id"),
        )
        actor_memory_views = review_ops.annotate_rows_with_narrative_chains(
            actor_memory_views,
            narrative_chains=narrative_chains,
        )
        feedback_maps = review_ops._apply_structured_repair_feedback(
            feedback_maps,
            turn_rows=turn_rows,
            fact_rows=fact_rows,
            event_rows=event_rows,
            obligation_rows=obligations,
            actor_view_rows=actor_memory_views,
        )
        fact_rows, bundle_rows, event_rows = review_ops._apply_quality_feedback_to_rows(
            db,
            session_id=session_id,
            fact_rows=fact_rows,
            bundle_rows=bundle_rows,
            turn_rows=turn_rows,
            event_rows=event_rows,
            feedback_maps=feedback_maps,
        )
        obligations, actor_memory_views = review_ops._apply_feedback_to_derived_rows(
            obligation_rows=obligations,
            actor_view_rows=actor_memory_views,
            feedback_maps=feedback_maps,
        )
        feedback_summary = review_ops._build_feedback_summary(
            fact_rows=fact_rows,
            event_rows=event_rows,
            bundle_rows=bundle_rows,
            obligation_rows=obligations,
            actor_view_rows=actor_memory_views,
        )
        trace_corpus = review_ops.build_trace_backed_memory_corpus(turn_rows)
        trace_corpus["policy_version"] = str(
            session_state.get("memory_tuning_policy_version")
            or trace_corpus.get("policy_version")
            or review_ops.DEFAULT_TUNING_ROLLOUT_POLICY.policy_version
        )
        trace_corpus["reviewed_through_turn"] = current_turn
        trace_corpus["feedback_summary"] = dict(feedback_summary)
        trace_corpus["frozen_evaluation_targets"] = evaluation_ops.build_frozen_evaluation_targets(
            fact_rows=fact_rows,
            obligation_rows=obligations,
            current_turn=current_turn,
        )
        benchmark_report = evaluation_ops.evaluate_memory_benchmarks_for_session(
            db,
            session_id=session_id,
            trace_corpus=trace_corpus,
        )
        active_fact_rows = [row for row in fact_rows if review_ops.normalize_state(row.get("state")) == "active"]
        analysis = review_ops.analyze_memory_review_rows(
            fact_rows=active_fact_rows,
            bundle_rows=bundle_rows,
            benchmark_report=benchmark_report,
            event_rows=event_rows,
            obligation_rows=obligations,
            conflict_rows=conflict_edges,
            narrative_chains=narrative_chains,
            actor_view_rows=actor_memory_views,
            trace_corpus=trace_corpus,
            turn_rows=turn_rows,
            current_turn=current_turn,
        )
        tuning_report = review_ops.build_tuning_recommendations(
            trace_corpus=trace_corpus,
            finding_counts=dict(analysis.get("finding_counts") or {}),
            active_weights=dict(session_state.get("memory_tuning_active_weights") or {}),
            active_lane_weights=dict(session_state.get("memory_tuning_active_lane_weights") or {}),
            active_family_weights=dict(session_state.get("memory_tuning_active_family_weights") or {}),
            active_policy_version=str(session_state.get("memory_tuning_policy_version") or ""),
        )
        saturation_diagnostics = review_ops.review_saturation_diagnostics(trace_corpus)
        memory_policy_state = review_ops.derive_memory_policy_state(
            turn_rows=turn_rows,
            session_state=session_state,
            session_memory_profile=initial_policy_state.session_memory_profile,
            session_narrative_mode=initial_policy_state.session_narrative_mode,
            current_memory_health=analysis.get("session_memory_health_score"),
            obligation_rows=obligations,
            tuning_report=tuning_report,
        )
        policy_state_payload = review_ops.memory_policy_state_payload(memory_policy_state)
        operational_alerts = review_ops.derive_session_operational_alerts(
            finding_counts=dict(analysis.get("finding_counts") or {}),
            findings=dict(analysis.get("findings") or {}),
            saturation_diagnostics=saturation_diagnostics,
            actor_view_rows=actor_memory_views,
            tuning_report=tuning_report,
            memory_policy_state=memory_policy_state,
        )
        operational_guardrails = review_ops.derive_operational_alert_guardrails(
            alerts=operational_alerts,
            lane_budgets=memory_policy_state.lane_budgets,
            saturation_cap_values=memory_policy_state.saturation_limits,
        )
        tuning_report["promotion_guardrail"] = dict(
            operational_guardrails.get("promotion_guardrail") or {}
        )
        trace_corpus["policy_version"] = str(tuning_report.get("policy_version") or trace_corpus.get("policy_version") or "")
        trace_corpus["memory_policy_state"] = policy_state_payload
        trace_corpus["operational_guardrails"] = dict(operational_guardrails)
        stable_trace_corpus = review_ops.stable_artifact_payload(
            {key: value for key, value in trace_corpus.items() if key != "stable_digest"}
        )
        trace_corpus["stable_digest"] = review_ops.stable_mapping_digest(stable_trace_corpus)
        benchmark_report["policy_version"] = str(tuning_report.get("policy_version") or benchmark_report.get("policy_version") or "")
        benchmark_report["memory_policy_state"] = policy_state_payload
        benchmark_report["frozen_evaluation_targets"] = dict(trace_corpus.get("frozen_evaluation_targets") or {})
        economy_state = economy_service.build_review_economy_state(
            db,
            session_id=session_id,
            current_turn=current_turn,
            fact_rows=active_fact_rows,
            obligation_rows=obligations,
            turn_rows=turn_rows,
        )
        report_payload = {
            "session_id": str(session_id),
            "generated_at": review_ops.datetime.now(review_ops.timezone.utc).isoformat(),
            "reviewed_through_turn": current_turn,
            "session_memory_profile": memory_policy_state.session_memory_profile,
            "session_memory_profile_override": memory_policy_state.session_memory_profile_override,
            "session_narrative_mode": memory_policy_state.session_narrative_mode,
            "session_narrative_mode_override": memory_policy_state.session_narrative_mode_override,
            "session_narrative_mode_source": memory_policy_state.session_narrative_mode_source,
            "memory_policy_state": policy_state_payload,
            "economy_state": economy_state,
            "trace_corpus": trace_corpus,
            "story_obligations": obligations,
            "memory_conflict_edges": conflict_edges,
            "narrative_graph_edges": narrative_graph_edges,
            "narrative_chains": narrative_chains,
            "actor_memory_views": actor_memory_views,
            "tuning_report": tuning_report,
            "saturation_diagnostics": saturation_diagnostics,
            "operational_alerts": operational_alerts,
            "operational_guardrails": operational_guardrails,
            "feedback_summary": feedback_summary,
            **analysis,
        }
        stable_report = review_ops.stable_artifact_payload(report_payload)
        stable_report["stable_digest"] = review_ops.stable_mapping_digest(stable_report)
        report_payload["stable_report"] = stable_report
        return ReviewReport.from_payload(report_payload)

    def run_review_once(self, *, session_id: uuid.UUID | None = None) -> None:
        db = SessionLocal()
        try:
            with db.begin():
                session_ids = [session_id] if session_id is not None else memory_projection_repository.list_session_ids(db)
                for current_session_id in session_ids:
                    report = self.build_review_report(db, session_id=current_session_id)
                    memory_projection_repository.replace_projection_rows(
                        db,
                        session_id=current_session_id,
                        object_type=review_ops.STORY_OBLIGATION_OBJECT_TYPE,
                        key_name="obligation_key",
                        name_prefix="story_obligation",
                        payloads=[item.to_payload() for item in report.story_obligations],
                    )
                    memory_projection_repository.replace_projection_rows(
                        db,
                        session_id=current_session_id,
                        object_type=review_ops.MEMORY_CONFLICT_EDGE_OBJECT_TYPE,
                        key_name="edge_key",
                        name_prefix="memory_conflict_edge",
                        payloads=[item.to_payload() for item in report.memory_conflict_edges],
                    )
                    memory_projection_repository.write_review_report(
                        db,
                        session_id=current_session_id,
                        report=report,
                    )
                    memory_projection_repository.write_evaluation_report(
                        db,
                        session_id=current_session_id,
                        report=dict(report.benchmark_report),
                    )
                    memory_projection_repository.update_session_memory_profile(
                        db,
                        session_id=current_session_id,
                        profile=report.session_memory_profile,
                    )
                    record_memory_review_run()
                    for benchmark in list(report.benchmark_report.get("benchmarks") or []):
                        if not isinstance(benchmark, dict):
                            continue
                        record_memory_benchmark_result(
                            benchmark=str(benchmark.get("benchmark") or "unknown"),
                            score=review_ops._safe_float(benchmark.get("score")),
                        )
                    for family_report in list(report.benchmark_report.get("family_reports") or []):
                        if not isinstance(family_report, dict):
                            continue
                        corpus_family = str(family_report.get("corpus_family") or "unknown")
                        for benchmark in list(family_report.get("benchmarks") or []):
                            if not isinstance(benchmark, dict):
                                continue
                            record_memory_benchmark_result(
                                benchmark=f"{corpus_family}:{str(benchmark.get('benchmark') or 'unknown')}",
                                score=review_ops._safe_float(benchmark.get("score")),
                            )
                    for finding_kind, finding_count in dict(report.payload.get("finding_counts") or {}).items():
                        record_memory_review_findings(str(finding_kind), review_ops._safe_int(finding_count))
        finally:
            db.close()

    def run_review_loop(self) -> None:
        sleep_seconds = max(float(MEMORY_REVIEW_INTERVAL_SECONDS), 300.0)
        while not background_workers.shutdown_requested():
            worker_trace = uuid.uuid4().hex[:8]
            trace_token = set_trace_id(worker_trace)
            try:
                with telemetry_context(request_type="maintenance:memory_review"):
                    self.run_review_once()
            except Exception:  # noqa: BLE001
                logger.exception("Memory review iteration failed")
            finally:
                reset_trace_id(trace_token)
            background_workers.wait_for_shutdown(sleep_seconds)


class MemoryEvaluationService:
    def _current_payload(self, db: Session, *, session_id: uuid.UUID) -> dict[str, Any]:
        payload = memory_projection_repository.read_evaluation_report(db, session_id=session_id)
        if payload:
            return payload
        review_report = memory_audit_service._current_review_report(db, session_id=session_id)
        return dict(review_report.benchmark_report)

    def get_evaluation_report(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
        corpus_family: str | None = None,
        regression_pack: str | None = None,
    ) -> dict[str, Any]:
        if corpus_family and regression_pack:
            raise ValueError("ambiguous_evaluation_selector")
        payload = self._current_payload(db, session_id=session_id)
        if corpus_family:
            try:
                selected_family = review_ops.normalize_frozen_trace_corpus_family(corpus_family)
            except ValueError as exc:
                raise ValueError("unsupported_corpus_family") from exc
            for family_report in list(payload.get("family_reports") or []):
                if isinstance(family_report, dict) and str(family_report.get("corpus_family") or "").strip() == selected_family:
                    return review_ops._evaluation_report_response(
                        session_id=session_id,
                        payload=payload,
                        selected_report=dict(family_report),
                    )
            raise ValueError("missing_corpus_family_report")
        if regression_pack:
            try:
                selected_pack = review_ops.normalize_trace_regression_pack_name(regression_pack)
            except ValueError as exc:
                raise ValueError("unsupported_regression_pack") from exc
            for regression_report in list(payload.get("regression_packs") or []):
                if isinstance(regression_report, dict) and str(regression_report.get("regression_pack") or "").strip() == selected_pack:
                    return review_ops._evaluation_report_response(
                        session_id=session_id,
                        payload=payload,
                        selected_report=dict(regression_report),
                    )
            raise ValueError("missing_regression_pack_report")
        return review_ops._evaluation_report_response(session_id=session_id, payload=payload)


class MemoryAuditService:
    def _current_review_report(self, db: Session, *, session_id: uuid.UUID) -> ReviewReport:
        report = memory_projection_repository.read_review_report(db, session_id=session_id)
        if report is not None:
            return report
        return memory_review_service.build_review_report(db, session_id=session_id)

    def get_health(self, db: Session, *, session_id: uuid.UUID) -> dict[str, Any]:
        report = self._current_review_report(db, session_id=session_id)
        payload = report.payload
        return {
            "session_id": report.session_id,
            "session_memory_profile": report.session_memory_profile,
            "session_memory_profile_override": report.session_memory_profile_override,
            "session_narrative_mode": report.session_narrative_mode,
            "session_narrative_mode_override": report.session_narrative_mode_override,
            "session_narrative_mode_source": report.session_narrative_mode_source,
            "memory_policy_state": dict(report.memory_policy_state),
            "economy_summary": dict(report.economy_state.get("summary") or {}),
            "economy_world_knobs": dict(report.economy_state.get("world_knobs") or {}),
            "economy_pressures": list(report.economy_state.get("pressures") or []),
            "economy_brief": dict(report.economy_state.get("brief") or {}),
            "economy_continuity_summary": dict(report.economy_state.get("continuity_summary") or {}),
            "economy_account_views": list(report.economy_state.get("account_views") or []),
            "economy_memory_bridges": list(report.economy_state.get("memory_bridges") or []),
            "session_memory_health_score": payload.get("session_memory_health_score"),
            "finding_counts": dict(payload.get("finding_counts") or {}),
            "feedback_summary": dict(payload.get("feedback_summary") or {}),
            "top_issue_samples": dict(payload.get("top_issue_samples") or {}),
            "top_obligation_clusters": list(payload.get("top_obligation_clusters") or []),
            "top_conflict_clusters": list(payload.get("top_conflict_clusters") or []),
            "low_quality_recall_families": list(payload.get("low_quality_recall_families") or []),
            "actor_view_hotspots": list(payload.get("actor_view_hotspots") or []),
            "operational_alerts": list(payload.get("operational_alerts") or []),
            "operational_guardrails": dict(payload.get("operational_guardrails") or {}),
            "saturation_diagnostics": dict(report.saturation_diagnostics),
            "tuning_report": dict(report.tuning_report),
            "generated_at": report.generated_at,
        }

    def get_snapshot(self, db: Session, *, session_id: uuid.UUID) -> dict[str, Any]:
        snapshot = AuditSnapshot.from_report(self._current_review_report(db, session_id=session_id))
        return snapshot.to_payload()

    def list_story_obligations(self, db: Session, *, session_id: uuid.UUID) -> list[dict[str, Any]]:
        return [item.to_payload() for item in memory_projection_repository.list_story_obligations(db, session_id=session_id)]

    def list_conflict_edges(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
        node_key: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = [item.to_payload() for item in memory_projection_repository.list_conflict_edges(db, session_id=session_id)]
        normalized_node_key = str(node_key or "").strip()
        if not normalized_node_key:
            return rows
        return [
            row
            for row in rows
            if str(row.get("source_node_key") or "").strip() == normalized_node_key
            or str(row.get("target_node_key") or "").strip() == normalized_node_key
        ]

    def list_narrative_graph_edges(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
        node_key: str | None = None,
        node_type: str | None = None,
    ) -> list[dict[str, Any]]:
        report = self._current_review_report(db, session_id=session_id)
        rows = [dict(item) for item in report.narrative_graph_edges]
        normalized_node_key = str(node_key or "").strip()
        normalized_node_type = str(node_type or "").strip().lower()
        if not normalized_node_key:
            return rows
        return [
            row
            for row in rows
            if (
                str(row.get("source_node_key") or "").strip() == normalized_node_key
                and (not normalized_node_type or str(row.get("source_node_type") or "").strip().lower() == normalized_node_type)
            )
            or (
                str(row.get("target_node_key") or "").strip() == normalized_node_key
                and (not normalized_node_type or str(row.get("target_node_type") or "").strip().lower() == normalized_node_type)
            )
        ]

    def get_graph_neighborhood(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
        node_key: str,
        node_type: str | None = None,
    ) -> dict[str, Any]:
        report = self._current_review_report(db, session_id=session_id)
        return review_ops.graph_neighborhood(
            node_key=node_key,
            node_type=node_type,
            narrative_graph_edges=[dict(item) for item in report.narrative_graph_edges],
        )

    def list_actor_memory_views(self, db: Session, *, session_id: uuid.UUID) -> list[dict[str, Any]]:
        report = self._current_review_report(db, session_id=session_id)
        return [dict(item) for item in report.actor_memory_views]

    def list_narrative_chains(self, db: Session, *, session_id: uuid.UUID) -> list[dict[str, Any]]:
        report = self._current_review_report(db, session_id=session_id)
        return [dict(item) for item in report.narrative_chains]

    def get_tuning_report(self, db: Session, *, session_id: uuid.UUID) -> dict[str, Any]:
        report = self._current_review_report(db, session_id=session_id)
        return {
            "session_id": report.session_id,
            "generated_at": report.generated_at,
            "memory_policy_state": dict(report.memory_policy_state),
            "tuning_report": dict(report.tuning_report),
        }

    def get_turn_memory_trace(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
        turn_index: int,
    ) -> dict[str, Any]:
        turn_row = db.get(models.TurnModel, (session_id, int(turn_index)))
        ai_json = dict(getattr(turn_row, "ai_json", {}) or {}) if turn_row is not None else {}
        memory_debug = dict(ai_json.get("memory_debug") or {})
        return {
            "session_id": str(session_id),
            "turn_index": int(turn_index),
            "memory_trace": dict(ai_json.get("memory_trace") or {}),
            "memory_debug": memory_debug,
            "turn_inspector": review_ops.build_turn_memory_inspector(memory_debug),
        }

    def list_turn_trace_index(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
        limit: int,
    ) -> list[dict[str, Any]]:
        return [
            item.to_payload()
            for item in memory_projection_repository.list_turn_trace_index(
                db,
                session_id=session_id,
                limit=limit,
            )
        ]

    def list_memory_bundle_graph(self, db: Session, *, session_id: uuid.UUID) -> list[dict[str, Any]]:
        return [item.to_payload() for item in memory_projection_repository.list_bundle_packets(db, session_id=session_id)]

    def set_session_memory_profile_override(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
        profile: str | None,
    ) -> dict[str, Any]:
        return review_ops.set_session_memory_profile_override(db, session_id=session_id, profile=profile)

    def set_session_narrative_mode_override(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
        narrative_mode: str | None,
    ) -> dict[str, Any]:
        return review_ops.set_session_narrative_mode_override(
            db,
            session_id=session_id,
            narrative_mode=narrative_mode,
        )

    def promote_session_tuning_candidate(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
    ) -> dict[str, Any]:
        return review_ops.promote_session_tuning_candidate(db, session_id=session_id)


memory_review_service = MemoryReviewService()
memory_evaluation_service = MemoryEvaluationService()
memory_audit_service = MemoryAuditService()
