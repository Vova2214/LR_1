# Architecture Phases

## Module map

- `src/routes/`
  - `sessions.py`: session lifecycle, lore upload, apply turn, snapshots, pending graph ops, timeline/diff, relationship graph.
  - `turns.py`: `PATCH /sessions/{session_id}/turns/{turn_index}`.
  - `internal.py`: telemetry, outbox, memory audit/review/evaluation endpoints and audit UI.
  - `objects.py`, `links.py`, `events.py`, `semantic_tools.py`, `debug.py`: thin transport for entity, semantic, and debug commands.
- `src/application/`
  - `session_services.py`: route-facing application services for session, telemetry, entity, semantic, and debug entrypoints.
  - `turn_services.py`: `TurnApplicationService`, `TurnPlanningService`, `TurnContextService`, `ContinuityService`, `MemoryWriteService`.
  - `memory_services.py`: `MemoryReviewService`, `MemoryEvaluationService`, `MemoryAuditService`.
  - `turn_contracts.py`: internal turn-plan and external-preparation contracts shared between application services and compatibility shims.
  - `dtos.py`: internal typed DTOs for staged turn execution and memory/review/audit boundaries.
  - `tooling_config.py`: typed config surface for thin audit/trace tooling transport limits.
- `src/domain/`
  - `memory_policy.py`: public domain import boundary for core memory policy and tuning policy.
  - `continuity_policy.py`: pure continuity helpers for scene-mode inference, anchor derivation, token/item trimming, and salient-item classification.
  - `memory_candidates.py`: durable-fact and memory-candidate identity, merge, canonicalization, supplementation, and deterministic coalescing.
- `src/persistence/`
  - `memory_repository.py`: memory review/audit projections, row loading, projection replacement, report persistence, trace index reads.
  - `session_read_repository.py`: session timeline, diff, relationship-graph read-side queries and row shaping.
  - `graph_repository.py`: pending graph-op state projection, zone/event graph queries, asymmetric relationship reads.
  - `memory_write_repository.py`: memory fact/event lookup and upsert helpers for live turn persistence.
- `src/workers/`
  - `registry.py`: startup registry for worker loop launch only.
- `src/architecture_contracts.py`
  - Typed package-boundary and compatibility-adapter contract registry.
- `src/crud_*`
  - `crud.py` is the permanent public facade.
  - `crud_core.py`, `crud_graph_ops.py`, `crud_embeddings_ops.py`, `crud_shared.py`, and `crud_continuity.py` keep explicitly documented deprecated shim surfaces only where callers still rely on them.
  - `crud_turns_logic.py`, `crud_context.py`, and `memory_review.py` still carry transitional wrappers listed below as deletion candidates.
- `src/models.py`
  - Persistence model only.
- `src/schemas.py`
  - External API schemas and shared typed LLM contracts.

## Use-case map

- Create session
  - Route: `src/routes/sessions.py:create_session`
  - Application: `src/application/session_services.py:SessionLifecycleService.create_session`
  - Persistence/runtime: `src/crud_entities.py:create_session_with_defaults`
- Read session
  - Route: `src/routes/sessions.py:get_session`
  - Application: `src/application/session_services.py:SessionLifecycleService.get_session`
  - Persistence/runtime: `src/crud_entities.py:get_session`
- Apply turn
  - Route: `src/routes/sessions.py:run_turn`
  - Application: `src/application/turn_services.py:TurnApplicationService.run_turn`
  - Stages:
    - allocation: `TurnApplicationService._allocation_stage`
    - resolve prompts/plan: `TurnPlanningService.build_turn_plan_outside_apply_tx`
    - validate plan boundary: `TurnApplicationService._validate_plan_stage`
    - apply world patch: `TurnApplicationService._apply_world_patch_stage`
    - persist memory: `TurnApplicationService._persist_memory_stage`
    - build observability/debug payload: `TurnApplicationService._build_observability_stage`
    - finalize turn: `TurnApplicationService.apply_turn_plan` and `TurnApplicationService.run_turn_locked`
  - Persistence/runtime: `src/crud_patch_apply.py`, `src/crud_embeddings_ops.py`, `src/crud_continuity.py`, `src/outbox_runtime.py`
- Build turn context
  - Application: `src/application/turn_services.py:TurnContextService.build_turn_context_pack`
  - Query/runtime helpers: `src/crud_context.py` helper functions plus `ContinuityService.build_memory_context_blocks`
  - Domain/persistence boundaries:
    - scene mode and trimming policy: `src/domain/continuity_policy.py`
    - retrieved memory/review projections: `src/persistence/memory_repository.py`
- Run review
  - Worker/runtime entrypoints:
    - worker registry: `src/workers/registry.py:startup_worker_specs`
    - review loop: `src/application/memory_services.py:MemoryReviewService.run_review_loop`
    - one-shot review: `src/application/memory_services.py:MemoryReviewService.run_review_once`
  - Persistence/runtime: `src/persistence/memory_repository.py`
- Run evaluation
  - Internal route: `src/routes/internal.py:get_memory_audit_evaluation_endpoint`
  - Application: `src/application/memory_services.py:MemoryEvaluationService.get_evaluation_report`
  - Runtime benchmark builder: `src/memory_evaluation.py:evaluate_memory_benchmarks_for_session`
- Internal audit
  - Internal routes: `src/routes/internal.py:get_memory_audit_*`
  - Application: `src/application/memory_services.py:MemoryAuditService.*`
  - Persistence/runtime: `src/persistence/memory_repository.py`, stored review/evaluation projection rows in `ObjectModel`
  - Thin tooling consumer: `src/routes/internal.py:get_memory_audit_ui_endpoint` renders dashboard, evaluation-family, and trace-viewer panels backed by JSON endpoints only.

## Current ownership

- Policy ownership
  - `src/domain/memory_policy.py`: public domain boundary for memory review/tuning policy and scoring policy.
  - `src/memory_policy.py`: underlying implementation module until a later low-risk file move.
  - `src/domain/continuity_policy.py`: pure continuity policy and trimming invariants.
  - `src/domain/memory_candidates.py`: pure durable-fact and memory-candidate identity/merge invariants.
- Application service ownership
  - `src/application/session_services.py`: route-facing session, telemetry, entity, semantic, and debug orchestration.
  - `src/application/turn_services.py`: turn planning, context build, continuity coordination, apply-turn staging, memory persistence orchestration.
  - `src/application/memory_services.py`: review, evaluation, audit, and review-loop orchestration.
  - `src/application/turn_contracts.py`: application-owned turn plan and external-preparation contracts.
- Persistence ownership
  - `src/persistence/session_read_repository.py`: session timeline/diff/relationship graph reads.
  - `src/persistence/graph_repository.py`: pending graph-op projection reads and writes.
  - `src/persistence/memory_write_repository.py`: memory fact/event write-side helpers.
  - `src/persistence/memory_repository.py`: review/audit/evaluation projection reads and writes.
- Worker startup ownership
  - `src/workers/registry.py`: the only worker registration surface used by `src/main.py`.

## Compatibility adapter status

- Permanent public facade
  - `src/crud.py`: stable public import surface and monkeypatch seam. New logic should not be added here.
- Temporary deprecated shims
  - `src/crud_core.py`: `get_session_timeline`, `get_session_diff`, `get_relationship_graph` forward to `src/persistence/session_read_repository.py`.
  - `src/crud_graph_ops.py`: selected graph query/state helpers forward to `src/persistence/graph_repository.py`.
  - `src/crud_embeddings_ops.py`: selected memory row lookup helpers forward to `src/persistence/memory_write_repository.py`.
  - `src/crud_shared.py`: pure durable-fact and memory-candidate helpers forward to `src/domain/memory_candidates.py`.
  - `src/crud_continuity.py`: pure continuity policy helpers forward to `src/domain/continuity_policy.py`.
- Removable now
  - None. Remaining shims still preserve live import contracts.

## Live invariants

- Turn allocation and `pending_turn` state must stay atomic with the advisory lock from `src/crud_shared.py:_acquire_session_turn_lock`.
- `TurnModel.ai_json` protected keys in `src/crud_turns_logic.py:_PROTECTED_TURN_AI_JSON_KEYS` remain backend-owned and cannot be patched through `PATCH /turns`.
- Pending-turn recovery must preserve committed session state and repair player location with `src/crud_core.py:_repair_player_location_after_pending_turn_recovery`.
- Context building must keep the immediately previous turn in the prompt window even when semantic retrieval evicts lower-ranked rows.
- Memory review/evaluation remain session-scoped; no cross-session retrieval or shared world ids were introduced.
- Memory policy selection and review tuning stay in `src/memory_policy.py`; no lexical, keyword, or regex routing was added to the extracted services.
- Pure continuity and memory-candidate invariants stay behind `src/domain/continuity_policy.py` and `src/domain/memory_candidates.py`; CRUD and persistence adapters do not own those rules.
- Callback cooldown and recall bookkeeping remain continuity-owned and are still applied after surfaced callback rows are used.
- Review projections (`__story_obligation`, `__memory_conflict_edge`, `__memory_review_report`, `__memory_evaluation_report`) remain derived artifacts, not canonical domain state.
- Worker threads only launch loops; loop registration is centralized in `src/workers/registry.py`.
- The internal audit UI remains a thin consumer over `/internal/memory-audit/*` JSON surfaces; no separate semantic backend was introduced for tooling.

## Boundary violations observed before extraction

- `src/crud_turns_logic.py` mixed rate limiting, plan resolution, patch application, memory persistence, observability, consequence scheduling, and fallback recovery in one module.
- `src/crud_context.py` mixed retrieval queries, prompt/context assembly, review-report reads, and observability shaping in one function.
- `src/memory_review.py` mixed query loading, policy orchestration, projection writes, and audit read endpoints in one file.
- `src/crud_core.py`, `src/crud_graph_ops.py`, and `src/crud_embeddings_ops.py` mixed read/query shaping or write-upsert behavior with public CRUD orchestration entrypoints.
- `src/crud_shared.py` and `src/crud_continuity.py` contained pure memory-candidate and continuity policy helpers alongside transactional/runtime helpers.
- Routes in `src/routes/internal.py` and `src/routes/sessions.py` depended directly on CRUD/review modules rather than application entrypoints.
- `src/main.py` hardcoded worker startup instead of using a registry/startup boundary.

## Legacy and deletion candidates

These remain for compatibility but are now candidates for removal once all call sites stop importing them directly:

- `src/crud_context.py:_build_turn_context_pack`
  - Now a compatibility wrapper over `TurnContextService.build_turn_context_pack`.
- `src/crud_turns_logic.py:_build_turn_plan_outside_apply_tx`
  - Now a compatibility wrapper over `TurnPlanningService.build_turn_plan_outside_apply_tx`.
- `src/crud_turns_logic.py:_apply_turn_plan`
  - Now a compatibility wrapper over `TurnApplicationService.apply_turn_plan`.
- `src/crud_turns_logic.py:run_turn`
  - Now a compatibility wrapper over `TurnApplicationService.run_turn`.
- `src/crud_turns_logic.py:_run_turn_locked`
  - Now a compatibility wrapper over `TurnApplicationService.run_turn_locked`.
- `src/memory_review.py:build_memory_review_report`
  - Now a compatibility wrapper over `MemoryReviewService.build_review_report`.
- `src/memory_review.py:run_memory_review_once`
  - Now a compatibility wrapper over `MemoryReviewService.run_review_once`.
- `src/memory_review.py:run_memory_review_loop`
  - Now a compatibility wrapper over `MemoryReviewService.run_review_loop`.
- `src/memory_review.py:get_memory_audit_*` and `list_*`
  - Now compatibility wrappers over `MemoryAuditService` / `MemoryEvaluationService`.
- `src/crud_core.py:get_session_timeline`, `get_session_diff`, `get_relationship_graph`
  - Now thin wrappers over `src/persistence/session_read_repository.py`.
- `src/crud_graph_ops.py:_list_recent_events_for_zone`, `_list_zone_entities_with_links`, `_list_asymmetric_relationships`, `_store_pending_graph_ops`, `get_pending_graph_ops`
  - Now thin wrappers over `src/persistence/graph_repository.py`.
- `src/crud_embeddings_ops.py:_find_existing_memory_event_row`, `_iter_memory_fact_rows_for_kind`
  - Now thin wrappers over `src/persistence/memory_write_repository.py`.
- `src/crud_shared.py:_merge_*`, `_canonicalize_*`, `_coalesce_*`, `_effective_durable_facts`
  - Now thin wrappers over `src/domain/memory_candidates.py`.
- `src/crud_continuity.py:_coerce_priority_score`, `_is_salient_item_name`, `_build_anchor_object_ids`, `_infer_scene_mode`, `_trim_memory_rows`
  - Now thin wrappers over `src/domain/continuity_policy.py`.

## How to add a new memory/continuity feature

1. Put pure scoring, classification, merge, or invariant logic in `src/domain/` or extend `src/domain/memory_policy.py` if it is memory-policy specific.
2. Put orchestration in the owning application service under `src/application/`; do not add new route logic or worker logic for that behavior.
3. Put reads/writes/query shaping in `src/persistence/` when the feature needs persistence-specific behavior.
4. Keep routes thin and worker startup thin: routes should call one service entrypoint, and workers should register only loop launchers in `src/workers/registry.py`.
5. Preserve the CRUD/facade contract only when an existing caller depends on it. If a shim is required, add a documented forwarder and register its status in `src/architecture_contracts.py`.

## Phase notes

- Phase 0
  - Completed with this repository-grounded map, use-case map, invariant list, and deletion-candidate list.
- Phase 1
  - Orchestration for turn execution, planning, context assembly, and memory review/evaluation/audit now lives in `src/application/`.
- Phase 2
  - Internal DTOs were added in `src/application/dtos.py` and are used across turn staging, memory retrieval shaping, callback rows, review reports, and audit snapshots.
- Phase 3
  - Memory review/audit projection reads and writes were extracted into `src/persistence/memory_repository.py`.
  - Broader live persistence/query paths were split into:
    - `src/persistence/session_read_repository.py`
    - `src/persistence/graph_repository.py`
    - `src/persistence/memory_write_repository.py`
- Phase 4
  - Apply-turn now executes through explicit staged methods in `TurnApplicationService`.
- Phase 5
  - Memory subsystem entrypoints are explicit:
    - write: `MemoryWriteService.persist_turn_memory_candidates`
    - retrieval/context: `TurnContextService.build_turn_context_pack`
    - review: `MemoryReviewService.run_review_once` / `build_review_report`
    - audit/evaluation: `MemoryAuditService.*`, `MemoryEvaluationService.get_evaluation_report`
- Phase 6
  - Routes call application services and startup uses `src/workers/registry.py`.
- Phase 7
  - Added focused tests for DTOs, route delegation, worker registry startup, repository review-row queries, staged turn service sequencing, direct domain policy modules, direct persistence repositories, and thin compatibility wrappers.
- Phase 8
  - Memory audit tooling now includes a thin trace-viewer/dashboard pass in the existing audit UI, backed by JSON health/snapshot/evaluation/trace-index endpoints without adding a new semantic backend layer.
