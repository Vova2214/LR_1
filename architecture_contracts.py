from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AdapterRole = Literal["permanent_public_facade", "temporary_deprecated_shim", "removable_now"]
AdapterVisibility = Literal["public", "internal", "deprecated"]


@dataclass(frozen=True)
class PackageExport:
    name: str
    module_name: str
    attr_name: str | None = None


@dataclass(frozen=True)
class PackageBoundary:
    package_name: str
    exports: tuple[PackageExport, ...]
    route_facing: tuple[str, ...] = ()
    facade_facing: tuple[str, ...] = ()
    internal_modules: tuple[str, ...] = ()
    test_only_modules: tuple[str, ...] = ()

    @property
    def public_exports(self) -> tuple[str, ...]:
        return tuple(export.name for export in self.exports)


@dataclass(frozen=True)
class CompatibilityModuleContract:
    module_name: str
    role: AdapterRole
    visibility: AdapterVisibility
    forwards_to: tuple[str, ...]
    symbols: tuple[str, ...]
    notes: str


CRUD_FACADE_ROUTE_API = (
    "create_session_with_defaults",
    "get_session",
    "delete_session",
    "create_session_snapshot",
    "get_session_snapshot",
    "list_session_snapshots",
    "create_object",
    "get_object",
    "list_objects",
    "create_link",
    "list_links",
    "list_events",
    "patch_turn",
    "run_turn",
    "recover_pending_turn",
    "move_player",
    "create_claim",
    "semantic_retrieve",
)

APPLICATION_PACKAGE_BOUNDARY = PackageBoundary(
    package_name="src.application",
    exports=(
        PackageExport("continuity_service", ".turn_services", "continuity_service"),
        PackageExport("debug_command_service", ".session_services", "debug_command_service"),
        PackageExport("entity_service", ".session_services", "entity_service"),
        PackageExport("memory_audit_service", ".memory_services", "memory_audit_service"),
        PackageExport("memory_evaluation_service", ".memory_services", "memory_evaluation_service"),
        PackageExport("memory_review_service", ".memory_services", "memory_review_service"),
        PackageExport("memory_write_service", ".turn_services", "memory_write_service"),
        PackageExport("player_command_service", ".command_services", "player_command_service"),
        PackageExport("semantic_tool_service", ".session_services", "semantic_tool_service"),
        PackageExport("session_lifecycle_service", ".session_services", "session_lifecycle_service"),
        PackageExport("telemetry_service", ".session_services", "telemetry_service"),
        PackageExport("turn_application_service", ".turn_services", "turn_application_service"),
        PackageExport("turn_context_service", ".turn_services", "turn_context_service"),
        PackageExport("turn_planning_service", ".turn_services", "turn_planning_service"),
    ),
    route_facing=(
        "debug_command_service",
        "entity_service",
        "memory_audit_service",
        "memory_evaluation_service",
        "memory_review_service",
        "player_command_service",
        "semantic_tool_service",
        "session_lifecycle_service",
        "telemetry_service",
        "turn_application_service",
    ),
    facade_facing=(
        "continuity_service",
        "memory_audit_service",
        "memory_evaluation_service",
        "memory_review_service",
        "memory_write_service",
        "turn_application_service",
        "turn_context_service",
        "turn_planning_service",
    ),
    internal_modules=(
        "command_services",
        "memory_services",
        "session_services",
        "tooling_config",
        "turn_contracts",
        "turn_services",
    ),
)

DOMAIN_PACKAGE_BOUNDARY = PackageBoundary(
    package_name="src.domain",
    exports=(
        PackageExport("continuity_policy", ".continuity_policy"),
        PackageExport("geography_policy", ".geography_policy"),
        PackageExport("memory_candidates", ".memory_candidates"),
        PackageExport("memory_policy", ".memory_policy"),
        PackageExport("player_commands", ".player_commands"),
    ),
    facade_facing=("continuity_policy", "geography_policy", "memory_candidates", "player_commands"),
)

PERSISTENCE_PACKAGE_BOUNDARY = PackageBoundary(
    package_name="src.persistence",
    exports=(
        PackageExport("graph_repository", ".graph_repository", "graph_repository"),
        PackageExport("memory_projection_repository", ".memory_repository", "memory_projection_repository"),
        PackageExport("memory_write_repository", ".memory_write_repository", "memory_write_repository"),
        PackageExport("player_command_repository", ".player_command_repository", "player_command_repository"),
        PackageExport("session_read_repository", ".session_read_repository", "session_read_repository"),
    ),
    facade_facing=(
        "graph_repository",
        "memory_projection_repository",
        "memory_write_repository",
        "player_command_repository",
        "session_read_repository",
    ),
    internal_modules=(
        "graph_repository",
        "memory_repository",
        "memory_write_repository",
        "player_command_repository",
        "session_read_repository",
    ),
)

WORKERS_PACKAGE_BOUNDARY = PackageBoundary(
    package_name="src.workers",
    exports=(
        PackageExport("WorkerSpec", ".registry", "WorkerSpec"),
        PackageExport("startup_worker_specs", ".registry", "startup_worker_specs"),
    ),
    internal_modules=("registry",),
)

PACKAGE_BOUNDARIES = {
    APPLICATION_PACKAGE_BOUNDARY.package_name: APPLICATION_PACKAGE_BOUNDARY,
    DOMAIN_PACKAGE_BOUNDARY.package_name: DOMAIN_PACKAGE_BOUNDARY,
    PERSISTENCE_PACKAGE_BOUNDARY.package_name: PERSISTENCE_PACKAGE_BOUNDARY,
    WORKERS_PACKAGE_BOUNDARY.package_name: WORKERS_PACKAGE_BOUNDARY,
}

COMPATIBILITY_MODULE_CONTRACTS = {
    "src.crud": CompatibilityModuleContract(
        module_name="src.crud",
        role="permanent_public_facade",
        visibility="public",
        forwards_to=(
            "src.application",
            "src.domain",
            "src.persistence",
            "leaf src.crud_* modules",
        ),
        symbols=CRUD_FACADE_ROUTE_API,
        notes="Stable legacy facade and monkeypatch surface. New orchestration should not be added here.",
    ),
    "src.crud_core": CompatibilityModuleContract(
        module_name="src.crud_core",
        role="temporary_deprecated_shim",
        visibility="deprecated",
        forwards_to=("src.persistence.session_read_repository",),
        symbols=("get_session_timeline", "get_session_diff", "get_relationship_graph"),
        notes="Legacy CRUD entrypoints preserved while callers migrate to services or repositories.",
    ),
    "src.crud_graph_ops": CompatibilityModuleContract(
        module_name="src.crud_graph_ops",
        role="temporary_deprecated_shim",
        visibility="deprecated",
        forwards_to=("src.persistence.graph_repository",),
        symbols=(
            "_normalize_pending_graph_op",
            "_list_recent_events_for_zone",
            "_list_zone_entities_with_links",
            "_list_asymmetric_relationships",
            "_store_pending_graph_ops",
            "get_pending_graph_ops",
        ),
        notes="Compatibility graph wrappers over repository-backed read/write helpers.",
    ),
    "src.crud_embeddings_ops": CompatibilityModuleContract(
        module_name="src.crud_embeddings_ops",
        role="temporary_deprecated_shim",
        visibility="deprecated",
        forwards_to=("src.persistence.memory_write_repository",),
        symbols=("_iter_memory_fact_rows_for_kind", "_find_existing_memory_event_row"),
        notes="Compatibility memory lookup wrappers over repository-backed persistence helpers.",
    ),
    "src.crud_shared": CompatibilityModuleContract(
        module_name="src.crud_shared",
        role="temporary_deprecated_shim",
        visibility="deprecated",
        forwards_to=("src.domain.memory_candidates",),
        symbols=(
            "_merge_durable_fact_priority",
            "_merge_durable_fact_scope",
            "_merge_callback_strength",
            "_merge_unique_refs",
            "_resolve_fact_identity_ref",
            "_canonicalize_durable_fact_identity_refs",
            "_canonicalize_memory_candidate_identity_refs",
            "_normalized_fact_identity_text",
            "_fact_identity_text_value",
            "_with_fact_identity_text",
            "_fact_identity_ref_signature",
            "_durable_fact_signature",
            "_merge_durable_fact",
            "_merge_fact_memory_candidates",
            "_merge_event_memory_candidates",
            "_commit_event_scene_ref_signature",
            "_memory_candidate_commit_scene_signature",
            "_memory_candidate_identity_signature",
            "_coalesce_memory_candidates_by_identity",
            "_memory_candidates_to_durable_facts",
            "_default_memory_candidate_durability",
            "_durable_fact_to_memory_candidate",
            "_committing_fact_memory_candidate",
            "_supplement_memory_candidates_with_durable_facts",
            "_memory_fact_identity_key",
            "_effective_durable_facts",
        ),
        notes="Legacy durable-fact and memory-candidate helpers preserved as domain forwards only.",
    ),
    "src.crud_continuity": CompatibilityModuleContract(
        module_name="src.crud_continuity",
        role="temporary_deprecated_shim",
        visibility="deprecated",
        forwards_to=("src.domain.continuity_policy",),
        symbols=(
            "_coerce_priority_score",
            "_is_salient_item_name",
            "_build_anchor_object_ids",
            "_infer_scene_mode",
            "_trim_memory_rows",
        ),
        notes="Legacy continuity policy helpers preserved as forwards to the domain boundary.",
    ),
}
