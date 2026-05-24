from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal

from .crud_shared import _truncate_text

LoreUsage = Literal["runtime", "campaign", "special_case", "reference", "bookkeeping"]
LoreScope = Literal["world", "region", "faction", "character", "campaign"]
LoreDomain = Literal[
    "magic",
    "economy",
    "law",
    "military",
    "religion",
    "geography",
    "social",
    "progression",
    "equipment",
]
LoreCriticality = Literal["blocking", "important", "optional"]
LoreFrameSignal = Literal["power_fantasy", "tone_profile", "activity_bias"]
LorePreviewBlock = Literal[
    "runtime_rules",
    "campaign_frame",
    "special_cases",
    "reference_catalog",
    "bookkeeping_rules",
    "open_questions",
]

_ALLOWED_USAGES = {"runtime", "campaign", "special_case", "reference", "bookkeeping"}
_ALLOWED_SCOPES = {"world", "region", "faction", "character", "campaign"}
_ALLOWED_DOMAINS = {
    "magic",
    "economy",
    "law",
    "military",
    "religion",
    "geography",
    "social",
    "progression",
    "equipment",
}
_ALLOWED_CRITICALITIES = {"blocking", "important", "optional"}
_ALLOWED_FRAME_SIGNALS = {"power_fantasy", "tone_profile", "activity_bias"}
_CRITICALITY_ORDER = {"blocking": 0, "important": 1, "optional": 2}


@dataclass(frozen=True, slots=True)
class LoreClaim:
    claim_key: str
    label: str
    value: str | list[str] | None
    usage: LoreUsage
    scope: LoreScope
    domain: LoreDomain
    criticality: LoreCriticality
    confidence: float
    source_section: str = ""
    source_excerpt: str = ""
    missing: bool = False
    question: str | None = None
    frame_signal: LoreFrameSignal | None = None


@dataclass(frozen=True, slots=True)
class LoreUxPolicy:
    top_level_blocks: tuple[LorePreviewBlock, ...]
    usage_to_preview_block: tuple[tuple[LoreUsage, LorePreviewBlock], ...]
    blocking_usages: tuple[LoreUsage, ...]
    blocking_confidence_floor: float
    max_blocking_questions: int
    max_optional_questions: int
    max_runtime_rules: int
    max_special_cases: int
    max_bookkeeping_rules: int
    max_reference_items_per_group: int
    max_campaign_signal_notes: int
    max_draft_preview_claims: int
    max_runtime_render_per_section: int
    max_runtime_render_lines: int
    max_compiled_list_items: int
    max_value_chars: int
    max_source_excerpt_chars: int
    blocking_usage_order: tuple[LoreUsage, ...]
    scope_to_reference_group: tuple[tuple[LoreScope, str], ...]


LORE_UX_POLICY = LoreUxPolicy(
    top_level_blocks=(
        "runtime_rules",
        "campaign_frame",
        "special_cases",
        "reference_catalog",
        "bookkeeping_rules",
        "open_questions",
    ),
    usage_to_preview_block=(
        ("runtime", "runtime_rules"),
        ("campaign", "campaign_frame"),
        ("special_case", "special_cases"),
        ("reference", "reference_catalog"),
        ("bookkeeping", "bookkeeping_rules"),
    ),
    blocking_usages=("runtime", "campaign", "special_case", "bookkeeping"),
    blocking_confidence_floor=0.6,
    max_blocking_questions=4,
    max_optional_questions=12,
    max_runtime_rules=6,
    max_special_cases=6,
    max_bookkeeping_rules=5,
    max_reference_items_per_group=6,
    max_campaign_signal_notes=4,
    max_draft_preview_claims=10,
    max_runtime_render_per_section=4,
    max_runtime_render_lines=16,
    max_compiled_list_items=5,
    max_value_chars=220,
    max_source_excerpt_chars=220,
    blocking_usage_order=("runtime", "campaign", "special_case", "bookkeeping", "reference"),
    scope_to_reference_group=(
        ("world", "world"),
        ("region", "regions"),
        ("faction", "factions"),
        ("character", "characters"),
        ("campaign", "campaign"),
    ),
)


def _coerce_text(value: Any, *, max_chars: int = 400) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return _truncate_text(text, max_chars)


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _coerce_claim_value(value: Any) -> str | list[str] | None:
    if isinstance(value, list):
        normalized = _dedupe_strings([_coerce_text(item, max_chars=120) for item in value])
        return normalized or None
    if isinstance(value, dict):
        rendered = _coerce_text(json.dumps(value, ensure_ascii=False, sort_keys=True), max_chars=600)
        return rendered or None
    text = _coerce_text(value, max_chars=600)
    return text or None


def _coerce_enum(value: Any, *, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    if text in allowed:
        return text
    return default


def claim_value_text(value: Any, *, max_chars: int | None = None) -> str:
    if isinstance(value, list):
        text = ", ".join(_dedupe_strings([_coerce_text(item, max_chars=120) for item in value]))
    elif isinstance(value, dict):
        text = _coerce_text(json.dumps(value, ensure_ascii=False, sort_keys=True), max_chars=600)
    else:
        text = _coerce_text(value, max_chars=600)
    if max_chars is not None and text:
        return _truncate_text(text, max_chars)
    return text


def claim_value_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return _dedupe_strings([_coerce_text(item, max_chars=80) for item in value])
    text = _coerce_text(value, max_chars=120)
    return [text] if text else []


def claim_has_resolved_value(claim: LoreClaim) -> bool:
    return not claim.missing and bool(claim_value_text(claim.value))


def serialize_claim(claim: LoreClaim) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "claim_key": claim.claim_key,
        "label": claim.label,
        "value": claim.value,
        "usage": claim.usage,
        "scope": claim.scope,
        "domain": claim.domain,
        "criticality": claim.criticality,
        "confidence": claim.confidence,
        "source_section": claim.source_section,
        "source_excerpt": claim.source_excerpt,
        "missing": claim.missing,
        "question": claim.question,
        "frame_signal": claim.frame_signal,
    }
    return payload


def deserialize_claim(raw_claim: Any) -> LoreClaim | None:
    if not isinstance(raw_claim, dict):
        return None
    claim_key = _coerce_text(raw_claim.get("claim_key") or raw_claim.get("key"), max_chars=120)
    if not claim_key:
        return None
    label = _coerce_text(raw_claim.get("label"), max_chars=200) or claim_key.replace("_", " ").title()
    value = _coerce_claim_value(raw_claim.get("value"))
    usage = _coerce_enum(raw_claim.get("usage"), allowed=_ALLOWED_USAGES, default="reference")
    scope = _coerce_enum(raw_claim.get("scope"), allowed=_ALLOWED_SCOPES, default="world")
    domain = _coerce_enum(raw_claim.get("domain"), allowed=_ALLOWED_DOMAINS, default="progression")
    criticality = _coerce_enum(
        raw_claim.get("criticality"),
        allowed=_ALLOWED_CRITICALITIES,
        default="important" if value else "optional",
    )
    frame_signal_raw = _coerce_enum(
        raw_claim.get("frame_signal"),
        allowed=_ALLOWED_FRAME_SIGNALS,
        default="",
    )
    question = _coerce_text(raw_claim.get("question"), max_chars=320) or None
    missing = bool(raw_claim.get("missing")) or (value is None and question is not None)
    if value is None and not missing:
        missing = True
    return LoreClaim(
        claim_key=claim_key,
        label=label,
        value=value,
        usage=usage,
        scope=scope,
        domain=domain,
        criticality=criticality,
        confidence=_coerce_float(raw_claim.get("confidence"), default=0.0),
        source_section=_coerce_text(raw_claim.get("source_section"), max_chars=160),
        source_excerpt=_coerce_text(raw_claim.get("source_excerpt"), max_chars=320),
        missing=missing,
        question=question,
        frame_signal=frame_signal_raw or None,
    )


def deserialize_claims(raw_claims: Any) -> list[LoreClaim]:
    if not isinstance(raw_claims, list):
        return []
    claims: list[LoreClaim] = []
    seen: set[str] = set()
    for raw_claim in raw_claims:
        claim = deserialize_claim(raw_claim)
        if claim is None or claim.claim_key in seen:
            continue
        seen.add(claim.claim_key)
        claims.append(claim)
    return claims


def legacy_profile_to_claims(profile: Any) -> list[LoreClaim]:
    if not isinstance(profile, dict):
        return []
    claims: list[LoreClaim] = []
    for raw_key, raw_value in sorted(profile.items()):
        claim_key = _coerce_text(raw_key, max_chars=120)
        value = _coerce_text(raw_value, max_chars=400)
        if not claim_key or not value:
            continue
        claims.append(
            LoreClaim(
                claim_key=claim_key,
                label=claim_key.replace("_", " ").title(),
                value=value,
                usage="runtime",
                scope="world",
                domain="progression",
                criticality="important",
                confidence=0.55,
            )
        )
    return claims


def claims_from_world_constitution_data(data: dict[str, Any]) -> list[LoreClaim]:
    if not isinstance(data, dict):
        return []
    claims = deserialize_claims(data.get("lore_claims"))
    if claims:
        return claims
    return legacy_profile_to_claims(data.get("lore_profile"))


def preview_block_for_usage(
    usage: str,
    *,
    policy: LoreUxPolicy = LORE_UX_POLICY,
) -> LorePreviewBlock:
    for candidate_usage, block_name in policy.usage_to_preview_block:
        if candidate_usage == usage:
            return block_name
    return "reference_catalog"


def reference_group_for_scope(
    scope: str,
    *,
    policy: LoreUxPolicy = LORE_UX_POLICY,
) -> str:
    for candidate_scope, group_name in policy.scope_to_reference_group:
        if candidate_scope == scope:
            return group_name
    return "other"


def gap_turn_block_candidates(
    gaps: list[dict[str, Any]],
    *,
    policy: LoreUxPolicy = LORE_UX_POLICY,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    usage_order = {usage: idx for idx, usage in enumerate(policy.blocking_usage_order)}
    for raw_gap in gaps:
        if not isinstance(raw_gap, dict):
            continue
        if str(raw_gap.get("status") or "open").strip().lower() != "open":
            continue
        usage = _coerce_enum(raw_gap.get("usage"), allowed=_ALLOWED_USAGES, default="reference")
        if usage not in policy.blocking_usages:
            continue
        criticality = _coerce_enum(
            raw_gap.get("criticality"),
            allowed=_ALLOWED_CRITICALITIES,
            default="optional",
        )
        if criticality != "blocking":
            continue
        confidence = _coerce_float(raw_gap.get("confidence"), default=0.0)
        if confidence < policy.blocking_confidence_floor:
            continue
        candidates.append(dict(raw_gap))
    candidates.sort(
        key=lambda gap: (
            usage_order.get(str(gap.get("usage") or ""), len(usage_order)),
            _CRITICALITY_ORDER.get(str(gap.get("criticality") or "optional"), 2),
            -_coerce_float(gap.get("confidence"), default=0.0),
            _coerce_text(gap.get("label"), max_chars=200) or _coerce_text(gap.get("key"), max_chars=120),
            _coerce_text(gap.get("gap_id"), max_chars=200),
        )
    )
    return candidates


def partition_open_gaps(
    gaps: list[dict[str, Any]],
    *,
    policy: LoreUxPolicy = LORE_UX_POLICY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = gap_turn_block_candidates(gaps, policy=policy)
    blocking_ids = {
        _coerce_text(gap.get("gap_id"), max_chars=200)
        for gap in candidates[: policy.max_blocking_questions]
    }
    blocking: list[dict[str, Any]] = []
    optional: list[dict[str, Any]] = []
    for raw_gap in gaps:
        if not isinstance(raw_gap, dict):
            continue
        if str(raw_gap.get("status") or "open").strip().lower() != "open":
            continue
        gap_id = _coerce_text(raw_gap.get("gap_id"), max_chars=200)
        if gap_id and gap_id in blocking_ids:
            blocking.append(dict(raw_gap))
        else:
            optional.append(dict(raw_gap))
    blocking.sort(
        key=lambda gap: (
            _coerce_text(gap.get("label"), max_chars=200) or _coerce_text(gap.get("key"), max_chars=120),
            _coerce_text(gap.get("gap_id"), max_chars=200),
        )
    )
    optional.sort(
        key=lambda gap: (
            _CRITICALITY_ORDER.get(str(gap.get("criticality") or "optional"), 2),
            -_coerce_float(gap.get("confidence"), default=0.0),
            _coerce_text(gap.get("label"), max_chars=200) or _coerce_text(gap.get("key"), max_chars=120),
            _coerce_text(gap.get("gap_id"), max_chars=200),
        )
    )
    return blocking, optional


def _claim_sort_key(
    claim: LoreClaim,
    *,
    policy: LoreUxPolicy = LORE_UX_POLICY,
) -> tuple[int, int, str, str, str]:
    usage_order = {usage: idx for idx, usage in enumerate(policy.blocking_usage_order)}
    scope_order = {group: idx for idx, group in enumerate(["world", "region", "faction", "character", "campaign"])}
    return (
        usage_order.get(claim.usage, len(usage_order)),
        scope_order.get(claim.scope, len(scope_order)),
        claim.label.lower(),
        claim.claim_key,
        claim.domain,
    )


def _claim_preview_item(
    claim: LoreClaim,
    *,
    policy: LoreUxPolicy = LORE_UX_POLICY,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "claim_key": claim.claim_key,
        "label": claim.label,
        "value": claim.value,
        "scope": claim.scope,
        "domain": claim.domain,
        "criticality": claim.criticality,
        "confidence": claim.confidence,
    }
    if claim.source_section:
        item["source_section"] = claim.source_section
    if claim.source_excerpt:
        item["source_excerpt"] = _truncate_text(claim.source_excerpt, policy.max_source_excerpt_chars)
    if claim.frame_signal:
        item["frame_signal"] = claim.frame_signal
    return item


def _gap_preview_item(gap: dict[str, Any], *, blocking: bool) -> dict[str, Any]:
    item: dict[str, Any] = {
        "gap_id": _coerce_text(gap.get("gap_id"), max_chars=200),
        "claim_key": _coerce_text(gap.get("key"), max_chars=120),
        "label": _coerce_text(gap.get("label"), max_chars=200),
        "question": _coerce_text(gap.get("question"), max_chars=320),
        "usage": _coerce_enum(gap.get("usage"), allowed=_ALLOWED_USAGES, default="reference"),
        "scope": _coerce_enum(gap.get("scope"), allowed=_ALLOWED_SCOPES, default="world"),
        "domain": _coerce_enum(gap.get("domain"), allowed=_ALLOWED_DOMAINS, default="progression"),
        "criticality": _coerce_enum(
            gap.get("criticality"),
            allowed=_ALLOWED_CRITICALITIES,
            default="optional",
        ),
        "confidence": _coerce_float(gap.get("confidence"), default=0.0),
        "blocks_turns": blocking,
    }
    why_text = _coerce_text(gap.get("why_this_blocks_turns"), max_chars=220)
    if why_text:
        item["why_this_blocks_turns"] = why_text
    return item


def campaign_frame_from_claims(
    claims: list[LoreClaim],
    *,
    compiled_world_model: dict[str, Any] | None = None,
    policy: LoreUxPolicy = LORE_UX_POLICY,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "power_fantasy": None,
        "tone_profile": [],
        "activity_bias": [],
        "signals": [],
    }
    for claim in sorted([item for item in claims if item.usage == "campaign"], key=lambda item: _claim_sort_key(item, policy=policy)):
        value_text = claim_value_text(claim.value, max_chars=policy.max_value_chars)
        if not value_text and not isinstance(claim.value, list):
            continue
        if claim.frame_signal == "power_fantasy":
            frame["power_fantasy"] = frame["power_fantasy"] or value_text
        elif claim.frame_signal == "tone_profile":
            frame["tone_profile"] = _dedupe_strings(frame["tone_profile"] + claim_value_list(claim.value))
        elif claim.frame_signal == "activity_bias":
            frame["activity_bias"] = _dedupe_strings(frame["activity_bias"] + claim_value_list(claim.value))
        elif len(frame["signals"]) < policy.max_campaign_signal_notes:
            frame["signals"].append(_claim_preview_item(claim, policy=policy))

    compiled_frame = dict((compiled_world_model or {}).get("campaign_frame") or {})
    if not frame["power_fantasy"]:
        frame["power_fantasy"] = _coerce_text(compiled_frame.get("power_fantasy"), max_chars=80) or None
    frame["tone_profile"] = _dedupe_strings(
        frame["tone_profile"]
        + claim_value_list(compiled_frame.get("tone_profile"))
    )
    frame["activity_bias"] = _dedupe_strings(
        frame["activity_bias"]
        + claim_value_list(compiled_frame.get("activity_bias"))
    )
    if not frame["signals"]:
        frame.pop("signals", None)
    if not frame["tone_profile"]:
        frame.pop("tone_profile", None)
    if not frame["activity_bias"]:
        frame.pop("activity_bias", None)
    if not frame["power_fantasy"]:
        frame.pop("power_fantasy", None)
    return frame


def build_lore_preview(
    claims: list[LoreClaim],
    gaps: list[dict[str, Any]],
    *,
    compiled_world_model: dict[str, Any] | None = None,
    policy: LoreUxPolicy = LORE_UX_POLICY,
) -> dict[str, Any]:
    preview: dict[str, Any] = {
        "runtime_rules": [],
        "campaign_frame": {},
        "special_cases": [],
        "reference_catalog": {},
        "bookkeeping_rules": [],
        "open_questions": {"blocking": [], "optional": []},
    }
    resolved_claims = [claim for claim in claims if claim_has_resolved_value(claim)]
    reference_catalog: dict[str, list[dict[str, Any]]] = {}
    for claim in sorted(resolved_claims, key=lambda item: _claim_sort_key(item, policy=policy)):
        block_name = preview_block_for_usage(claim.usage, policy=policy)
        item = _claim_preview_item(claim, policy=policy)
        if block_name == "runtime_rules":
            if len(preview["runtime_rules"]) < policy.max_runtime_rules:
                preview["runtime_rules"].append(item)
            continue
        if block_name == "special_cases":
            if len(preview["special_cases"]) < policy.max_special_cases:
                preview["special_cases"].append(item)
            continue
        if block_name == "bookkeeping_rules":
            if len(preview["bookkeeping_rules"]) < policy.max_bookkeeping_rules:
                preview["bookkeeping_rules"].append(item)
            continue
        if block_name == "reference_catalog":
            group_name = reference_group_for_scope(claim.scope, policy=policy)
            group_items = reference_catalog.setdefault(group_name, [])
            if len(group_items) < policy.max_reference_items_per_group:
                group_items.append(item)

    preview["campaign_frame"] = campaign_frame_from_claims(
        claims,
        compiled_world_model=compiled_world_model,
        policy=policy,
    )
    preview["reference_catalog"] = reference_catalog

    blocking_gaps, optional_gaps = partition_open_gaps(gaps, policy=policy)
    preview["open_questions"]["blocking"] = [
        _gap_preview_item(gap, blocking=True)
        for gap in blocking_gaps[: policy.max_blocking_questions]
    ]
    preview["open_questions"]["optional"] = [
        _gap_preview_item(gap, blocking=False)
        for gap in optional_gaps[: policy.max_optional_questions]
    ]
    return preview


def build_lore_coverage_summary(
    claims: list[LoreClaim],
    gaps: list[dict[str, Any]],
    *,
    policy: LoreUxPolicy = LORE_UX_POLICY,
) -> dict[str, Any]:
    resolved_claims = [claim for claim in claims if claim_has_resolved_value(claim)]
    blocking_gaps, optional_gaps = partition_open_gaps(gaps, policy=policy)
    return {
        "claim_count": len(claims),
        "resolved_claim_count": len(resolved_claims),
        "runtime_rule_count": len([claim for claim in resolved_claims if claim.usage == "runtime"]),
        "campaign_signal_count": len([claim for claim in resolved_claims if claim.usage == "campaign"]),
        "special_case_count": len([claim for claim in resolved_claims if claim.usage == "special_case"]),
        "reference_item_count": len([claim for claim in resolved_claims if claim.usage == "reference"]),
        "bookkeeping_rule_count": len([claim for claim in resolved_claims if claim.usage == "bookkeeping"]),
        "blocking_count": len(blocking_gaps),
        "optional_count": len(optional_gaps),
    }


def build_lore_blocking_summary(
    status: str,
    gaps: list[dict[str, Any]],
    *,
    policy: LoreUxPolicy = LORE_UX_POLICY,
) -> dict[str, Any]:
    normalized_status = str(status or "").strip().lower()
    if normalized_status == "processing":
        return {
            "blocked": True,
            "blocking_gap_ids": [],
            "blocking_labels": [],
            "blocking_domains": [],
            "why_this_blocks_turns": "Lore adaptation is still processing.",
        }
    blocking_gaps, _optional_gaps = partition_open_gaps(gaps, policy=policy)
    blocking_domains = _dedupe_strings(
        [
            _coerce_enum(gap.get("domain"), allowed=_ALLOWED_DOMAINS, default="progression")
            for gap in blocking_gaps
        ]
    )
    blocking_labels = [
        _coerce_text(gap.get("label"), max_chars=200) or _coerce_text(gap.get("key"), max_chars=120)
        for gap in blocking_gaps
    ]
    blocked = bool(blocking_gaps)
    return {
        "blocked": blocked,
        "blocking_gap_ids": [
            _coerce_text(gap.get("gap_id"), max_chars=200)
            for gap in blocking_gaps
        ],
        "blocking_labels": blocking_labels,
        "blocking_domains": blocking_domains,
        "why_this_blocks_turns": (
            "Turns are blocked only for unresolved gameplay-critical lore questions about runtime rules, "
            "campaign framing, special cases, or bookkeeping."
            if blocked
            else ""
        ),
    }


def build_lore_draft_profile_preview(
    claims: list[LoreClaim],
    *,
    policy: LoreUxPolicy = LORE_UX_POLICY,
) -> dict[str, Any]:
    resolved_claims = [
        _claim_preview_item(claim, policy=policy)
        for claim in sorted(claims, key=lambda item: _claim_sort_key(item, policy=policy))
        if claim_has_resolved_value(claim)
    ]
    return {
        "claim_count": len(resolved_claims),
        "claims": resolved_claims[: policy.max_draft_preview_claims],
    }


def build_compiled_world_model_preview(
    compiled_world_model: dict[str, Any] | None,
    *,
    policy: LoreUxPolicy = LORE_UX_POLICY,
) -> dict[str, Any]:
    compiled = dict(compiled_world_model or {})
    if not compiled:
        return {}
    preview: dict[str, Any] = {}
    expansion_policy = _coerce_text(compiled.get("expansion_policy"), max_chars=80)
    if expansion_policy:
        preview["expansion_policy"] = expansion_policy
    core_envelope_raw = dict(compiled.get("core_envelope") or {})
    core_envelope: dict[str, Any] = {}
    for field_name in (
        "tech_level",
        "magic_level",
        "mobility_profile",
        "conflict_profile",
        "social_structure",
        "genre_signals",
        "tone_signals",
    ):
        raw_value = core_envelope_raw.get(field_name)
        if isinstance(raw_value, list):
            normalized = _dedupe_strings(
                [_coerce_text(item, max_chars=80) for item in raw_value]
            )[: policy.max_compiled_list_items]
            if normalized:
                core_envelope[field_name] = normalized
        else:
            value_text = _coerce_text(raw_value, max_chars=120)
            if value_text:
                core_envelope[field_name] = value_text
    if core_envelope:
        preview["core_envelope"] = core_envelope
    for field_name in ("affordances", "exceptions", "forbidden_or_rare_elements"):
        raw_value = compiled.get(field_name)
        if not isinstance(raw_value, list):
            continue
        normalized = _dedupe_strings(
            [_coerce_text(item, max_chars=80) for item in raw_value]
        )[: policy.max_compiled_list_items]
        if normalized:
            preview[field_name] = normalized
    campaign_frame = campaign_frame_from_claims([], compiled_world_model=compiled, policy=policy)
    if campaign_frame:
        preview["campaign_frame"] = campaign_frame
    custom_axes = dict(compiled.get("custom_axes") or {})
    if custom_axes:
        preview["custom_axes"] = {
            _coerce_text(key, max_chars=40): _coerce_text(value, max_chars=120)
            for key, value in list(custom_axes.items())[: policy.max_compiled_list_items]
            if _coerce_text(key, max_chars=40) and _coerce_text(value, max_chars=120)
        }
    return preview


def build_runtime_lore_brief(
    claims: list[LoreClaim],
    *,
    policy: LoreUxPolicy = LORE_UX_POLICY,
) -> str:
    resolved_claims = [claim for claim in claims if claim_has_resolved_value(claim)]
    runtime_rules = [claim for claim in resolved_claims if claim.usage == "runtime"]
    special_cases = [claim for claim in resolved_claims if claim.usage == "special_case"]
    bookkeeping_rules = [claim for claim in resolved_claims if claim.usage == "bookkeeping"]
    campaign_frame = campaign_frame_from_claims(resolved_claims, policy=policy)

    lines: list[str] = []
    if runtime_rules:
        lines.append("RUNTIME RULES:")
        for claim in sorted(runtime_rules, key=lambda item: _claim_sort_key(item, policy=policy))[
            : policy.max_runtime_render_per_section
        ]:
            lines.append(
                f"- {claim.label}: {claim_value_text(claim.value, max_chars=policy.max_value_chars)}"
            )
    if campaign_frame:
        lines.append("CAMPAIGN FRAME:")
        power_fantasy = _coerce_text(campaign_frame.get("power_fantasy"), max_chars=80)
        if power_fantasy:
            lines.append(f"- power_fantasy: {power_fantasy}")
        tone_profile = claim_value_list(campaign_frame.get("tone_profile"))
        if tone_profile:
            lines.append(
                f"- tone_profile: {', '.join(tone_profile[: policy.max_runtime_render_per_section])}"
            )
        activity_bias = claim_value_list(campaign_frame.get("activity_bias"))
        if activity_bias:
            lines.append(
                f"- activity_bias: {', '.join(activity_bias[: policy.max_runtime_render_per_section])}"
            )
        for signal in list(campaign_frame.get("signals") or [])[: policy.max_campaign_signal_notes]:
            label = _coerce_text(signal.get("label"), max_chars=80)
            value_text = claim_value_text(signal.get("value"), max_chars=policy.max_value_chars)
            if label and value_text:
                lines.append(f"- {label}: {value_text}")
    if special_cases:
        lines.append("SPECIAL CASES:")
        for claim in sorted(special_cases, key=lambda item: _claim_sort_key(item, policy=policy))[
            : policy.max_runtime_render_per_section
        ]:
            lines.append(
                f"- {claim.label}: {claim_value_text(claim.value, max_chars=policy.max_value_chars)}"
            )
    if bookkeeping_rules:
        lines.append("BOOKKEEPING RULES:")
        for claim in sorted(bookkeeping_rules, key=lambda item: _claim_sort_key(item, policy=policy))[
            : policy.max_runtime_render_per_section
        ]:
            lines.append(
                f"- {claim.label}: {claim_value_text(claim.value, max_chars=policy.max_value_chars)}"
            )
    return "\n".join(lines[: policy.max_runtime_render_lines])

