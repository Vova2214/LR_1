from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from . import models, outbox_runtime, schemas
from .crud_shared import (
    _acquire_session_turn_lock,
    _create_internal_turn_row,
    _extract_in_game_time,
    _normalize_json_preview,
    _recover_abandoned_pending_turn_locked,
    _require_session,
    _truncate_text,
)
from .db import (
    LORE_ADAPTATION_MAX_CHARS,
    LORE_ADAPTATION_RETRY_AFTER_SECONDS,
    LORE_ADAPTATION_TIMEOUT_SECONDS,
    OPENROUTER_CHAT_MODEL,
    SessionLocal,
    USE_LORE_ADAPTATION,
)
from .llm import openrouter_chat
from .llm_telemetry import telemetry_context
from .lore_ux import (
    LORE_UX_POLICY,
    LoreClaim,
    build_compiled_world_model_preview,
    build_lore_blocking_summary,
    build_lore_coverage_summary,
    build_lore_draft_profile_preview,
    build_lore_preview,
    claim_has_resolved_value,
    claim_value_list,
    claim_value_text,
    claims_from_world_constitution_data,
    deserialize_claims,
    gap_turn_block_candidates,
    serialize_claim,
)
from .observability import get_trace_id

logger = logging.getLogger(__name__)

_LORE_STATE_KEY = "lore_adaptation"
_VALID_LORE_STATUSES = {"idle", "processing", "awaiting_answers", "finalized"}
_MAX_LORE_CLAIM_COUNT = 24
_VALID_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,118}$")
_LORE_PROCESSING_REQUEST_ID_KEY = "processing_request_id"
_LORE_STABLE_STATE_KEY = "stable_state"
_LORE_MAX_OUTBOX_ATTEMPTS = 3
_WORLD_MODEL_HINTS_KEY = "world_model_hints"
_WORLD_MODEL_MAX_LIST_ITEMS = 12
_WORLD_MODEL_MAX_CUSTOM_AXES = 8
_WORLD_MODEL_VERSION = 2
_VALID_LORE_USAGES = {usage for usage, _block in LORE_UX_POLICY.usage_to_preview_block}
_VALID_LORE_SCOPES = {"world", "region", "faction", "character", "campaign"}
_VALID_LORE_DOMAINS = {
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
_VALID_LORE_CRITICALITIES = {"blocking", "important", "optional"}
_VALID_FRAME_SIGNALS = {"power_fantasy", "tone_profile", "activity_bias"}

_LORE_ANALYSIS_SYSTEM = (
    "You are a world-lore adaptation parser for an RPG engine. "
    "Read the provided lore text and extract normalized gameplay claims for consistent narration. "
    "Do NOT use a fixed checklist. Discover claims organically from the lore, but always classify "
    "each claim into the server-owned taxonomies below. "
    "Return JSON only with fields: language, claims, world_model. "
    "claims must be an array (max 24 entries) where each entry is: "
    "{claim_key, label, value, missing, question, usage, scope, domain, criticality, confidence, "
    "source_section, source_excerpt, frame_signal}. "
    "claim_key must be a unique lowercase snake_case identifier (e.g. 'blood_magic_rules'). "
    "usage must be one of: runtime, campaign, special_case, reference, bookkeeping. "
    "scope must be one of: world, region, faction, character, campaign. "
    "domain must be one of: magic, economy, law, military, religion, geography, social, progression, equipment. "
    "criticality must be one of: blocking, important, optional. "
    "frame_signal is optional and only for usage=campaign; if used, it must be one of: "
    "power_fantasy, tone_profile, activity_bias. "
    "Set missing=true when the lore does not define the claim clearly enough for gameplay decisions. "
    "If missing=true, provide a concrete player-facing question in the same language as the lore. "
    "If missing=false, provide a concise claim value extracted from the lore. "
    "confidence must be a number from 0 to 1. "
    "Only include claims that are genuinely useful for this world. "
    "Do not invent custom top-level preview blocks. "
    "world_model is optional but should be included when inferable. "
    "world_model must describe the session-specific world envelope, not the entire lore verbatim. "
    "Use lowercase English identifiers for structured world_model values when possible. "
    "world_model fields may include: expansion_policy, tech_level, magic_level, mobility_profile, "
    "conflict_profile, social_structure, genre_signals, tone_signals, affordances, exceptions, "
    "forbidden_or_rare_elements, custom_axes, campaign_frame. "
    "campaign_frame may include: power_fantasy, tone_profile, activity_bias. "
    "Treat the lore as directional, not exhaustive: infer the baseline world envelope from what the lore says exists. "
    "Do not require the player to explicitly list everything that does not exist. "
    "Only put unusual but allowed elements into exceptions. "
    "Never invent non-JSON text."
)
_LORE_FILL_SYSTEM = (
    "You fill unresolved RPG mechanics from lore context. "
    "Return JSON only: {fills:[{key,value}]}. "
    "Every requested key must be present exactly once with a practical rules value "
    "that fits the world described in the lore. "
    "Output values in the requested language."
)


@dataclass(slots=True)
class _WorldConstitutionMutation:
    object_row: models.ObjectModel
    created: bool
    changed: bool
    patch_data: dict[str, Any]


def _ensure_lore_adaptation_enabled() -> None:
    if not USE_LORE_ADAPTATION:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Lore adaptation feature is disabled",
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_lore_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text(value: Any, *, max_chars: int = 2_000) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return _truncate_text(text, max_chars)


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _normalize_language(value: Any, *, fallback: str = "unknown") -> str:
    text = _normalize_text(value, max_chars=32)
    if not text:
        return fallback
    return text


def _normalize_world_model_code(value: Any, *, max_chars: int = 64) -> str:
    text = _normalize_text(value, max_chars=max_chars)
    if not text:
        return ""
    lowered = text.lower()
    ascii_code = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    if ascii_code:
        return ascii_code[:max_chars]
    return re.sub(r"\s+", "_", lowered).strip("_")[:max_chars]


def _normalize_world_model_list(
    raw_value: Any,
    *,
    max_items: int = _WORLD_MODEL_MAX_LIST_ITEMS,
    max_chars: int = 80,
) -> list[str]:
    if isinstance(raw_value, list):
        candidates = raw_value
    elif isinstance(raw_value, str):
        candidates = re.split(r"[\n,;/]+", raw_value)
    else:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        code = _normalize_world_model_code(item, max_chars=max_chars)
        if not code or code in seen:
            continue
        seen.add(code)
        normalized.append(code)
        if len(normalized) >= max_items:
            break
    return normalized


def _normalize_world_model_custom_axes(raw_value: Any) -> dict[str, str]:
    if not isinstance(raw_value, dict):
        return {}
    normalized: dict[str, str] = {}
    for raw_key, raw_item in raw_value.items():
        key = _normalize_world_model_code(raw_key, max_chars=48)
        value = _normalize_text(raw_item, max_chars=180)
        if not key or not value:
            continue
        normalized[key] = value
        if len(normalized) >= _WORLD_MODEL_MAX_CUSTOM_AXES:
            break
    return normalized


def _normalize_campaign_frame_hints(raw_value: Any) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        return {}
    normalized = {
        "power_fantasy": _normalize_world_model_code(raw_value.get("power_fantasy")),
        "tone_profile": _normalize_world_model_list(
            raw_value.get("tone_profile") or raw_value.get("tone_signals")
        ),
        "activity_bias": _normalize_world_model_list(
            raw_value.get("activity_bias") or raw_value.get("activity_focus")
        ),
    }
    return {key: value for key, value in normalized.items() if value}


def _extract_world_model_corpus(profile: dict[str, str]) -> str:
    parts: list[str] = []
    for raw_key, raw_value in profile.items():
        key = _normalize_text(raw_key, max_chars=120)
        value = _normalize_text(raw_value, max_chars=400)
        if key:
            parts.append(key.replace("_", " "))
        if value:
            parts.append(value)
    return " ".join(parts).lower()


def _infer_world_model_scalar(
    corpus: str,
    *,
    hint_value: str,
    keyword_map: list[tuple[str, tuple[str, ...]]],
    default: str,
) -> str:
    if hint_value:
        return hint_value
    for normalized_value, tokens in keyword_map:
        if any(token in corpus for token in tokens):
            return normalized_value
    return default


def _extract_world_model_flags(
    corpus: str,
    *,
    hinted_values: list[str],
    keyword_map: dict[str, tuple[str, ...]],
) -> list[str]:
    values = list(hinted_values)
    seen = set(values)
    for normalized_value, tokens in keyword_map.items():
        if normalized_value in seen:
            continue
        if any(token in corpus for token in tokens):
            values.append(normalized_value)
            seen.add(normalized_value)
    return values[:_WORLD_MODEL_MAX_LIST_ITEMS]


def _coerce_world_model_hints(raw_analysis: dict[str, Any]) -> dict[str, Any]:
    candidate = raw_analysis.get("world_model")
    if not isinstance(candidate, dict):
        candidate = raw_analysis.get("world_envelope")
    if not isinstance(candidate, dict):
        candidate = {}

    core_envelope = {
        "tech_level": _normalize_world_model_code(
            candidate.get("tech_level") or candidate.get("technology_level")
        ),
        "magic_level": _normalize_world_model_code(candidate.get("magic_level")),
        "mobility_profile": _normalize_world_model_list(
            candidate.get("mobility_profile") or candidate.get("mobility_modes")
        ),
        "conflict_profile": _normalize_world_model_list(
            candidate.get("conflict_profile") or candidate.get("conflict_modes")
        ),
        "social_structure": _normalize_world_model_list(
            candidate.get("social_structure") or candidate.get("institution_profile")
        ),
        "genre_signals": _normalize_world_model_list(candidate.get("genre_signals") or candidate.get("genre")),
        "tone_signals": _normalize_world_model_list(candidate.get("tone_signals") or candidate.get("tone")),
    }
    core_envelope = {key: value for key, value in core_envelope.items() if value}

    hints = {
        "expansion_policy": _normalize_world_model_code(
            candidate.get("expansion_policy") or raw_analysis.get("expansion_policy")
        )
        or "open_within_envelope",
        "core_envelope": core_envelope,
        "affordances": _normalize_world_model_list(
            candidate.get("affordances") or candidate.get("activity_affordances")
        ),
        "exceptions": _normalize_world_model_list(candidate.get("exceptions")),
        "forbidden_or_rare_elements": _normalize_world_model_list(
            candidate.get("forbidden_or_rare_elements") or candidate.get("forbidden_elements")
        ),
        "custom_axes": _normalize_world_model_custom_axes(candidate.get("custom_axes")),
        "campaign_frame": _normalize_campaign_frame_hints(
            candidate.get("campaign_frame")
            or {
                "power_fantasy": candidate.get("power_fantasy") or raw_analysis.get("power_fantasy"),
                "tone_profile": candidate.get("tone_profile") or candidate.get("tone_signals"),
                "activity_bias": candidate.get("activity_bias") or candidate.get("affordances"),
            }
        ),
    }
    return {key: value for key, value in hints.items() if value}


def _derive_world_model_affordances(
    *,
    tech_level: str,
    magic_level: str,
    mobility_profile: list[str],
    social_structure: list[str],
    hinted_affordances: list[str],
) -> list[str]:
    affordances = list(hinted_affordances)
    seen = set(affordances)

    def _add(value: str) -> None:
        if value in seen:
            return
        seen.add(value)
        affordances.append(value)

    if "guilds" in social_structure:
        _add("trade")
        _add("guild_politics")
        _add("apprenticeships")
    if "nobility" in social_structure or "great_houses" in social_structure:
        _add("court_intrigue")
        _add("duels")
        if tech_level in {"preindustrial", "industrial"}:
            _add("tournaments")
    if "academies" in social_structure:
        _add("research")
        _add("scholarship")
    if "churches" in social_structure or "cults" in social_structure:
        _add("rites")
        _add("pilgrimages")
    if magic_level not in {"", "unspecified", "no_magic"}:
        _add("rituals")
        _add("arcane_study")
    if "ship" in mobility_profile:
        _add("ports")
        _add("shipping")
    if "airship" in mobility_profile:
        _add("air_travel")
    if "rail" in mobility_profile:
        _add("rail_travel")
    if tech_level == "modern":
        _add("mass_media")
    if tech_level == "futuristic":
        _add("advanced_engineering")
    return affordances[:_WORLD_MODEL_MAX_LIST_ITEMS]


def _compile_session_world_model(
    *,
    language: str,
    profile: dict[str, str],
    world_model_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hints = dict(world_model_hints or {})
    core_hints = dict(hints.get("core_envelope") or {})
    corpus = _extract_world_model_corpus(profile)

    tech_level = _infer_world_model_scalar(
        corpus,
        hint_value=_normalize_world_model_code(core_hints.get("tech_level")),
        keyword_map=[
            ("futuristic", ("cyber", "android", "spacefaring", "orbital", "nanotech", "кибер", "косм", "андроид")),
            ("modern", ("telephone", "automobile", "radio", "internet", "smartphone", "автомоб", "телефон", "интернет", "радио")),
            ("industrial", ("steam", "steampunk", "diesel", "rail", "locomotive", "factory", "gunpowder", "clockwork", "engine", "паров", "поезд", "фабрик", "порох", "механизм")),
            ("preindustrial", ("medieval", "feudal", "kingdom", "sword", "horse", "castle", "средневек", "феод", "рыцар", "замок", "меч", "лошад")),
        ],
        default="unspecified",
    )
    magic_level = _infer_world_model_scalar(
        corpus,
        hint_value=_normalize_world_model_code(core_hints.get("magic_level")),
        keyword_map=[
            ("no_magic", ("no magic", "without magic", "magic is forbidden", "магии нет", "без магии", "магия запрещ")),
            ("high_magic", ("high magic", "common magic", "widespread magic", "arcane", "sorcery", "высокая магия", "магия повсемест", "колдов")),
            ("low_magic", ("low magic", "rare magic", "hidden magic", "низкая магия", "редкая магия", "скрытая магия")),
        ],
        default="unspecified",
    )

    mobility_profile = _extract_world_model_flags(
        corpus,
        hinted_values=_normalize_world_model_list(core_hints.get("mobility_profile")),
        keyword_map={
            "horseback": ("horse", "cavalry", "mounted", "лошад", "конн"),
            "carriage": ("carriage", "wagon", "cart", "телег", "повоз"),
            "ship": ("ship", "naval", "sail", "port", "кораб", "мор", "порт"),
            "portal": ("portal", "teleport", "gate", "портал", "телепорт"),
            "airship": ("airship", "dirigible", "дирижаб"),
            "rail": ("train", "rail", "locomotive", "поезд", "рельс"),
            "automobile": ("automobile", "car", "truck", "автомоб", "машин"),
        },
    )
    conflict_profile = _extract_world_model_flags(
        corpus,
        hinted_values=_normalize_world_model_list(core_hints.get("conflict_profile")),
        keyword_map={
            "duels": ("duel", "honor", "дуэл", "честь"),
            "court_intrigue": ("court", "intrigue", "politics", "двор", "интриг", "полит"),
            "ritual_magic": ("ritual", "sorcery", "ритуал", "колдов"),
            "gunpowder_warfare": ("gunpowder", "musket", "rifle", "порох", "мушкет", "руж"),
            "skirmishes": ("skirmish", "raider", "bandit", "схватк", "набег", "разбой"),
            "naval": ("naval", "fleet", "ship", "флот", "мор", "кораб"),
            "tournaments": ("tournament", "joust", "турнир", "рицар"),
        },
    )
    social_structure = _extract_world_model_flags(
        corpus,
        hinted_values=_normalize_world_model_list(core_hints.get("social_structure")),
        keyword_map={
            "guilds": ("guild", "craft", "merchant", "гильд", "цех", "торгов"),
            "nobility": ("nobility", "lords", "houses", "барон", "лорд", "дворян", "дом"),
            "clans": ("clan", "tribe", "клан", "плем"),
            "churches": ("church", "faith", "temple", "церк", "храм", "вера"),
            "cults": ("cult", "sect", "культ", "сект"),
            "academies": ("academy", "college", "university", "академ", "универс"),
            "corporations": ("corporation", "megacorp", "корпорац"),
        },
    )
    genre_signals = _extract_world_model_flags(
        corpus,
        hinted_values=_normalize_world_model_list(core_hints.get("genre_signals")),
        keyword_map={
            "fantasy": ("fantasy", "magic", "фэнтез", "маг"),
            "science_fiction": ("science fiction", "space", "sci fi", "научн", "косм"),
            "steampunk": ("steampunk", "паров"),
            "grimdark": ("grimdark", "bleak", "grim", "мрачн"),
            "mythic": ("mythic", "legend", "эпос", "миф"),
        },
    )
    tone_signals = _extract_world_model_flags(
        corpus,
        hinted_values=_normalize_world_model_list(core_hints.get("tone_signals")),
        keyword_map={
            "grounded": ("grounded", "realistic", "приземлен", "реалист"),
            "heroic": ("heroic", "heroism", "героич"),
            "grim": ("grim", "dark", "мрач"),
            "whimsical": ("whimsical", "playful", "сказоч", "игрив"),
            "political": ("political", "intrigue", "полит", "интриг"),
        },
    )

    affordances = _derive_world_model_affordances(
        tech_level=tech_level,
        magic_level=magic_level,
        mobility_profile=mobility_profile,
        social_structure=social_structure,
        hinted_affordances=_normalize_world_model_list(hints.get("affordances")),
    )

    compiled = {
        "version": _WORLD_MODEL_VERSION,
        "language": _normalize_language(language),
        "expansion_policy": _normalize_world_model_code(hints.get("expansion_policy")) or "open_within_envelope",
        "core_envelope": {
            "tech_level": tech_level,
            "magic_level": magic_level,
            "mobility_profile": mobility_profile,
            "conflict_profile": conflict_profile,
            "social_structure": social_structure,
            "genre_signals": genre_signals,
            "tone_signals": tone_signals,
        },
        "affordances": affordances,
        "exceptions": _normalize_world_model_list(hints.get("exceptions")),
        "forbidden_or_rare_elements": _normalize_world_model_list(hints.get("forbidden_or_rare_elements")),
        "custom_axes": _normalize_world_model_custom_axes(hints.get("custom_axes")),
        "campaign_frame": _normalize_campaign_frame_hints(hints.get("campaign_frame")),
        "source_profile_keys": sorted(
            key
            for key in (
                _normalize_text(raw_key, max_chars=120)
                for raw_key in profile.keys()
            )
            if key
        )[:_WORLD_MODEL_MAX_LIST_ITEMS],
    }
    compiled["core_envelope"] = {
        key: value for key, value in dict(compiled["core_envelope"]).items() if value
    }
    return {key: value for key, value in compiled.items() if value not in ("", [], {})}


def _build_gap_id(session_id: uuid.UUID, key: str) -> str:
    return f"{session_id}:{key}"


def _normalize_lore_claim_enum(value: Any, *, allowed: set[str], fallback: str) -> str:
    text = _normalize_text(value, max_chars=32).lower()
    if text in allowed:
        return text
    return fallback


def _normalize_lore_claim_confidence(value: Any, *, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _normalize_lore_claim_value(value: Any) -> str | list[str] | None:
    if isinstance(value, list):
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = _normalize_text(item, max_chars=120)
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized or None
    if isinstance(value, dict):
        rendered = _normalize_json_preview(value, max_chars=600)
        return rendered or None
    text = _normalize_text(value, max_chars=600)
    return text or None


def _coerce_lore_claim(
    raw_claim: Any,
    *,
    default_usage: str,
    default_scope: str,
    default_domain: str,
    default_criticality: str,
    fallback_confidence: float,
) -> LoreClaim | None:
    if not isinstance(raw_claim, dict):
        return None
    claim_key = _normalize_text(raw_claim.get("claim_key") or raw_claim.get("key"), max_chars=120)
    if not claim_key or not _VALID_KEY_RE.match(claim_key):
        return None
    label = _normalize_text(raw_claim.get("label"), max_chars=200) or claim_key.replace("_", " ").title()
    value = _normalize_lore_claim_value(raw_claim.get("value"))
    usage = _normalize_lore_claim_enum(
        raw_claim.get("usage"),
        allowed=_VALID_LORE_USAGES,
        fallback=default_usage,
    )
    scope = _normalize_lore_claim_enum(
        raw_claim.get("scope"),
        allowed=_VALID_LORE_SCOPES,
        fallback=default_scope,
    )
    domain = _normalize_lore_claim_enum(
        raw_claim.get("domain"),
        allowed=_VALID_LORE_DOMAINS,
        fallback=default_domain,
    )
    criticality = _normalize_lore_claim_enum(
        raw_claim.get("criticality"),
        allowed=_VALID_LORE_CRITICALITIES,
        fallback=default_criticality,
    )
    frame_signal = _normalize_lore_claim_enum(
        raw_claim.get("frame_signal"),
        allowed=_VALID_FRAME_SIGNALS,
        fallback="",
    )
    question = _normalize_text(raw_claim.get("question"), max_chars=320) or None
    missing = _normalize_bool(raw_claim.get("missing"))
    if value is None and not missing:
        missing = True
    if missing and not question:
        question = f"Please define '{label}' for this world."
    return LoreClaim(
        claim_key=claim_key,
        label=label,
        value=value,
        usage=usage,
        scope=scope,
        domain=domain,
        criticality=criticality,
        confidence=_normalize_lore_claim_confidence(raw_claim.get("confidence"), fallback=fallback_confidence),
        source_section=_normalize_text(raw_claim.get("source_section"), max_chars=160),
        source_excerpt=_normalize_text(raw_claim.get("source_excerpt"), max_chars=320),
        missing=missing,
        question=question,
        frame_signal=frame_signal or None,
    )


def _coerce_lore_claim_list(
    raw_claims: Any,
    *,
    default_usage: str,
    default_scope: str,
    default_domain: str,
    default_criticality: str,
    fallback_confidence: float,
    max_items: int = _MAX_LORE_CLAIM_COUNT,
) -> list[LoreClaim]:
    if not isinstance(raw_claims, list):
        return []
    claims: list[LoreClaim] = []
    seen: set[str] = set()
    for raw_claim in raw_claims:
        if len(claims) >= max_items:
            break
        claim = _coerce_lore_claim(
            raw_claim,
            default_usage=default_usage,
            default_scope=default_scope,
            default_domain=default_domain,
            default_criticality=default_criticality,
            fallback_confidence=fallback_confidence,
        )
        if claim is None or claim.claim_key in seen:
            continue
        seen.add(claim.claim_key)
        claims.append(claim)
    return claims


def _synthesize_structured_claims_from_world_model(
    *,
    existing_claims: list[LoreClaim],
    world_model_hints: dict[str, Any],
) -> list[LoreClaim]:
    claims = list(existing_claims)
    seen_keys = {claim.claim_key for claim in claims}
    campaign_frame = dict(world_model_hints.get("campaign_frame") or {})

    def _add_claim(claim: LoreClaim) -> None:
        if claim.claim_key in seen_keys:
            return
        seen_keys.add(claim.claim_key)
        claims.append(claim)

    power_fantasy = _normalize_text(campaign_frame.get("power_fantasy"), max_chars=80)
    if power_fantasy:
        _add_claim(
            LoreClaim(
                claim_key="campaign_power_fantasy",
                label="Campaign Power Fantasy",
                value=power_fantasy,
                usage="campaign",
                scope="campaign",
                domain="progression",
                criticality="important",
                confidence=0.85,
                source_section="campaign_frame",
                frame_signal="power_fantasy",
            )
        )

    tone_profile = claim_value_list(campaign_frame.get("tone_profile"))
    if tone_profile:
        _add_claim(
            LoreClaim(
                claim_key="campaign_tone_profile",
                label="Campaign Tone Profile",
                value=tone_profile,
                usage="campaign",
                scope="campaign",
                domain="social",
                criticality="important",
                confidence=0.8,
                source_section="campaign_frame",
                frame_signal="tone_profile",
            )
        )

    activity_bias = claim_value_list(campaign_frame.get("activity_bias"))
    if activity_bias:
        _add_claim(
            LoreClaim(
                claim_key="campaign_activity_bias",
                label="Campaign Activity Bias",
                value=activity_bias,
                usage="campaign",
                scope="campaign",
                domain="progression",
                criticality="important",
                confidence=0.8,
                source_section="campaign_frame",
                frame_signal="activity_bias",
            )
        )

    for exception in claim_value_list(world_model_hints.get("exceptions"))[: LORE_UX_POLICY.max_special_cases]:
        normalized_key = _normalize_world_model_code(exception, max_chars=60)
        if not normalized_key:
            continue
        _add_claim(
            LoreClaim(
                claim_key=f"special_case_{normalized_key}",
                label=exception.replace("_", " ").title(),
                value=exception,
                usage="special_case",
                scope="world",
                domain="progression",
                criticality="important",
                confidence=0.7,
                source_section="world_model.exceptions",
            )
        )

    return claims


def _build_gap_from_claim(session_id: uuid.UUID, claim: LoreClaim) -> dict[str, Any] | None:
    if not claim.missing:
        return None
    return {
        "gap_id": _build_gap_id(session_id, claim.claim_key),
        "key": claim.claim_key,
        "label": claim.label,
        "question": _normalize_text(claim.question, max_chars=320),
        "relevant": True,
        "missing": True,
        "critical": False,
        "status": "open",
        "answer": None,
        "source": "llm",
        "usage": claim.usage,
        "scope": claim.scope,
        "domain": claim.domain,
        "criticality": claim.criticality,
        "confidence": claim.confidence,
        "blocks_turns": False,
        "why_this_blocks_turns": None,
    }


def _build_gap_list_from_claims(
    *,
    session_id: uuid.UUID,
    claims: list[LoreClaim],
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for claim in claims:
        gap = _build_gap_from_claim(session_id, claim)
        if gap is not None:
            gaps.append(gap)
    return gaps


def _claims_to_draft_profile(claims: list[LoreClaim]) -> dict[str, str]:
    profile: dict[str, str] = {}
    for claim in claims:
        if not claim_has_resolved_value(claim):
            continue
        rendered_value = claim_value_text(claim.value, max_chars=600)
        if rendered_value:
            profile[claim.claim_key] = rendered_value
    return profile


def _coerce_stored_claims(raw_claims: Any) -> list[LoreClaim]:
    return deserialize_claims(raw_claims)


def _annotate_gaps_for_output(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocking_candidates = gap_turn_block_candidates(gaps, policy=LORE_UX_POLICY)
    blocking_ids = {
        _normalize_text(gap.get("gap_id"), max_chars=200)
        for gap in blocking_candidates[: LORE_UX_POLICY.max_blocking_questions]
    }
    annotated: list[dict[str, Any]] = []
    for raw_gap in gaps:
        if not isinstance(raw_gap, dict):
            continue
        gap = dict(raw_gap)
        gap_id = _normalize_text(gap.get("gap_id"), max_chars=200)
        is_open = str(gap.get("status") or "open").strip().lower() == "open"
        blocks_turns = bool(gap_id) and is_open and gap_id in blocking_ids
        gap["usage"] = _normalize_lore_claim_enum(
            gap.get("usage"),
            allowed=_VALID_LORE_USAGES,
            fallback="runtime" if _normalize_bool(gap.get("critical")) else "reference",
        )
        gap["scope"] = _normalize_lore_claim_enum(
            gap.get("scope"),
            allowed=_VALID_LORE_SCOPES,
            fallback="world",
        )
        gap["domain"] = _normalize_lore_claim_enum(
            gap.get("domain"),
            allowed=_VALID_LORE_DOMAINS,
            fallback="progression",
        )
        gap["criticality"] = _normalize_lore_claim_enum(
            gap.get("criticality"),
            allowed=_VALID_LORE_CRITICALITIES,
            fallback="blocking" if _normalize_bool(gap.get("critical")) else "optional",
        )
        gap["confidence"] = _normalize_lore_claim_confidence(
            gap.get("confidence"),
            fallback=1.0 if _normalize_bool(gap.get("critical")) else 0.0,
        )
        gap["critical"] = blocks_turns
        gap["blocks_turns"] = blocks_turns
        gap["why_this_blocks_turns"] = (
            "This question defines gameplay-critical runtime lore for turns."
            if blocks_turns
            else None
        )
        annotated.append(gap)
    return annotated


def _collect_open_gap_ids(gaps: list[dict[str, Any]]) -> list[str]:
    open_gap_ids: list[str] = []
    for gap in gaps:
        if str(gap.get("status") or "open").strip().lower() != "open":
            continue
        gap_id = _normalize_text(gap.get("gap_id"), max_chars=200)
        if gap_id:
            open_gap_ids.append(gap_id)
    return open_gap_ids


def _get_world_constitution(
    db: Session,
    session_id: uuid.UUID,
    *,
    for_update: bool = False,
) -> models.ObjectModel | None:
    query = (
        select(models.ObjectModel)
        .where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == "world_constitution",
        )
        .order_by(models.ObjectModel.created_at.desc())
        .limit(1)
    )
    if for_update:
        query = query.with_for_update()
    return db.execute(query).scalar_one_or_none()


def _extract_finalized_lore_profile(world_constitution_row: models.ObjectModel | None) -> dict[str, Any] | None:
    if world_constitution_row is None:
        return None
    data = dict(world_constitution_row.data or {})
    lore_profile = data.get("lore_profile")
    if not isinstance(lore_profile, dict):
        return None
    return dict(lore_profile)


def _extract_finalized_lore_meta(world_constitution_row: models.ObjectModel | None) -> dict[str, Any]:
    if world_constitution_row is None:
        return {}
    data = dict(world_constitution_row.data or {})
    raw_meta = data.get("lore_profile_meta")
    if not isinstance(raw_meta, dict):
        return {}
    return dict(raw_meta)


def _extract_lore_state(session_row: models.SessionModel) -> dict[str, Any]:
    raw_state = dict(session_row.state_json or {})
    candidate = raw_state.get(_LORE_STATE_KEY)
    if isinstance(candidate, dict):
        return dict(candidate)
    return {}


def _lock_lore_session_for_mutation(
    db: Session,
    session_id: uuid.UUID,
) -> models.SessionModel:
    _acquire_session_turn_lock(db, session_id)
    locked_session = _require_session(db, session_id, for_update=True)
    _recover_abandoned_pending_turn_locked(
        db=db,
        session_id=session_id,
        session_row=locked_session,
        state_payload=dict(locked_session.state_json or {}),
    )
    return locked_session


def _persist_lore_state(session_row: models.SessionModel, lore_state: dict[str, Any]) -> None:
    state_payload = dict(session_row.state_json or {})
    state_payload[_LORE_STATE_KEY] = lore_state
    session_row.state_json = state_payload


def _lore_status(lore_state: dict[str, Any]) -> str:
    status_value = str(lore_state.get("status") or "").strip().lower()
    if status_value in _VALID_LORE_STATUSES:
        return status_value
    return "idle"


def _extract_processing_request_id(lore_state: dict[str, Any]) -> str:
    return _normalize_text(lore_state.get(_LORE_PROCESSING_REQUEST_ID_KEY), max_chars=64)


def _extract_stable_lore_state(lore_state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(lore_state, dict):
        return {}
    status_value = _lore_status(lore_state)
    if status_value == "processing":
        candidate = lore_state.get(_LORE_STABLE_STATE_KEY)
        if isinstance(candidate, dict):
            return _extract_stable_lore_state(candidate)
        return {}

    stable_state = dict(lore_state)
    stable_state.pop(_LORE_PROCESSING_REQUEST_ID_KEY, None)
    stable_state.pop(_LORE_STABLE_STATE_KEY, None)
    stable_state.pop("mode", None)
    return stable_state


def _build_processing_lore_state(
    *,
    uploaded_lore_hash: str,
    lore_text: str,
    mode: str,
    stable_state: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    return {
        "status": "processing",
        "uploaded_lore_hash": uploaded_lore_hash,
        "language": "unknown",
        "gaps": [],
        "source_text": lore_text,
        "mode": mode if mode in {"interactive", "auto_fill"} else "interactive",
        _LORE_PROCESSING_REQUEST_ID_KEY: request_id,
        _LORE_STABLE_STATE_KEY: _extract_stable_lore_state(stable_state),
        "updated_at": _now_iso(),
    }


def _is_current_processing_job(
    lore_state: dict[str, Any],
    *,
    request_id: str,
    uploaded_lore_hash: str | None = None,
) -> bool:
    if _lore_status(lore_state) != "processing":
        return False
    if _extract_processing_request_id(lore_state) != request_id:
        return False
    if uploaded_lore_hash is None:
        return True
    current_hash = _normalize_text(lore_state.get("uploaded_lore_hash"), max_chars=128)
    return bool(current_hash) and current_hash == uploaded_lore_hash


def _reset_implicit_tx_if_needed(db: Session, *, had_active_tx: bool) -> None:
    if not had_active_tx and db.in_transaction():
        db.rollback()


def _coerce_gap(raw_gap: Any, *, session_id: uuid.UUID) -> dict[str, Any] | None:
    if not isinstance(raw_gap, dict):
        return None
    key = _normalize_text(raw_gap.get("key"), max_chars=120)
    if not key or not _VALID_KEY_RE.match(key):
        return None
    label = _normalize_text(raw_gap.get("label"), max_chars=200)
    if not label:
        label = key.replace("_", " ").title()
    question = _normalize_text(raw_gap.get("question"), max_chars=300)
    relevant = True if raw_gap.get("relevant") is None else _normalize_bool(raw_gap.get("relevant"))
    missing = _normalize_bool(raw_gap.get("missing"))
    status_value = str(raw_gap.get("status") or "").strip().lower()
    status_normalized = "resolved" if status_value == "resolved" else "open"
    answer = _normalize_text(raw_gap.get("answer"), max_chars=2_000) or None
    source_value = str(raw_gap.get("source") or "").strip().lower()
    source_normalized = source_value if source_value in {"llm", "player", "ai"} else "llm"
    gap_id = _normalize_text(raw_gap.get("gap_id"), max_chars=200)
    if not gap_id:
        gap_id = _build_gap_id(session_id, key)
    usage = _normalize_lore_claim_enum(
        raw_gap.get("usage"),
        allowed=_VALID_LORE_USAGES,
        fallback="runtime" if _normalize_bool(raw_gap.get("critical")) else "reference",
    )
    scope = _normalize_lore_claim_enum(
        raw_gap.get("scope"),
        allowed=_VALID_LORE_SCOPES,
        fallback="world",
    )
    domain = _normalize_lore_claim_enum(
        raw_gap.get("domain"),
        allowed=_VALID_LORE_DOMAINS,
        fallback="progression",
    )
    criticality = _normalize_lore_claim_enum(
        raw_gap.get("criticality"),
        allowed=_VALID_LORE_CRITICALITIES,
        fallback="blocking" if _normalize_bool(raw_gap.get("critical")) else "optional",
    )
    confidence = _normalize_lore_claim_confidence(
        raw_gap.get("confidence"),
        fallback=1.0 if _normalize_bool(raw_gap.get("critical")) else 0.0,
    )
    return {
        "gap_id": gap_id,
        "key": key,
        "label": label,
        "question": question,
        "relevant": relevant,
        "missing": missing,
        "critical": False,
        "status": status_normalized,
        "answer": answer,
        "source": source_normalized,
        "usage": usage,
        "scope": scope,
        "domain": domain,
        "criticality": criticality,
        "confidence": confidence,
        "blocks_turns": False,
        "why_this_blocks_turns": None,
    }


def _coerce_gap_list(raw_gaps: Any, *, session_id: uuid.UUID) -> list[dict[str, Any]]:
    if not isinstance(raw_gaps, list):
        return []
    coerced: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_gap in raw_gaps:
        gap = _coerce_gap(raw_gap, session_id=session_id)
        if gap is None:
            continue
        gap_id = str(gap.get("gap_id") or "")
        if not gap_id or gap_id in seen:
            continue
        seen.add(gap_id)
        coerced.append(gap)
    return coerced


def _collect_unresolved_critical_gap_ids(gaps: list[dict[str, Any]]) -> list[str]:
    return [
        _normalize_text(gap.get("gap_id"), max_chars=200)
        for gap in _annotate_gaps_for_output(gaps)
        if _normalize_bool(gap.get("blocks_turns"))
        and _normalize_text(gap.get("gap_id"), max_chars=200)
    ]


def _extract_finalized_compiled_world_model(
    world_constitution_row: models.ObjectModel | None,
) -> dict[str, Any]:
    if world_constitution_row is None:
        return {}
    data = dict(world_constitution_row.data or {})
    compiled_world_model = data.get("compiled_world_model")
    if not isinstance(compiled_world_model, dict):
        return {}
    return dict(compiled_world_model)


def _build_response_payload(
    *,
    session_id: uuid.UUID,
    lore_state: dict[str, Any],
    world_constitution_row: models.ObjectModel | None,
) -> dict[str, Any]:
    has_finalized_profile = _extract_finalized_lore_profile(world_constitution_row) is not None
    finalized_meta = _extract_finalized_lore_meta(world_constitution_row)
    status_value = str(lore_state.get("status") or "").strip().lower()
    if status_value not in _VALID_LORE_STATUSES:
        status_value = "finalized" if has_finalized_profile else "idle"

    language = _normalize_language(
        lore_state.get("language") or finalized_meta.get("language"),
        fallback="unknown",
    )
    uploaded_hash = _normalize_text(
        lore_state.get("uploaded_lore_hash") or finalized_meta.get("uploaded_lore_hash"),
        max_chars=128,
    ) or None

    raw_gaps = _coerce_gap_list(lore_state.get("gaps"), session_id=session_id)
    gaps = _annotate_gaps_for_output(raw_gaps)
    unresolved_critical = _collect_unresolved_critical_gap_ids(gaps)

    state_claims = _coerce_stored_claims(lore_state.get("claims"))
    finalized_claims = claims_from_world_constitution_data(
        dict(world_constitution_row.data or {}) if world_constitution_row is not None else {}
    )
    claims = state_claims or finalized_claims
    if not claims:
        raw_profile = lore_state.get("draft_profile") or _extract_finalized_lore_profile(world_constitution_row) or {}
        if isinstance(raw_profile, dict):
            claims = [
                LoreClaim(
                    claim_key=key,
                    label=key.replace("_", " ").title(),
                    value=_normalize_text(value, max_chars=600),
                    usage="runtime",
                    scope="world",
                    domain="progression",
                    criticality="important",
                    confidence=0.55,
                )
                for key, value in raw_profile.items()
                if _normalize_text(key, max_chars=120) and _normalize_text(value, max_chars=600)
            ]

    draft_profile = _claims_to_draft_profile(claims)
    compiled_world_model = _extract_finalized_compiled_world_model(world_constitution_row)
    if not compiled_world_model and (draft_profile or dict(lore_state.get(_WORLD_MODEL_HINTS_KEY) or {})):
        compiled_world_model = _compile_session_world_model(
            language=language,
            profile=draft_profile,
            world_model_hints=dict(lore_state.get(_WORLD_MODEL_HINTS_KEY) or {}),
        )

    preview = build_lore_preview(
        claims,
        gaps,
        compiled_world_model=compiled_world_model,
        policy=LORE_UX_POLICY,
    )
    coverage_summary = build_lore_coverage_summary(claims, gaps, policy=LORE_UX_POLICY)
    blocking_summary = build_lore_blocking_summary(status_value, gaps, policy=LORE_UX_POLICY)
    runtime_ready = not blocking_summary.get("blocked") and status_value != "processing"
    complete = status_value != "processing" and not _collect_open_gap_ids(gaps)

    if status_value == "finalized":
        gaps = []
        unresolved_critical = []

    return {
        "session_id": session_id,
        "status": status_value,
        "language": language,
        "uploaded_lore_hash": uploaded_hash,
        "complete": complete,
        "runtime_ready": runtime_ready,
        "has_finalized_profile": has_finalized_profile,
        "unresolved_critical_gap_ids": unresolved_critical,
        "gaps": gaps,
        "preview": preview,
        "coverage_summary": coverage_summary,
        "blocking_summary": blocking_summary,
        "compiled_world_model_preview": build_compiled_world_model_preview(
            compiled_world_model,
            policy=LORE_UX_POLICY,
        ),
        "draft_profile_preview": build_lore_draft_profile_preview(claims, policy=LORE_UX_POLICY),
    }


def _build_session_lore_payload(
    *,
    session_id: uuid.UUID,
    lore_state: dict[str, Any],
    world_constitution_row: models.ObjectModel | None,
) -> dict[str, Any]:
    if not lore_state:
        default_state: dict[str, Any] = {
            "status": "finalized" if _extract_finalized_lore_profile(world_constitution_row) is not None else "idle",
            "uploaded_lore_hash": _extract_finalized_lore_meta(world_constitution_row).get("uploaded_lore_hash"),
            "language": _extract_finalized_lore_meta(world_constitution_row).get("language"),
            "gaps": [],
            "updated_at": _now_iso(),
        }
        return _build_response_payload(
            session_id=session_id,
            lore_state=default_state,
            world_constitution_row=world_constitution_row,
        )

    return _build_response_payload(
        session_id=session_id,
        lore_state=lore_state,
        world_constitution_row=world_constitution_row,
    )


def _coerce_lore_analysis(
    *,
    session_id: uuid.UUID,
    raw_analysis: dict[str, Any],
) -> tuple[str, list[LoreClaim], list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    language = _normalize_language(raw_analysis.get("language"), fallback="unknown")
    world_model_hints = _coerce_world_model_hints(raw_analysis)
    claims = _coerce_lore_claim_list(
        raw_analysis.get("claims"),
        default_usage="reference",
        default_scope="world",
        default_domain="progression",
        default_criticality="optional",
        fallback_confidence=0.55,
    )
    if not claims:
        claims = _coerce_lore_claim_list(
            raw_analysis.get("mechanics"),
            default_usage="runtime",
            default_scope="world",
            default_domain="progression",
            default_criticality="blocking",
            fallback_confidence=0.75,
        )
    claims = _synthesize_structured_claims_from_world_model(
        existing_claims=claims,
        world_model_hints=world_model_hints,
    )
    gaps = _build_gap_list_from_claims(session_id=session_id, claims=claims)
    draft_profile = _claims_to_draft_profile(claims)
    return language, claims, gaps, draft_profile, world_model_hints


def _call_lore_analysis(lore_text: str) -> dict[str, Any]:
    user_payload = {
        "lore_text": lore_text,
        "instructions": {
            "language": "Detect lore language and output all questions in that same language.",
            "discovery": (
                "Read the lore and discover normalized gameplay claims this world needs. "
                "Do not use a fixed checklist. Classify each claim into usage, scope, domain, "
                "criticality, and confidence."
            ),
            "output": (
                "For each claim: generate a snake_case claim_key, a label, and classify it into "
                "usage/scope/domain/criticality. Mark missing=true if the lore does not define it "
                "clearly, and if missing write a concrete question for the player."
            ),
        },
    }
    with telemetry_context(request_type="lore_analysis"):
        result = openrouter_chat.generate_json(
            model=OPENROUTER_CHAT_MODEL,
            system_prompt=_LORE_ANALYSIS_SYSTEM,
            user_prompt=_normalize_json_preview(user_payload, max_chars=20_000),
            max_tokens=2_200,
            timeout_seconds=float(LORE_ADAPTATION_TIMEOUT_SECONDS),
        )
    if not isinstance(result, dict):
        raise RuntimeError("lore analysis did not return a JSON object")
    return result


def _call_lore_gap_fill(
    *,
    lore_text: str,
    language: str,
    draft_profile: dict[str, str],
    keys: list[str],
) -> dict[str, str]:
    requested_keys = [key for key in keys if key and _VALID_KEY_RE.match(key)]
    if not requested_keys:
        return {}
    user_payload = {
        "lore_text": lore_text,
        "language": language,
        "existing_profile": draft_profile,
        "requested_keys": requested_keys,
    }
    with telemetry_context(request_type="lore_gap_fill"):
        result = openrouter_chat.generate_json(
            model=OPENROUTER_CHAT_MODEL,
            system_prompt=_LORE_FILL_SYSTEM,
            user_prompt=_normalize_json_preview(user_payload, max_chars=20_000),
            max_tokens=1_400,
            timeout_seconds=float(LORE_ADAPTATION_TIMEOUT_SECONDS),
        )
    fills: dict[str, str] = {}
    if not isinstance(result, dict):
        return fills
    raw_fills = result.get("fills")
    if not isinstance(raw_fills, list):
        return fills
    for raw_item in raw_fills:
        if not isinstance(raw_item, dict):
            continue
        key = _normalize_text(raw_item.get("key"), max_chars=120)
        if key not in requested_keys:
            continue
        value = _normalize_text(raw_item.get("value"), max_chars=2_000)
        if not value:
            continue
        fills[key] = value
    return fills


def _is_timeout_like_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc).strip().lower()
    return "timeout" in message or "exhausted retries" in message


def _upsert_world_constitution_profile(
    *,
    db: Session,
    session_id: uuid.UUID,
    profile: dict[str, str],
    uploaded_lore_hash: str,
    language: str,
    compiled_world_model: dict[str, Any],
    lore_claims: list[LoreClaim] | None = None,
) -> _WorldConstitutionMutation:
    profile_payload = {str(key): _normalize_text(value, max_chars=2_000) for key, value in profile.items() if value}
    meta_payload = {
        "uploaded_lore_hash": uploaded_lore_hash,
        "language": _normalize_language(language),
        "finalized_at": _now_iso(),
        "version": _WORLD_MODEL_VERSION,
    }
    serialized_claims = [
        serialize_claim(claim)
        for claim in (lore_claims or [])
        if claim_has_resolved_value(claim)
    ]
    merged_payload = {
        "lore_profile": profile_payload,
        "lore_profile_meta": meta_payload,
        "compiled_world_model": dict(compiled_world_model or {}),
        "lore_claims": serialized_claims,
    }

    world_constitution_row = _get_world_constitution(db, session_id, for_update=True)
    created = world_constitution_row is None
    if world_constitution_row is not None:
        previous_data = dict(world_constitution_row.data or {})
        merged_data = dict(previous_data)
        merged_data["lore_profile"] = profile_payload
        merged_data["lore_profile_meta"] = meta_payload
        merged_data["compiled_world_model"] = dict(compiled_world_model or {})
        merged_data["lore_claims"] = serialized_claims
        changed = merged_data != previous_data
        if changed:
            world_constitution_row.data = merged_data
        return _WorldConstitutionMutation(
            object_row=world_constitution_row,
            created=False,
            changed=changed,
            patch_data=merged_payload,
        )

    bind = db.get_bind()
    dialect_name = str(getattr(getattr(bind, "dialect", None), "name", "")).lower()
    if dialect_name == "postgresql":
        insert_stmt = pg_insert(models.ObjectModel).values(
            session_id=session_id,
            object_id=uuid.uuid4(),
            type="world_constitution",
            name="World Constitution",
            data=merged_payload,
        )
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=["session_id"],
            index_where=text("type = 'world_constitution'"),
            set_={
                "data": models.ObjectModel.data.op("||")(insert_stmt.excluded.data),
                "name": insert_stmt.excluded.name,
            },
        )
        db.execute(upsert_stmt)
        world_constitution_row = _get_world_constitution(db, session_id, for_update=True)
        if world_constitution_row is None:
            raise RuntimeError(
                f"world_constitution upsert failed to materialize for session_id={session_id}"
            )
        return _WorldConstitutionMutation(
            object_row=world_constitution_row,
            created=created,
            changed=True,
            patch_data=merged_payload,
        )

    world_constitution_row = models.ObjectModel(
        session_id=session_id,
        type="world_constitution",
        name="World Constitution",
        data=merged_payload,
    )
    db.add(world_constitution_row)
    db.flush()
    return _WorldConstitutionMutation(
        object_row=world_constitution_row,
        created=created,
        changed=True,
        patch_data=merged_payload,
    )


def _record_world_constitution_mutation(
    *,
    db: Session,
    locked_session: models.SessionModel,
    session_id: uuid.UUID,
    mutation: _WorldConstitutionMutation,
    uploaded_lore_hash: str,
    language: str,
) -> None:
    if not mutation.changed:
        return

    from . import crud_entities as _entities

    in_game_day, in_game_minute = _extract_in_game_time(dict(locked_session.state_json or {}))
    turn_index = max(int(getattr(locked_session, "turn_index", 0) or 0), 0) + 1
    object_row = mutation.object_row
    if mutation.created:
        applied_ops = [
            {
                "op": "object.create",
                "ref": str(object_row.object_id),
                "type": "world_constitution",
                "name": str(getattr(object_row, "name", "") or "World Constitution"),
                "data": dict(object_row.data or {}),
            }
        ]
        ai_text = "(lore) created world constitution"
    else:
        applied_ops = [
            {
                "op": "object.update",
                "object": str(object_row.object_id),
                "patch": dict(mutation.patch_data),
            }
        ]
        ai_text = "(lore) finalized world constitution"

    _create_internal_turn_row(
        db,
        session_id,
        locked_session,
        turn_index=turn_index,
        user_input=_truncate_text(
            f"[internal lore.finalize {uploaded_lore_hash}]",
            240,
        ),
        ai_text=ai_text,
        note="lore_finalization",
        applied_ops=applied_ops,
        in_game_day=in_game_day,
        in_game_minute=in_game_minute,
        extra_ai_json={
            "uploaded_lore_hash": uploaded_lore_hash,
            "language": language,
        },
    )
    if mutation.created:
        _entities._add_internal_object_created_event(
            db,
            session_id=session_id,
            turn_index=turn_index,
            object_row=object_row,
            object_data=dict(object_row.data or {}),
        )
    else:
        _entities._add_internal_object_updated_event(
            db,
            session_id=session_id,
            turn_index=turn_index,
            object_row=object_row,
            patch_data=dict(mutation.patch_data),
        )
    _entities._enqueue_turn_chronicle_sync_event(
        db,
        session_id=session_id,
        turn_index=turn_index,
    )


def _finalized_lore_state(*, uploaded_lore_hash: str, language: str) -> dict[str, Any]:
    return {
        "status": "finalized",
        "uploaded_lore_hash": uploaded_lore_hash,
        "language": _normalize_language(language),
        "gaps": [],
        "updated_at": _now_iso(),
    }


def _apply_answers_to_state(
    *,
    state: dict[str, Any],
    answers: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    next_state = dict(state)
    claims = _coerce_stored_claims(next_state.get("claims"))
    if claims:
        updated_claims: list[dict[str, Any]] = []
        for claim in claims:
            if claim.claim_key in answers:
                value, _source = answers[claim.claim_key]
                updated_claims.append(
                    serialize_claim(
                        LoreClaim(
                            claim_key=claim.claim_key,
                            label=claim.label,
                            value=_normalize_text(value, max_chars=600),
                            usage=claim.usage,
                            scope=claim.scope,
                            domain=claim.domain,
                            criticality=claim.criticality,
                            confidence=claim.confidence,
                            source_section=claim.source_section,
                            source_excerpt=claim.source_excerpt,
                            missing=False,
                            question=claim.question,
                            frame_signal=claim.frame_signal,
                        )
                    )
                )
            else:
                updated_claims.append(serialize_claim(claim))
        next_state["claims"] = updated_claims

    # Note: session-scoped gap IDs are already stable in stored state, so we mutate by key.
    raw_gaps = next_state.get("gaps")
    if not isinstance(raw_gaps, list):
        raw_gaps = []
    updated_gaps: list[dict[str, Any]] = []
    for raw_gap in raw_gaps:
        if not isinstance(raw_gap, dict):
            continue
        gap = dict(raw_gap)
        key = _normalize_text(gap.get("key"), max_chars=120)
        if key in answers:
            value, source = answers[key]
            gap["answer"] = value
            gap["status"] = "resolved"
            gap["missing"] = False
            gap["critical"] = False
            gap["blocks_turns"] = False
            gap["why_this_blocks_turns"] = None
            gap["source"] = source
        updated_gaps.append(gap)
    next_state["gaps"] = updated_gaps
    refreshed_claims = _coerce_stored_claims(next_state.get("claims"))
    if refreshed_claims:
        next_state["draft_profile"] = _claims_to_draft_profile(refreshed_claims)
    else:
        draft_profile = dict(next_state.get("draft_profile") or {})
        for key, (value, _) in answers.items():
            draft_profile[key] = _normalize_text(value, max_chars=2_000)
        next_state["draft_profile"] = draft_profile
    next_state["updated_at"] = _now_iso()
    return next_state


def _stale_lore_conflict(*, detail: str = "Lore adaptation state changed; refresh and retry") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "code": "lore_adaptation_stale",
            "message": detail,
        },
    )


def _read_processing_job_snapshot(
    db: Session,
    *,
    session_id: uuid.UUID,
    request_id: str,
) -> dict[str, str] | None:
    session_row = _require_session(db, session_id)
    lore_state = _extract_lore_state(session_row)
    if not _is_current_processing_job(lore_state, request_id=request_id):
        return None
    lore_text = _normalize_text(lore_state.get("source_text"), max_chars=LORE_ADAPTATION_MAX_CHARS)
    uploaded_lore_hash = _normalize_text(lore_state.get("uploaded_lore_hash"), max_chars=128)
    if not lore_text or not uploaded_lore_hash:
        return None
    mode = str(lore_state.get("mode") or "interactive").strip().lower()
    if mode not in {"interactive", "auto_fill"}:
        mode = "interactive"
    return {
        "lore_text": lore_text,
        "uploaded_lore_hash": uploaded_lore_hash,
        "mode": mode,
    }


def _build_working_lore_state(
    *,
    session_id: uuid.UUID,
    uploaded_lore_hash: str,
    lore_text: str,
    mode: str,
) -> dict[str, Any]:
    raw_analysis = _call_lore_analysis(lore_text)
    language, claims, gaps, draft_profile, world_model_hints = _coerce_lore_analysis(
        session_id=session_id,
        raw_analysis=raw_analysis,
    )
    working_state: dict[str, Any] = {
        "status": "awaiting_answers",
        "uploaded_lore_hash": uploaded_lore_hash,
        "language": language,
        "claims": [serialize_claim(claim) for claim in claims],
        "gaps": gaps,
        "draft_profile": draft_profile,
        _WORLD_MODEL_HINTS_KEY: world_model_hints,
        "source_text": lore_text,
        "updated_at": _now_iso(),
    }

    open_keys = [
            _normalize_text(item.get("key"), max_chars=120)
            for item in gaps
            if isinstance(item, dict) and str(item.get("status") or "open").strip().lower() != "resolved"
        ]
    open_keys = [key for key in open_keys if key]
    if mode == "auto_fill" and open_keys:
        try:
            ai_fills = _call_lore_gap_fill(
                lore_text=lore_text,
                language=language,
                draft_profile=draft_profile,
                keys=open_keys,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Lore upload auto-fill degraded to manual answers for session_id=%s",
                session_id,
                exc_info=True,
            )
            ai_fills = {}
        answers = {key: (value, "ai") for key, value in ai_fills.items() if value}
        if answers:
            working_state = _apply_answers_to_state(state=working_state, answers=answers)

    return working_state


def _persist_completed_lore_state(
    *,
    db: Session,
    locked_session: models.SessionModel,
    session_id: uuid.UUID,
    lore_state: dict[str, Any],
) -> dict[str, Any]:
    normalized_state = dict(lore_state)
    normalized_state["status"] = "awaiting_answers"
    normalized_state.pop(_LORE_PROCESSING_REQUEST_ID_KEY, None)
    normalized_state.pop(_LORE_STABLE_STATE_KEY, None)
    normalized_state.pop("mode", None)

    updated_gaps = _annotate_gaps_for_output(
        _coerce_gap_list(normalized_state.get("gaps"), session_id=session_id)
    )
    unresolved_critical = _collect_unresolved_critical_gap_ids(updated_gaps)
    open_gap_ids = _collect_open_gap_ids(updated_gaps)
    normalized_state["gaps"] = updated_gaps
    claims = _coerce_stored_claims(normalized_state.get("claims"))
    if not claims and dict(normalized_state.get("draft_profile") or {}):
        claims = [
            LoreClaim(
                claim_key=key,
                label=key.replace("_", " ").title(),
                value=_normalize_text(value, max_chars=600),
                usage="runtime",
                scope="world",
                domain="progression",
                criticality="important",
                confidence=0.55,
            )
            for key, value in dict(normalized_state.get("draft_profile") or {}).items()
            if _normalize_text(key, max_chars=120) and _normalize_text(value, max_chars=600)
        ]
        normalized_state["claims"] = [serialize_claim(claim) for claim in claims]

    if not unresolved_critical:
        profile_payload = {
            str(key): _normalize_text(value, max_chars=2_000)
            for key, value in dict(normalized_state.get("draft_profile") or {}).items()
            if _normalize_text(value, max_chars=2_000)
        }
        uploaded_lore_hash = _normalize_text(normalized_state.get("uploaded_lore_hash"), max_chars=128)
        language = _normalize_language(normalized_state.get("language"), fallback="unknown")
        compiled_world_model = _compile_session_world_model(
            language=language,
            profile=profile_payload,
            world_model_hints=dict(normalized_state.get(_WORLD_MODEL_HINTS_KEY) or {}),
        )
        mutation = _upsert_world_constitution_profile(
            db=db,
            session_id=session_id,
            profile=profile_payload,
            uploaded_lore_hash=uploaded_lore_hash,
            language=language,
            compiled_world_model=compiled_world_model,
            lore_claims=claims,
        )
        _record_world_constitution_mutation(
            db=db,
            locked_session=locked_session,
            session_id=session_id,
            mutation=mutation,
            uploaded_lore_hash=uploaded_lore_hash,
            language=language,
        )
        if not open_gap_ids:
            final_state = _finalized_lore_state(
                uploaded_lore_hash=uploaded_lore_hash,
                language=language,
            )
            _persist_lore_state(locked_session, final_state)
            return final_state
        _persist_lore_state(locked_session, normalized_state)
        return normalized_state

    _persist_lore_state(locked_session, normalized_state)
    return normalized_state


def _restore_stable_lore_state_if_current(
    db: Session,
    *,
    session_id: uuid.UUID,
    request_id: str,
) -> bool:
    had_active_tx = db.in_transaction()
    _reset_implicit_tx_if_needed(db, had_active_tx=had_active_tx)
    with db.begin():
        locked_session = _lock_lore_session_for_mutation(db, session_id)
        current_state = _extract_lore_state(locked_session)
        if not _is_current_processing_job(current_state, request_id=request_id):
            return False
        _persist_lore_state(locked_session, _extract_stable_lore_state(current_state))
    return True


def _enqueue_lore_processing_event_if_current(
    db: Session,
    *,
    session_id: uuid.UUID,
    request_id: str,
    uploaded_lore_hash: str,
) -> dict[str, Any]:
    had_active_tx = db.in_transaction()
    _reset_implicit_tx_if_needed(db, had_active_tx=had_active_tx)
    with db.begin():
        locked_session = _lock_lore_session_for_mutation(db, session_id)
        current_state = _extract_lore_state(locked_session)
        current_world_constitution = _get_world_constitution(db, session_id)
        if not _is_current_processing_job(current_state, request_id=request_id, uploaded_lore_hash=uploaded_lore_hash):
            current_hash = _normalize_text(current_state.get("uploaded_lore_hash"), max_chars=128)
            if current_hash and current_hash != uploaded_lore_hash:
                raise _stale_lore_conflict(detail="Lore upload was superseded by a newer upload.")
            return _build_session_lore_payload(
                session_id=session_id,
                lore_state=current_state,
                world_constitution_row=current_world_constitution,
            )

        outbox_runtime.enqueue_outbox_event(
            db,
            event_type=outbox_runtime.EVENT_LORE_ADAPTATION_UPLOAD,
            payload={"request_id": request_id},
            session_id=session_id,
            turn_index=None,
            trace_id=get_trace_id(),
            dedupe_key=f"lore_adaptation_upload:{session_id}:{request_id}",
            max_attempts=_LORE_MAX_OUTBOX_ATTEMPTS,
        )
        return _build_session_lore_payload(
            session_id=session_id,
            lore_state=current_state,
            world_constitution_row=current_world_constitution,
        )


def _apply_processed_lore_state_if_current(
    db: Session,
    *,
    session_id: uuid.UUID,
    request_id: str,
    uploaded_lore_hash: str,
    working_state: dict[str, Any],
    raise_on_superseded: bool,
) -> dict[str, Any]:
    had_active_tx = db.in_transaction()
    _reset_implicit_tx_if_needed(db, had_active_tx=had_active_tx)
    with db.begin():
        locked_session = _lock_lore_session_for_mutation(db, session_id)
        current_state = _extract_lore_state(locked_session)
        current_world_constitution = _get_world_constitution(db, session_id)
        if not _is_current_processing_job(current_state, request_id=request_id, uploaded_lore_hash=uploaded_lore_hash):
            current_hash = _normalize_text(current_state.get("uploaded_lore_hash"), max_chars=128)
            if raise_on_superseded and current_hash and current_hash != uploaded_lore_hash:
                raise _stale_lore_conflict(detail="Lore upload was superseded by a newer upload.")
            return _build_session_lore_payload(
                session_id=session_id,
                lore_state=current_state,
                world_constitution_row=current_world_constitution,
            )

        persisted_state = _persist_completed_lore_state(
            db=db,
            locked_session=locked_session,
            session_id=session_id,
            lore_state=working_state,
        )
        current_world_constitution = _get_world_constitution(db, session_id)
        return _build_session_lore_payload(
            session_id=session_id,
            lore_state=persisted_state,
            world_constitution_row=current_world_constitution,
        )


def _run_lore_processing_outbox_event(
    *,
    session_id: uuid.UUID,
    request_id: str,
    attempt_number: int,
    max_attempts: int,
) -> None:
    read_db = SessionLocal()
    try:
        snapshot = _read_processing_job_snapshot(
            read_db,
            session_id=session_id,
            request_id=request_id,
        )
    finally:
        if read_db.in_transaction():
            read_db.rollback()
        read_db.close()

    if snapshot is None:
        return

    try:
        working_state = _build_working_lore_state(
            session_id=session_id,
            uploaded_lore_hash=snapshot["uploaded_lore_hash"],
            lore_text=snapshot["lore_text"],
            mode=snapshot["mode"],
        )

        write_db = SessionLocal()
        try:
            _apply_processed_lore_state_if_current(
                write_db,
                session_id=session_id,
                request_id=request_id,
                uploaded_lore_hash=snapshot["uploaded_lore_hash"],
                working_state=working_state,
                raise_on_superseded=False,
            )
        finally:
            write_db.close()
    except Exception:
        if max(int(max_attempts), 1) <= max(int(attempt_number), 1):
            restore_db = SessionLocal()
            try:
                _restore_stable_lore_state_if_current(
                    restore_db,
                    session_id=session_id,
                    request_id=request_id,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to restore lore state after final async processing error for session_id=%s",
                    session_id,
                )
            finally:
                restore_db.close()
        raise


def upload_lore(
    db: Session,
    session_id: uuid.UUID,
    payload: schemas.LoreUploadIn,
) -> dict[str, Any]:
    _ensure_lore_adaptation_enabled()
    had_active_tx = db.in_transaction()
    lore_text = _normalize_text(payload.lore_text, max_chars=LORE_ADAPTATION_MAX_CHARS)
    if not lore_text:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="lore_text is required")
    if len(payload.lore_text) > LORE_ADAPTATION_MAX_CHARS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"lore_text exceeds {LORE_ADAPTATION_MAX_CHARS} chars",
        )

    lore_hash = _hash_lore_text(lore_text)
    mode = payload.mode if payload.mode in {"interactive", "auto_fill"} else "interactive"
    _reset_implicit_tx_if_needed(db, had_active_tx=had_active_tx)
    request_id = uuid.uuid4().hex
    with db.begin():
        locked_session = _lock_lore_session_for_mutation(db, session_id)
        current_lore_state = _extract_lore_state(locked_session)
        world_constitution_row = _get_world_constitution(db, session_id)
        finalized_meta = _extract_finalized_lore_meta(world_constitution_row)
        finalized_hash = _normalize_text(finalized_meta.get("uploaded_lore_hash"), max_chars=128)
        finalized_language = _normalize_language(finalized_meta.get("language"), fallback="unknown")

        current_hash = _normalize_text(current_lore_state.get("uploaded_lore_hash"), max_chars=128)
        current_status = _lore_status(current_lore_state)
        if current_hash and current_hash == lore_hash:
            payload_data = _build_session_lore_payload(
                session_id=session_id,
                lore_state=current_lore_state,
                world_constitution_row=world_constitution_row,
            )
            if current_status == "processing":
                raise HTTPException(
                    status_code=status.HTTP_202_ACCEPTED,
                    detail=payload_data,
                    headers={"Retry-After": str(max(LORE_ADAPTATION_RETRY_AFTER_SECONDS, 1))},
                )
            if current_status in {"awaiting_answers", "finalized"}:
                return payload_data

        if finalized_hash and finalized_hash == lore_hash:
            next_state = _finalized_lore_state(
                uploaded_lore_hash=lore_hash,
                language=finalized_language,
            )
            _persist_lore_state(locked_session, next_state)
            return _build_session_lore_payload(
                session_id=session_id,
                lore_state=next_state,
                world_constitution_row=world_constitution_row,
            )

        processing_state = _build_processing_lore_state(
            uploaded_lore_hash=lore_hash,
            lore_text=lore_text,
            mode=mode,
            stable_state=current_lore_state,
            request_id=request_id,
        )
        _persist_lore_state(locked_session, processing_state)
        processing_payload = _build_session_lore_payload(
            session_id=session_id,
            lore_state=processing_state,
            world_constitution_row=world_constitution_row,
        )

    try:
        working_state = _build_working_lore_state(
            session_id=session_id,
            uploaded_lore_hash=lore_hash,
            lore_text=lore_text,
            mode=mode,
        )
    except Exception as exc:  # noqa: BLE001
        if _is_timeout_like_error(exc):
            queued_payload = _enqueue_lore_processing_event_if_current(
                db,
                session_id=session_id,
                request_id=request_id,
                uploaded_lore_hash=lore_hash,
            )
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail=queued_payload,
                headers={"Retry-After": str(max(LORE_ADAPTATION_RETRY_AFTER_SECONDS, 1))},
            ) from exc
        _restore_stable_lore_state_if_current(
            db,
            session_id=session_id,
            request_id=request_id,
        )
        logger.exception("Lore analysis failed for session_id=%s", session_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="lore analysis provider unavailable",
        ) from exc

    result_payload = _apply_processed_lore_state_if_current(
        db,
        session_id=session_id,
        request_id=request_id,
        uploaded_lore_hash=lore_hash,
        working_state=working_state,
        raise_on_superseded=True,
    )
    if result_payload["status"] == "processing":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=processing_payload,
            headers={"Retry-After": str(max(LORE_ADAPTATION_RETRY_AFTER_SECONDS, 1))},
        )
    return result_payload


def get_lore_adaptation(db: Session, session_id: uuid.UUID) -> dict[str, Any]:
    _ensure_lore_adaptation_enabled()
    session_row = _require_session(db, session_id)
    lore_state = _extract_lore_state(session_row)
    world_constitution_row = _get_world_constitution(db, session_id)
    return _build_session_lore_payload(
        session_id=session_id,
        lore_state=lore_state,
        world_constitution_row=world_constitution_row,
    )


def answer_lore_gap(
    db: Session,
    session_id: uuid.UUID,
    gap_id: str,
    payload: schemas.LoreGapAnswerIn,
) -> dict[str, Any]:
    _ensure_lore_adaptation_enabled()
    had_active_tx = db.in_transaction()
    normalized_gap_id = _normalize_text(gap_id, max_chars=200)
    if not normalized_gap_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="gap_id is required")

    _reset_implicit_tx_if_needed(db, had_active_tx=had_active_tx)
    if not payload.let_ai_decide:
        answer_text = _normalize_text(payload.answer, max_chars=2_000)
        if not answer_text:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="answer is required")
        with db.begin():
            locked_session = _lock_lore_session_for_mutation(db, session_id)
            latest_state = _extract_lore_state(locked_session)
            latest_gaps = _coerce_gap_list(latest_state.get("gaps"), session_id=session_id)
            if not latest_gaps:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active lore adaptation gaps")
            target_gap = next((gap for gap in latest_gaps if gap.get("gap_id") == normalized_gap_id), None)
            if target_gap is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Gap not found")
            key = _normalize_text(target_gap.get("key"), max_chars=120)
            if not key:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Gap key is invalid")

            persisted_state = _persist_completed_lore_state(
                db=db,
                locked_session=locked_session,
                session_id=session_id,
                lore_state=_apply_answers_to_state(
                    state=latest_state,
                    answers={key: (answer_text, "player")},
                ),
            )
            world_constitution_row = _get_world_constitution(db, session_id)
            return _build_session_lore_payload(
                session_id=session_id,
                lore_state=persisted_state,
                world_constitution_row=world_constitution_row,
            )

    with db.begin():
        locked_session = _lock_lore_session_for_mutation(db, session_id)
        lore_state = _extract_lore_state(locked_session)
        gaps = _coerce_gap_list(lore_state.get("gaps"), session_id=session_id)
        if not gaps:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active lore adaptation gaps")
        target_gap = next((gap for gap in gaps if gap.get("gap_id") == normalized_gap_id), None)
        if target_gap is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Gap not found")
        key = _normalize_text(target_gap.get("key"), max_chars=120)
        if not key:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Gap key is invalid")
        base_hash = _normalize_text(lore_state.get("uploaded_lore_hash"), max_chars=128)
        source_text = _normalize_text(lore_state.get("source_text"), max_chars=LORE_ADAPTATION_MAX_CHARS)
        if not source_text:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No uploaded lore source available for AI auto-fill",
            )
        language = _normalize_language(lore_state.get("language"), fallback="unknown")
        draft_profile = {
            str(item_key): _normalize_text(item_value, max_chars=2_000)
            for item_key, item_value in dict(lore_state.get("draft_profile") or {}).items()
            if _normalize_text(item_value, max_chars=2_000)
        }

    try:
        fills = _call_lore_gap_fill(
            lore_text=source_text,
            language=language,
            draft_profile=draft_profile,
            keys=[key],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lore single-gap auto-fill failed for session_id=%s gap=%s", session_id, key)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="lore auto-fill provider unavailable",
        ) from exc

    answer_text = _normalize_text(fills.get(key), max_chars=2_000)
    if not answer_text:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI failed to resolve the requested gap",
        )

    with db.begin():
        locked_session = _lock_lore_session_for_mutation(db, session_id)
        latest_state = _extract_lore_state(locked_session)
        latest_hash = _normalize_text(latest_state.get("uploaded_lore_hash"), max_chars=128)
        if latest_hash and latest_hash != base_hash:
            raise _stale_lore_conflict()
        latest_gaps = _coerce_gap_list(latest_state.get("gaps"), session_id=session_id)
        target_gap = next((gap for gap in latest_gaps if gap.get("gap_id") == normalized_gap_id), None)
        if target_gap is None or str(target_gap.get("status") or "open").strip().lower() == "resolved":
            world_constitution_row = _get_world_constitution(db, session_id)
            return _build_session_lore_payload(
                session_id=session_id,
                lore_state=latest_state,
                world_constitution_row=world_constitution_row,
            )

        persisted_state = _persist_completed_lore_state(
            db=db,
            locked_session=locked_session,
            session_id=session_id,
            lore_state=_apply_answers_to_state(
                state=latest_state,
                answers={key: (answer_text, "ai")},
            ),
        )
        world_constitution_row = _get_world_constitution(db, session_id)
        return _build_session_lore_payload(
            session_id=session_id,
            lore_state=persisted_state,
            world_constitution_row=world_constitution_row,
        )


def auto_resolve_lore_gaps(
    db: Session,
    session_id: uuid.UUID,
) -> dict[str, Any]:
    _ensure_lore_adaptation_enabled()
    had_active_tx = db.in_transaction()

    _reset_implicit_tx_if_needed(db, had_active_tx=had_active_tx)
    with db.begin():
        locked_session = _lock_lore_session_for_mutation(db, session_id)
        lore_state = _extract_lore_state(locked_session)
        gaps = _coerce_gap_list(lore_state.get("gaps"), session_id=session_id)
        if not gaps:
            world_constitution_row = _get_world_constitution(db, session_id)
            return _build_session_lore_payload(
                session_id=session_id,
                lore_state=lore_state,
                world_constitution_row=world_constitution_row,
            )

        open_keys = [
            _normalize_text(gap.get("key"), max_chars=120)
            for gap in gaps
            if str(gap.get("status") or "open").strip().lower() != "resolved"
        ]
        open_keys = [key for key in open_keys if key and _VALID_KEY_RE.match(key)]
        if not open_keys:
            world_constitution_row = _get_world_constitution(db, session_id)
            return _build_session_lore_payload(
                session_id=session_id,
                lore_state=lore_state,
                world_constitution_row=world_constitution_row,
            )

        source_text = _normalize_text(lore_state.get("source_text"), max_chars=LORE_ADAPTATION_MAX_CHARS)
        if not source_text:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No uploaded lore source available for AI auto-fill",
            )

        base_hash = _normalize_text(lore_state.get("uploaded_lore_hash"), max_chars=128)
        language = _normalize_language(lore_state.get("language"), fallback="unknown")
        draft_profile = {
            str(key): _normalize_text(value, max_chars=2_000)
            for key, value in dict(lore_state.get("draft_profile") or {}).items()
            if _normalize_text(value, max_chars=2_000)
        }

    try:
        fills = _call_lore_gap_fill(
            lore_text=source_text,
            language=language,
            draft_profile=draft_profile,
            keys=open_keys,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lore bulk auto-fill failed for session_id=%s", session_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="lore auto-fill provider unavailable",
        ) from exc
    answers = {key: (value, "ai") for key, value in fills.items() if _normalize_text(value, max_chars=2_000)}
    if not answers:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI failed to auto-resolve lore gaps",
        )

    with db.begin():
        locked_session = _lock_lore_session_for_mutation(db, session_id)
        latest_state = _extract_lore_state(locked_session)
        latest_hash = _normalize_text(latest_state.get("uploaded_lore_hash"), max_chars=128)
        if latest_hash and latest_hash != base_hash:
            raise _stale_lore_conflict()

        latest_gaps = _coerce_gap_list(latest_state.get("gaps"), session_id=session_id)
        if not latest_gaps:
            world_constitution_row = _get_world_constitution(db, session_id)
            return _build_session_lore_payload(
                session_id=session_id,
                lore_state=latest_state,
                world_constitution_row=world_constitution_row,
            )

        current_open_keys = {
            _normalize_text(gap.get("key"), max_chars=120)
            for gap in latest_gaps
            if str(gap.get("status") or "open").strip().lower() != "resolved"
        }
        merged_answers = {
            key: value
            for key, value in answers.items()
            if key in current_open_keys
        }
        if not merged_answers:
            world_constitution_row = _get_world_constitution(db, session_id)
            return _build_session_lore_payload(
                session_id=session_id,
                lore_state=latest_state,
                world_constitution_row=world_constitution_row,
            )

        persisted_state = _persist_completed_lore_state(
            db=db,
            locked_session=locked_session,
            session_id=session_id,
            lore_state=_apply_answers_to_state(state=latest_state, answers=merged_answers),
        )
        world_constitution_row = _get_world_constitution(db, session_id)
        return _build_session_lore_payload(
            session_id=session_id,
            lore_state=persisted_state,
            world_constitution_row=world_constitution_row,
        )


def get_lore_turn_blocker(db: Session, session_id: uuid.UUID) -> dict[str, Any] | None:
    if not USE_LORE_ADAPTATION:
        return None
    session_row = _require_session(db, session_id)
    lore_state = _extract_lore_state(session_row)
    if _lore_status(lore_state) == "processing":
        return {
            "code": "lore_adaptation_processing",
            "message": "Lore adaptation is still processing.",
            "required_gap_ids": [],
            "blocking_labels": [],
            "blocking_domains": [],
            "why_this_blocks_turns": "Lore adaptation is still processing.",
        }
    gaps = _annotate_gaps_for_output(_coerce_gap_list(lore_state.get("gaps"), session_id=session_id))
    blocking_summary = build_lore_blocking_summary(_lore_status(lore_state), gaps, policy=LORE_UX_POLICY)
    required_gap_ids = [
        _normalize_text(gap_id, max_chars=200)
        for gap_id in list(blocking_summary.get("blocking_gap_ids") or [])
        if _normalize_text(gap_id, max_chars=200)
    ]
    if not required_gap_ids:
        return None

    return {
        "code": "lore_adaptation_required",
        "message": "Resolve gameplay-critical lore questions before running turns.",
        "required_gap_ids": required_gap_ids,
        "blocking_labels": list(blocking_summary.get("blocking_labels") or []),
        "blocking_domains": list(blocking_summary.get("blocking_domains") or []),
        "why_this_blocks_turns": str(blocking_summary.get("why_this_blocks_turns") or ""),
    }


__all__ = [
    "upload_lore",
    "get_lore_adaptation",
    "answer_lore_gap",
    "auto_resolve_lore_gaps",
    "get_lore_turn_blocker",
]
