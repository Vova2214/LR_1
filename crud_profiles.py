from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from .cache_backend import TwoTierCache
from .crud_shared import (
    TurnApplyExternalPreparationRequired,
    TurnApplyExternalRequest,
    _normalize_json_preview,
    _truncate_text,
    current_turn_apply_external_artifacts,
)
from .db import OPENROUTER_CHAT_MODEL, USE_PROFILE_SYNTHESIZER
from .llm import openrouter_chat
from .llm_telemetry import telemetry_context
from .strings import (
    DATA_KEY_NPC_KNOWLEDGE_PREFIX_EN,
    DATA_KEY_NPC_KNOWLEDGE_PREFIX_RU,
    LABEL_RELATION_TO_PLAYER,
)

PLAYER_PROFILE_TEXT_KEYS = (
    "short_desc",
    "description",
    "desc",
    "backstory",
    "bio",
    "history",
    "goal",
    "class",
    "role",
    "skills",
    "traits",
)
PLAYER_PROFILE_TECHNICAL_KEYS = {
    "status",
    "spawn",
    "ephemeral",
    "pinned",
    "despawn_turn",
    "despawned_turn",
    "despawn_reason",
}
NPC_PROFILE_SPECIAL_KEYS = {
    "short_desc",
    "отношение",
    "attitude",
    "relation_to_player",
}
NPC_PROFILE_TECHNICAL_KEYS = {
    "ephemeral",
    "pinned",
    "despawn_turn",
    "spawn",
    "status",
    "despawned_turn",
    "despawn_reason",
}
ZONE_PROFILE_TEXT_KEYS = ("short_desc", "description", "desc", "описание")
ZONE_PROFILE_TECHNICAL_KEYS = {
    "status",
    "spawn",
    "ephemeral",
    "pinned",
    "despawn_turn",
    "despawned_turn",
    "despawn_reason",
}
ITEM_FACTION_PROFILE_TEXT_KEYS = ("short_desc", "description", "desc", "summary", "lore", "описание")
ITEM_FACTION_PROFILE_TECHNICAL_KEYS = {
    "status",
    "spawn",
    "ephemeral",
    "pinned",
    "despawn_turn",
    "despawned_turn",
    "despawn_reason",
}
QUEST_PROFILE_TEXT_KEYS = (
    "short_desc",
    "description",
    "desc",
    "summary",
    "objective",
    "goal",
    "current_step",
    "stage",
    "lore",
    "описание",
)
QUEST_PROFILE_TECHNICAL_KEYS = {
    "status",
    "spawn",
    "ephemeral",
    "pinned",
    "despawn_turn",
    "despawned_turn",
    "despawn_reason",
}
_PROFILE_SYNTHESIZER_SYSTEM = (
    "You convert structured RPG entity data into one coherent embedding-friendly profile sentence. "
    "Preserve explicit facts and constraints from input. No inventions. "
    "Avoid key:value lists and avoid markdown. Max 65 words."
)
_PROFILE_SYNTH_CACHE_MAX_ENTRIES = 2048
_profile_synth_cache = TwoTierCache(
    name="profile_synth",
    max_entries=_PROFILE_SYNTH_CACHE_MAX_ENTRIES,
)
_PROFILE_SYNTH_MAX_CHARS = 800

logger = logging.getLogger(__name__)


def _profile_synth_cache_key(
    *,
    object_type: str,
    name: str,
    data: dict[str, Any],
) -> str:
    payload = {
        "object_type": str(object_type or "").strip().lower(),
        "name": str(name or "").strip(),
        "data": data,
    }
    try:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:  # noqa: BLE001
        serialized = _normalize_json_preview(payload, 7000)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _get_cached_profile_text(cache_key: str) -> str | None:
    cached = _profile_synth_cache.get(cache_key)
    if cached is None:
        return None
    return str(cached)


def _set_cached_profile_text(cache_key: str, text: str) -> None:
    _profile_synth_cache.set(cache_key, text)


def _clear_profile_synth_cache() -> None:
    _profile_synth_cache.clear()


def _maybe_synthesize_profile_text(
    *,
    object_type: str,
    name: str,
    data: dict[str, Any],
    fallback_text: str,
) -> str:
    if not USE_PROFILE_SYNTHESIZER:
        return fallback_text

    cache_key = _profile_synth_cache_key(object_type=object_type, name=name, data=data)
    cached = _get_cached_profile_text(cache_key)
    if cached is not None:
        return cached

    artifacts = current_turn_apply_external_artifacts()
    if artifacts is not None:
        prepared_text = artifacts.profile_texts.get(cache_key)
        if prepared_text is None:
            raise TurnApplyExternalPreparationRequired(
                TurnApplyExternalRequest(
                    kind="profile_text",
                    profile_cache_key=cache_key,
                    profile_object_type=str(object_type or "").strip(),
                    profile_name=str(name or "").strip(),
                    profile_data=dict(data or {}),
                    profile_fallback_text=fallback_text,
                )
            )
        return str(prepared_text)

    payload = {
        "object_type": object_type,
        "name": _truncate_text(str(name or "").strip(), 120),
        "data": data,
        "fallback_profile": _truncate_text(fallback_text, _PROFILE_SYNTH_MAX_CHARS),
    }
    try:
        with telemetry_context(request_type="profile_synthesizer"):
            synthesized = openrouter_chat.generate_text(
                model=OPENROUTER_CHAT_MODEL,
                system_prompt=_PROFILE_SYNTHESIZER_SYSTEM,
                user_prompt=_normalize_json_preview(payload, 7000),
                max_tokens=180,
            )
    except Exception:  # noqa: BLE001
        logger.warning("Profile synthesizer failed, fallback to deterministic profile text", exc_info=True)
        return fallback_text

    text = _truncate_text(" ".join(str(synthesized or "").split()).strip(), _PROFILE_SYNTH_MAX_CHARS)
    if not text:
        logger.warning(
            "Profile synthesizer returned empty text, fallback to deterministic profile text for type=%s name=%s",
            object_type,
            name,
        )
        return fallback_text
    _set_cached_profile_text(cache_key, text)
    return text


def _build_player_profile_text(name: str, data: dict[str, Any]) -> str:
    parts: list[str] = []
    base_name = str(name or "").strip() or "Player"
    parts.append(base_name)

    description: str | None = None
    for key in PLAYER_PROFILE_TEXT_KEYS:
        raw_value = data.get(key)
        if raw_value is None:
            continue
        raw_text = str(raw_value).strip()
        if raw_text:
            description = raw_text
            break
    if description:
        parts.append(description)

    scalar_facts: list[str] = []
    skip_keys = set(PLAYER_PROFILE_TEXT_KEYS) | PLAYER_PROFILE_TECHNICAL_KEYS
    for key, value in data.items():
        if key in skip_keys:
            continue
        if isinstance(value, (str, int, float, bool)):
            scalar_facts.append(f"{key}: {value}")
    if scalar_facts:
        parts.append("; ".join(scalar_facts[:8]))

    text = ". ".join(p.strip().rstrip(".") for p in parts if p and str(p).strip())
    base_text = _truncate_text(text.strip(), 800) or base_name
    return _maybe_synthesize_profile_text(
        object_type="player",
        name=base_name,
        data=data,
        fallback_text=base_text,
    )


def _should_refresh_player_profile_embedding(
    old_name: str,
    new_name: str,
    old_data: dict[str, Any],
    patch_data: dict[str, Any],
) -> bool:
    if new_name.strip() != old_name.strip():
        return True
    if not patch_data:
        return False

    for key, new_value in patch_data.items():
        if key in PLAYER_PROFILE_TECHNICAL_KEYS:
            continue
        if key in PLAYER_PROFILE_TEXT_KEYS:
            old_text = "" if old_data.get(key) is None else str(old_data.get(key)).strip()
            new_text = "" if new_value is None else str(new_value).strip()
            if old_text != new_text:
                return True
            continue

        old_value = old_data.get(key)
        if isinstance(new_value, (str, int, float, bool)) or isinstance(old_value, (str, int, float, bool)):
            return True
    return False


def _build_npc_profile_text(name: str, data: dict[str, Any]) -> str:
    parts: list[str] = []
    base_name = str(name or "").strip() or "NPC"
    parts.append(base_name)

    short_desc = str(data.get("short_desc", "")).strip()
    if short_desc:
        parts.append(short_desc)

    relation = (
        data.get("отношение")
        or data.get("attitude")
        or data.get("relation_to_player")
    )
    if relation is not None and str(relation).strip():
        parts.append(f"{LABEL_RELATION_TO_PLAYER}: {str(relation).strip()}.")

    scalar_facts: list[str] = []
    skip_keys = NPC_PROFILE_SPECIAL_KEYS | NPC_PROFILE_TECHNICAL_KEYS
    for key, value in data.items():
        if key in skip_keys:
            continue
        if isinstance(value, (str, int, float, bool)):
            scalar_facts.append(f"{key}: {value}")
    if scalar_facts:
        parts.append("; ".join(scalar_facts[:8]))

    text = ". ".join(p.strip().rstrip(".") for p in parts if p and str(p).strip())
    base_text = _truncate_text(text.strip(), 800) or base_name
    return _maybe_synthesize_profile_text(
        object_type="npc",
        name=base_name,
        data=data,
        fallback_text=base_text,
    )


def _should_refresh_npc_profile_embedding(
    old_name: str,
    new_name: str,
    old_data: dict[str, Any],
    patch_data: dict[str, Any],
) -> bool:
    if new_name.strip() != old_name.strip():
        return True
    if not patch_data:
        return False

    for key, new_value in patch_data.items():
        if key in NPC_PROFILE_SPECIAL_KEYS:
            return True
        if key.startswith(DATA_KEY_NPC_KNOWLEDGE_PREFIX_RU) or key.startswith(
            DATA_KEY_NPC_KNOWLEDGE_PREFIX_EN
        ):
            return True
        if key in NPC_PROFILE_TECHNICAL_KEYS:
            continue

        old_value = old_data.get(key)
        if isinstance(new_value, (str, int, float, bool)) or isinstance(old_value, (str, int, float, bool)):
            return True
    return False


def _build_zone_profile_text(name: str, data: dict[str, Any]) -> str:
    parts: list[str] = []
    base_name = str(name or "").strip() or "Zone"
    parts.append(base_name)

    description: str | None = None
    for key in ZONE_PROFILE_TEXT_KEYS:
        raw_value = data.get(key)
        if raw_value is None:
            continue
        raw_text = str(raw_value).strip()
        if raw_text:
            description = raw_text
            break
    if description:
        parts.append(description)

    scalar_facts: list[str] = []
    skip_keys = set(ZONE_PROFILE_TEXT_KEYS) | ZONE_PROFILE_TECHNICAL_KEYS
    for key, value in data.items():
        if key in skip_keys:
            continue
        if isinstance(value, (str, int, float, bool)):
            scalar_facts.append(f"{key}: {value}")
    if scalar_facts:
        parts.append("; ".join(scalar_facts[:8]))

    text = ". ".join(p.strip().rstrip(".") for p in parts if p and str(p).strip())
    base_text = _truncate_text(text.strip(), 800) or base_name
    return _maybe_synthesize_profile_text(
        object_type="zone",
        name=base_name,
        data=data,
        fallback_text=base_text,
    )


def _should_refresh_zone_profile_embedding(
    old_name: str,
    new_name: str,
    old_data: dict[str, Any],
    patch_data: dict[str, Any],
) -> bool:
    if new_name.strip() != old_name.strip():
        return True
    for key in ZONE_PROFILE_TEXT_KEYS:
        if key not in patch_data:
            continue
        old_value = "" if old_data.get(key) is None else str(old_data.get(key)).strip()
        new_value = "" if patch_data.get(key) is None else str(patch_data.get(key)).strip()
        if old_value != new_value:
            return True
    return False


def _build_item_profile_text(name: str, data: dict[str, Any]) -> str:
    parts: list[str] = []
    base_name = str(name or "").strip() or "Item"
    parts.append(base_name)

    description: str | None = None
    for key in ITEM_FACTION_PROFILE_TEXT_KEYS:
        raw_value = data.get(key)
        if raw_value is None:
            continue
        raw_text = str(raw_value).strip()
        if raw_text:
            description = raw_text
            break
    if description:
        parts.append(description)

    scalar_facts: list[str] = []
    skip_keys = set(ITEM_FACTION_PROFILE_TEXT_KEYS) | ITEM_FACTION_PROFILE_TECHNICAL_KEYS
    for key, value in data.items():
        if key in skip_keys:
            continue
        if isinstance(value, (str, int, float, bool)):
            scalar_facts.append(f"{key}: {value}")
    if scalar_facts:
        parts.append("; ".join(scalar_facts[:8]))

    text = ". ".join(p.strip().rstrip(".") for p in parts if p and str(p).strip())
    base_text = _truncate_text(text.strip(), 800) or base_name
    return _maybe_synthesize_profile_text(
        object_type="item",
        name=base_name,
        data=data,
        fallback_text=base_text,
    )


def _build_faction_profile_text(name: str, data: dict[str, Any]) -> str:
    parts: list[str] = []
    base_name = str(name or "").strip() or "Faction"
    parts.append(base_name)

    description: str | None = None
    for key in ITEM_FACTION_PROFILE_TEXT_KEYS:
        raw_value = data.get(key)
        if raw_value is None:
            continue
        raw_text = str(raw_value).strip()
        if raw_text:
            description = raw_text
            break
    if description:
        parts.append(description)

    scalar_facts: list[str] = []
    skip_keys = set(ITEM_FACTION_PROFILE_TEXT_KEYS) | ITEM_FACTION_PROFILE_TECHNICAL_KEYS
    for key, value in data.items():
        if key in skip_keys:
            continue
        if isinstance(value, (str, int, float, bool)):
            scalar_facts.append(f"{key}: {value}")
    if scalar_facts:
        parts.append("; ".join(scalar_facts[:8]))

    text = ". ".join(p.strip().rstrip(".") for p in parts if p and str(p).strip())
    base_text = _truncate_text(text.strip(), 800) or base_name
    return _maybe_synthesize_profile_text(
        object_type="faction",
        name=base_name,
        data=data,
        fallback_text=base_text,
    )


def _build_quest_profile_text(name: str, data: dict[str, Any]) -> str:
    parts: list[str] = []
    base_name = str(name or "").strip() or "Quest"
    parts.append(base_name)

    description: str | None = None
    for key in QUEST_PROFILE_TEXT_KEYS:
        raw_value = data.get(key)
        if raw_value is None:
            continue
        raw_text = str(raw_value).strip()
        if raw_text:
            description = raw_text
            break
    if description:
        parts.append(description)

    scalar_facts: list[str] = []
    skip_keys = set(QUEST_PROFILE_TEXT_KEYS) | QUEST_PROFILE_TECHNICAL_KEYS
    for key, value in data.items():
        if key in skip_keys:
            continue
        if isinstance(value, (str, int, float, bool)):
            scalar_facts.append(f"{key}: {value}")
    if scalar_facts:
        parts.append("; ".join(scalar_facts[:8]))

    text = ". ".join(p.strip().rstrip(".") for p in parts if p and str(p).strip())
    base_text = _truncate_text(text.strip(), 800) or base_name
    return _maybe_synthesize_profile_text(
        object_type="quest",
        name=base_name,
        data=data,
        fallback_text=base_text,
    )


def _should_refresh_item_or_faction_profile_embedding(
    old_name: str,
    new_name: str,
    old_data: dict[str, Any],
    patch_data: dict[str, Any],
) -> bool:
    if new_name.strip() != old_name.strip():
        return True
    for key in ITEM_FACTION_PROFILE_TEXT_KEYS:
        if key not in patch_data:
            continue
        old_value = "" if old_data.get(key) is None else str(old_data.get(key)).strip()
        new_value = "" if patch_data.get(key) is None else str(patch_data.get(key)).strip()
        if old_value != new_value:
            return True
    return False


def _should_refresh_quest_profile_embedding(
    old_name: str,
    new_name: str,
    old_data: dict[str, Any],
    patch_data: dict[str, Any],
) -> bool:
    if new_name.strip() != old_name.strip():
        return True
    for key in QUEST_PROFILE_TEXT_KEYS:
        if key not in patch_data:
            continue
        old_value = "" if old_data.get(key) is None else str(old_data.get(key)).strip()
        new_value = "" if patch_data.get(key) is None else str(patch_data.get(key)).strip()
        if old_value != new_value:
            return True
    return False


__all__ = [
    "_clear_profile_synth_cache",
    "_build_player_profile_text",
    "_should_refresh_player_profile_embedding",
    "_build_npc_profile_text",
    "_should_refresh_npc_profile_embedding",
    "_build_zone_profile_text",
    "_should_refresh_zone_profile_embedding",
    "_build_item_profile_text",
    "_build_faction_profile_text",
    "_build_quest_profile_text",
    "_should_refresh_item_or_faction_profile_embedding",
    "_should_refresh_quest_profile_embedding",
]
