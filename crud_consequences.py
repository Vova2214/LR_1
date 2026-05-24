from __future__ import annotations

import hashlib
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession, object_session

from . import models, schemas
from .constants import SESSION_PLAYER_REF

PRIORITY_WEIGHT = {"high": 1.0, "med": 0.6, "low": 0.3}
COOLDOWN_TURNS = 3
MAX_CONSEQUENCE_CHAIN_DEPTH = 2
MAX_CONSEQUENCE_WINDOW_SPAN = 24
CONSEQUENCE_WINDOW_OBJECT_TYPE = "__consequence_window"
CONSEQUENCE_WINDOW_NAME_PREFIX = "consequence_window"


def _stable_jitter(seed_id: str, turn_index: int) -> float:
    digest = hashlib.sha256(f"{seed_id}:{turn_index}".encode("utf-8")).digest()
    random_part = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return float(random_part) * 0.05


def _iter_pending_seeds(state_json: dict[str, Any]) -> list[schemas.ConsequenceSeed]:
    raw_pending = state_json.get("pending_consequences")
    if not isinstance(raw_pending, list):
        return []

    parsed: list[schemas.ConsequenceSeed] = []
    seen_ids: set[str] = set()
    for item in raw_pending:
        try:
            seed = schemas.ConsequenceSeed.model_validate(item)
        except Exception:  # noqa: BLE001
            continue
        if seed.id in seen_ids:
            continue
        seen_ids.add(seed.id)
        parsed.append(seed)
    return parsed


def _dump_pending_seeds(state_json: dict[str, Any], seeds: list[schemas.ConsequenceSeed]) -> None:
    state_json["pending_consequences"] = [seed.model_dump(mode="json") for seed in seeds]


def _legacy_window_key(session_id: uuid.UUID, seed_id: str) -> str:
    digest = hashlib.sha256(f"{session_id}:{seed_id}".encode("utf-8")).hexdigest()
    return f"legacy:{digest}"


def _coerce_window_payload(raw_payload: Any) -> dict[str, Any] | None:
    if not isinstance(raw_payload, dict):
        return None

    window_key = str(raw_payload.get("window_key") or "").strip()
    if not window_key:
        return None

    status = str(raw_payload.get("status") or "open").strip().lower()
    if status not in {"open", "shown", "resolved", "expired", "suppressed"}:
        status = "open"

    priority = str(raw_payload.get("priority") or "med").strip().lower()
    if priority not in {"low", "med", "high"}:
        priority = "med"

    earliest_turn = raw_payload.get("earliest_turn")
    latest_turn = raw_payload.get("latest_turn")
    if not isinstance(earliest_turn, int) or not isinstance(latest_turn, int):
        return None
    if latest_turn < earliest_turn:
        return None

    shows = raw_payload.get("shows", 0)
    if isinstance(shows, bool) or not isinstance(shows, int):
        shows = 0
    max_shows = raw_payload.get("max_shows", 2)
    if isinstance(max_shows, bool) or not isinstance(max_shows, int):
        max_shows = 2
    depth = raw_payload.get("depth", 0)
    if isinstance(depth, bool) or not isinstance(depth, int):
        depth = 0
    source_turn = raw_payload.get("source_turn")
    if isinstance(source_turn, bool) or (source_turn is not None and not isinstance(source_turn, int)):
        source_turn = None

    payload: dict[str, Any] = {
        "window_key": window_key,
        "status": status,
        "priority": priority,
        "earliest_turn": earliest_turn,
        "latest_turn": latest_turn,
        "shows": max(shows, 0),
        "max_shows": max(max_shows, 0),
        "depth": max(depth, 0),
        "target_object_ids": [
            str(item).strip()
            for item in list(raw_payload.get("target_object_ids") or [])
            if str(item).strip()
        ],
        "anchor_object_ids": [
            str(item).strip()
            for item in list(raw_payload.get("anchor_object_ids") or [])
            if str(item).strip()
        ],
        "source_turn": source_turn,
        "source_event_ids": [
            str(item).strip()
            for item in list(raw_payload.get("source_event_ids") or [])
            if str(item).strip()
        ],
        "source_fact_keys": [
            str(item).strip()
            for item in list(raw_payload.get("source_fact_keys") or [])
            if str(item).strip()
        ],
        "opened_obligation_keys": [
            str(item).strip()
            for item in list(raw_payload.get("opened_obligation_keys") or [])
            if str(item).strip()
        ],
        "created_by_turn_kind": str(raw_payload.get("created_by_turn_kind") or "player").strip() or "player",
        "provenance_status": str(raw_payload.get("provenance_status") or "exact").strip() or "exact",
        "reason_kind": str(raw_payload.get("reason_kind") or "followup").strip() or "followup",
        "text": str(raw_payload.get("text") or "").strip(),
    }
    suppressed_until_turn = raw_payload.get("suppressed_until_turn")
    if isinstance(suppressed_until_turn, int) and not isinstance(suppressed_until_turn, bool):
        payload["suppressed_until_turn"] = suppressed_until_turn
    return payload


def _window_payload_to_seed(payload: dict[str, Any]) -> schemas.ConsequenceSeed | None:
    coerced = _coerce_window_payload(payload)
    if coerced is None:
        return None
    text_value = str(coerced.get("text") or coerced.get("reason_kind") or coerced.get("window_key") or "").strip()
    if not text_value:
        text_value = str(coerced.get("window_key") or "").strip()
    try:
        return schemas.ConsequenceSeed(
            id=str(coerced.get("window_key") or "").strip(),
            text=text_value,
            priority=str(coerced.get("priority") or "med"),
            earliest_turn=int(coerced.get("earliest_turn") or 0),
            latest_turn=int(coerced.get("latest_turn") or 0),
            max_shows=int(coerced.get("max_shows") or 0),
            shows=int(coerced.get("shows") or 0),
            last_shown_turn=None,
            anchor_object_ids=list(coerced.get("anchor_object_ids") or []),
            created_turn=int(coerced.get("source_turn") or 0),
            depth=int(coerced.get("depth") or 0),
        )
    except Exception:  # noqa: BLE001
        return None


def list_consequence_window_payloads(
    db: OrmSession,
    session_id: uuid.UUID,
) -> list[dict[str, Any]]:
    rows = db.execute(
        select(models.ObjectModel).where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == CONSEQUENCE_WINDOW_OBJECT_TYPE,
        )
    ).scalars().all()
    payloads: list[dict[str, Any]] = []
    for row in rows:
        payload = _coerce_window_payload(dict(row.data or {}))
        if payload is None:
            continue
        payloads.append(payload)
    return payloads


def ensure_persisted_consequence_windows(
    db: OrmSession,
    session_id: uuid.UUID,
    *,
    state_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """Best-effort legacy backfill helper.

    This exists for migrations, repair jobs, and explicit backfill flows.
    Runtime consequence selection should read persisted windows directly and
    must not depend on session.state_json["pending_consequences"].
    """
    existing_payloads = list_consequence_window_payloads(db, session_id)
    existing_keys = {str(item.get("window_key") or "").strip() for item in existing_payloads}
    created_payloads: list[dict[str, Any]] = []
    for seed in _iter_pending_seeds(state_json):
        window_key = _legacy_window_key(session_id, seed.id)
        if window_key in existing_keys:
            continue
        payload = {
            "window_key": window_key,
            "status": "open",
            "priority": seed.priority,
            "earliest_turn": seed.earliest_turn,
            "latest_turn": seed.latest_turn,
            "shows": seed.shows,
            "max_shows": seed.max_shows,
            "depth": seed.depth,
            "target_object_ids": [],
            "anchor_object_ids": list(seed.anchor_object_ids or []),
            "source_turn": None,
            "source_event_ids": [],
            "source_fact_keys": [],
            "opened_obligation_keys": [],
            "created_by_turn_kind": "legacy_backfill",
            "provenance_status": "legacy_backfill",
            "reason_kind": "legacy_backfill",
            "text": seed.text,
        }
        db.add(
            models.ObjectModel(
                session_id=session_id,
                type=CONSEQUENCE_WINDOW_OBJECT_TYPE,
                name=f"{CONSEQUENCE_WINDOW_NAME_PREFIX}:{window_key}",
                data=payload,
            )
        )
        created_payloads.append(payload)
        existing_keys.add(window_key)
    if created_payloads:
        db.flush()
    return created_payloads


def _normalize_ref_to_object_id(
    raw_ref: Any,
    *,
    ref_map: dict[str, str],
    player_object_id: uuid.UUID | None,
) -> str | None:
    ref_text = str(raw_ref or "").strip()
    if not ref_text:
        return None
    if ref_text == SESSION_PLAYER_REF:
        return str(player_object_id) if player_object_id is not None else None
    mapped_value = str(ref_map.get(ref_text) or ref_text).strip()
    try:
        return str(uuid.UUID(mapped_value))
    except (TypeError, ValueError, AttributeError):
        return None


def materialize_consequence_windows(
    db: OrmSession,
    session_id: uuid.UUID,
    *,
    turn_index: int,
    intents: list[schemas.ConsequenceIntent],
    ref_map: dict[str, str],
    player_object_id: uuid.UUID | None,
    source_event_ids: list[str],
    source_fact_keys: list[str],
    opened_obligation_keys: list[str],
    created_by_turn_kind: str,
    depth: int,
    legacy_seed_lookup: dict[str, schemas.ConsequenceSeed] | None = None,
) -> list[dict[str, Any]]:
    if not intents:
        return []

    existing_rows = db.execute(
        select(models.ObjectModel).where(
            models.ObjectModel.session_id == session_id,
            models.ObjectModel.type == CONSEQUENCE_WINDOW_OBJECT_TYPE,
        )
    ).scalars().all()
    existing_by_key = {
        str(dict(row.data or {}).get("window_key") or "").strip(): row
        for row in existing_rows
        if str(dict(row.data or {}).get("window_key") or "").strip()
    }
    created_payloads: list[dict[str, Any]] = []
    legacy_seed_lookup = legacy_seed_lookup or {}

    for intent in intents:
        window_key = str(intent.intent_key or "").strip()
        if not window_key:
            continue
        legacy_seed = legacy_seed_lookup.get(window_key)
        target_object_ids = [
            target_id
            for target_id in (
                _normalize_ref_to_object_id(item, ref_map=ref_map, player_object_id=player_object_id)
                for item in list(intent.target_refs or [])
            )
            if target_id is not None
        ]
        anchor_object_ids = [
            anchor_id
            for anchor_id in (
                _normalize_ref_to_object_id(item, ref_map=ref_map, player_object_id=player_object_id)
                for item in list(intent.anchor_refs or [])
            )
            if anchor_id is not None
        ]
        payload = {
            "window_key": window_key,
            "status": "open",
            "priority": intent.priority,
            "earliest_turn": intent.earliest_turn,
            "latest_turn": min(intent.latest_turn, intent.earliest_turn + MAX_CONSEQUENCE_WINDOW_SPAN),
            "shows": 0,
            "max_shows": intent.max_shows,
            "depth": max(int(depth), 0),
            "target_object_ids": target_object_ids,
            "anchor_object_ids": anchor_object_ids,
            "source_turn": turn_index,
            "source_event_ids": list(source_event_ids),
            "source_fact_keys": list(source_fact_keys),
            "opened_obligation_keys": list(opened_obligation_keys),
            "created_by_turn_kind": str(created_by_turn_kind or "player"),
            "provenance_status": "exact",
            "reason_kind": str(intent.reason_kind or "followup"),
            "text": str(getattr(legacy_seed, "text", "") or intent.reason_kind or window_key).strip(),
        }
        existing_row = existing_by_key.get(window_key)
        if existing_row is None:
            db.add(
                models.ObjectModel(
                    session_id=session_id,
                    type=CONSEQUENCE_WINDOW_OBJECT_TYPE,
                    name=f"{CONSEQUENCE_WINDOW_NAME_PREFIX}:{window_key}",
                    data=payload,
                )
            )
        else:
            current_payload = _coerce_window_payload(dict(existing_row.data or {})) or {}
            payload["shows"] = int(current_payload.get("shows") or 0)
            if str(current_payload.get("status") or "").strip() in {"shown", "suppressed"}:
                payload["status"] = str(current_payload.get("status") or "open").strip()
            if "suppressed_until_turn" in current_payload:
                payload["suppressed_until_turn"] = current_payload["suppressed_until_turn"]
            existing_row.name = f"{CONSEQUENCE_WINDOW_NAME_PREFIX}:{window_key}"
            existing_row.data = payload
        created_payloads.append(payload)
    if created_payloads:
        db.flush()
    return created_payloads


def _select_due_window_payloads(
    db: OrmSession,
    session_id: uuid.UUID,
    *,
    turn_index: int,
    budget: int,
) -> list[dict[str, Any]]:
    if budget <= 0:
        return []

    payloads = list_consequence_window_payloads(db, session_id)
    scored: list[tuple[float, int, int, str, dict[str, Any]]] = []
    for payload in payloads:
        if int(payload.get("depth") or 0) > MAX_CONSEQUENCE_CHAIN_DEPTH:
            continue
        status = str(payload.get("status") or "open").strip().lower()
        if status not in {"open", "shown"}:
            if status == "suppressed":
                suppressed_until_turn = payload.get("suppressed_until_turn")
                if isinstance(suppressed_until_turn, int) and turn_index > suppressed_until_turn:
                    status = "open"
                else:
                    continue
            else:
                continue
        earliest_turn = int(payload.get("earliest_turn") or 0)
        latest_turn = int(payload.get("latest_turn") or 0)
        if earliest_turn > turn_index or latest_turn < turn_index:
            continue
        shows = int(payload.get("shows") or 0)
        max_shows = max(int(payload.get("max_shows") or 0), 0)
        if shows >= max_shows:
            continue

        priority_score = float(PRIORITY_WEIGHT.get(str(payload.get("priority") or "med"), PRIORITY_WEIGHT["med"]))
        window = max(latest_turn - earliest_turn + 1, 1)
        turns_left = max(latest_turn - turn_index, 0)
        expiry_pressure = 1.0 - min(turns_left / window, 1.0)
        show_pressure = 1.0 - min(shows / max(max_shows, 1), 1.0)
        window_key = str(payload.get("window_key") or "").strip()
        score = priority_score + 0.35 * expiry_pressure + 0.2 * show_pressure + _stable_jitter(window_key, turn_index)
        scored.append((score, latest_turn, int(payload.get("source_turn") or 0), window_key, payload))

    scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    return [item[-1] for item in scored[:budget]]


def select_due_consequence_window_payloads(
    db: OrmSession,
    session_id: uuid.UUID,
    *,
    turn_index: int,
    budget: int = 2,
) -> list[dict[str, Any]]:
    return _select_due_window_payloads(
        db,
        session_id,
        turn_index=turn_index,
        budget=budget,
    )


def select_due_consequences_from_state_json(
    state_json: dict[str, Any],
    turn_index: int,
    budget: int = 2,
) -> list[schemas.ConsequenceSeed]:
    """Legacy selector for non-persisted pending_consequences payloads."""
    if budget <= 0:
        return []

    scored: list[tuple[float, int, int, str, schemas.ConsequenceSeed]] = []
    for seed in _iter_pending_seeds(state_json):
        if seed.depth > MAX_CONSEQUENCE_CHAIN_DEPTH:
            continue
        if seed.earliest_turn > turn_index or seed.latest_turn < turn_index:
            continue
        if seed.shows >= max(seed.max_shows, 0):
            continue
        if seed.last_shown_turn is not None and (turn_index - seed.last_shown_turn) < COOLDOWN_TURNS:
            continue

        priority_score = float(PRIORITY_WEIGHT.get(seed.priority, PRIORITY_WEIGHT["med"]))
        window = max(seed.latest_turn - seed.earliest_turn + 1, 1)
        turns_left = max(seed.latest_turn - turn_index, 0)
        expiry_pressure = 1.0 - min(turns_left / window, 1.0)
        show_pressure = 1.0 - min(seed.shows / max(seed.max_shows, 1), 1.0)
        score = priority_score + 0.35 * expiry_pressure + 0.2 * show_pressure + _stable_jitter(seed.id, turn_index)
        scored.append((score, seed.latest_turn, seed.created_turn, seed.id, seed))

    scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    return [item[-1] for item in scored[:budget]]


def enqueue_consequences(
    state_json: dict[str, Any],
    seeds: list[schemas.ConsequenceSeed],
    *,
    resolved_parent_ids: list[str] | None = None,
    current_turn: int = 0,
) -> None:
    """Adds seeds to state_json["pending_consequences"], deduplicates by id."""
    pending = _iter_pending_seeds(state_json)
    pending_by_id = {seed.id: seed for seed in pending}
    normalized_current_turn = current_turn if isinstance(current_turn, int) and not isinstance(current_turn, bool) else 0
    normalized_current_turn = max(normalized_current_turn, 0)
    resolved_parent_set = {
        str(raw_id).strip()
        for raw_id in (resolved_parent_ids or [])
        if str(raw_id).strip()
    }
    parent_depth = -1
    if resolved_parent_set:
        for parent_id in resolved_parent_set:
            parent_seed = pending_by_id.get(parent_id)
            if parent_seed is None:
                continue
            parent_depth = max(parent_depth, max(int(parent_seed.depth), 0))
    next_depth = max(parent_depth + 1, 0)

    for seed in seeds:
        if seed.id in pending_by_id:
            continue

        normalized_earliest = max(seed.earliest_turn, normalized_current_turn)
        normalized_latest = max(seed.latest_turn, normalized_earliest)
        normalized_latest = min(
            normalized_latest,
            normalized_earliest + MAX_CONSEQUENCE_WINDOW_SPAN,
        )
        seed_with_window = seed.model_copy(
            update={
                "earliest_turn": normalized_earliest,
                "latest_turn": normalized_latest,
            }
        )
        seed_with_depth = seed_with_window.model_copy(update={"depth": next_depth})
        pending_by_id[seed.id] = seed_with_depth
        pending.append(seed_with_depth)
    _dump_pending_seeds(state_json, pending)


def select_due_consequences(
    db: OrmSession,
    session_id: uuid.UUID,
    *,
    turn_index: int,
    budget: int = 2,
) -> list[schemas.ConsequenceSeed]:
    """Runtime selector backed only by persisted consequence windows."""
    due_payloads = _select_due_window_payloads(db, session_id, turn_index=turn_index, budget=budget)
    return [
        seed
        for seed in (
            _window_payload_to_seed(payload)
            for payload in due_payloads
        )
        if seed is not None
    ]


def apply_consequence_results(
    state_json: dict[str, Any],
    resolved_ids: list[str],
    shown_ids: list[str],
    turn_index: int,
    *,
    db: OrmSession | None = None,
    session_id: uuid.UUID | None = None,
) -> None:
    """Removes resolved, increments shows, cleans expired (latest_turn < turn_index)."""
    if db is not None and session_id is not None:
        resolved = {str(item).strip() for item in resolved_ids if str(item).strip()}
        shown = {str(item).strip() for item in shown_ids if str(item).strip()}
        rows = db.execute(
            select(models.ObjectModel).where(
                models.ObjectModel.session_id == session_id,
                models.ObjectModel.type == CONSEQUENCE_WINDOW_OBJECT_TYPE,
            )
        ).scalars().all()
        for row in rows:
            payload = _coerce_window_payload(dict(row.data or {}))
            if payload is None:
                continue
            window_key = str(payload.get("window_key") or "").strip()
            if not window_key:
                continue
            if window_key in resolved:
                payload["status"] = "resolved"
                row.data = payload
                continue
            if int(payload.get("latest_turn") or 0) < turn_index:
                payload["status"] = "expired"
                row.data = payload
                continue
            if window_key in shown and int(payload.get("shows") or 0) < max(int(payload.get("max_shows") or 0), 0):
                payload["shows"] = int(payload.get("shows") or 0) + 1
                payload["status"] = "shown"
                row.data = payload
                continue
            if str(payload.get("status") or "").strip() == "shown" and int(payload.get("shows") or 0) < max(int(payload.get("max_shows") or 0), 0):
                row.data = payload
        db.flush()
        return

    resolved = {str(item).strip() for item in resolved_ids if str(item).strip()}
    shown = {str(item).strip() for item in shown_ids if str(item).strip()}

    updated: list[schemas.ConsequenceSeed] = []
    for seed in _iter_pending_seeds(state_json):
        if seed.id in resolved:
            continue
        if seed.depth > MAX_CONSEQUENCE_CHAIN_DEPTH:
            continue
        if seed.latest_turn < turn_index:
            continue
        if seed.id in shown and seed.shows < max(seed.max_shows, 0):
            seed.shows += 1
            seed.last_shown_turn = turn_index
        updated.append(seed)

    _dump_pending_seeds(state_json, updated)


def _get_db_session(session_row: models.SessionModel) -> OrmSession | None:
    return object_session(session_row)


def get_object_by_type(
    session_row: models.SessionModel,
    object_type: str,
) -> models.ObjectModel | None:
    db = _get_db_session(session_row)
    if db is None:
        return None
    return db.execute(
        select(models.ObjectModel)
        .where(
            models.ObjectModel.session_id == session_row.id,
            models.ObjectModel.type == object_type,
        )
        .order_by(models.ObjectModel.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def _extract_tags(data: dict[str, Any]) -> set[str]:
    tags: set[str] = set()
    for key in ("tags", "object_tags"):
        raw = data.get(key)
        if isinstance(raw, str):
            text = raw.strip().lower()
            if text:
                tags.add(text)
        elif isinstance(raw, list):
            for item in raw:
                text = str(item).strip().lower()
                if text:
                    tags.add(text)
    return tags


def _lookup_data_flag_value(data: dict[str, Any], key: str) -> Any:
    current: Any = data
    for part in [chunk for chunk in key.split(".") if chunk]:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _matches_data_flag(data: dict[str, Any], raw_flag: Any) -> bool:
    if raw_flag is None:
        return True
    if isinstance(raw_flag, str):
        text = raw_flag.strip()
        if not text:
            return True
        if "=" in text:
            key, expected = text.split("=", 1)
            value = _lookup_data_flag_value(data, key.strip())
            return str(value).strip().lower() == expected.strip().lower()
        return bool(_lookup_data_flag_value(data, text))
    if not isinstance(raw_flag, dict):
        return False

    key = str(raw_flag.get("key") or raw_flag.get("path") or raw_flag.get("field") or "").strip()
    if not key:
        return False
    value = _lookup_data_flag_value(data, key)

    if "exists" in raw_flag:
        expected_exists = bool(raw_flag.get("exists"))
        if (value is not None) != expected_exists:
            return False
    if "equals" in raw_flag or "value" in raw_flag:
        expected_value = raw_flag.get("equals", raw_flag.get("value"))
        if value != expected_value:
            return False
    if "not_equals" in raw_flag and value == raw_flag.get("not_equals"):
        return False
    raw_in = raw_flag.get("in")
    if isinstance(raw_in, list) and value not in raw_in:
        return False
    if (
        "exists" not in raw_flag
        and "equals" not in raw_flag
        and "value" not in raw_flag
        and "not_equals" not in raw_flag
        and not isinstance(raw_in, list)
    ):
        return bool(value)
    return True


def _normalize_trigger_tags(raw_tags: Any) -> set[str]:
    if isinstance(raw_tags, str):
        text = raw_tags.strip().lower()
        return {text} if text else set()
    if not isinstance(raw_tags, list):
        return set()
    result: set[str] = set()
    for item in raw_tags:
        text = str(item).strip().lower()
        if text:
            result.add(text)
    return result


def _resolve_ref_to_object(
    session_row: models.SessionModel,
    raw_ref: Any,
) -> models.ObjectModel | None:
    db = _get_db_session(session_row)
    if db is None:
        return None

    ref_text = str(raw_ref or "").strip()
    if not ref_text:
        return None
    if ref_text == SESSION_PLAYER_REF:
        player_raw = (session_row.state_json or {}).get("player_object_id")
        ref_text = str(player_raw or "").strip()
        if not ref_text:
            return None
    try:
        object_id = uuid.UUID(ref_text)
    except (ValueError, TypeError, AttributeError):
        return None
    return db.get(models.ObjectModel, (session_row.id, object_id))


def _matches_trigger(
    candidate: dict[str, Any],
    trigger: dict[str, Any],
) -> tuple[bool, str]:
    trigger_name = str(trigger.get("trigger") or trigger.get("id") or "").strip()
    if not trigger_name:
        return False, ""

    op_filter = trigger.get("ops")
    if op_filter is None:
        op_filter = trigger.get("op")
    if isinstance(op_filter, str):
        if candidate.get("op") != op_filter:
            return False, trigger_name
    elif isinstance(op_filter, list):
        allowed = {str(item).strip() for item in op_filter if str(item).strip()}
        if allowed and candidate.get("op") not in allowed:
            return False, trigger_name

    object_type = str(trigger.get("object_type") or "").strip()
    if object_type and candidate.get("object_type") != object_type:
        return False, trigger_name

    required_tags = _normalize_trigger_tags(trigger.get("object_tags"))
    if required_tags and not required_tags.issubset(set(candidate.get("object_tags") or set())):
        return False, trigger_name

    if not _matches_data_flag(candidate.get("data") or {}, trigger.get("data_flag")):
        return False, trigger_name
    return True, trigger_name


def _build_candidates_from_op(
    op: dict[str, Any],
    session_row: models.SessionModel,
) -> list[dict[str, Any]]:
    op_name = str(op.get("op") or "").strip()
    if not op_name:
        return []

    if op_name == "object.create":
        data = dict(op.get("data") or {}) if isinstance(op.get("data"), dict) else {}
        object_id = str(op.get("object_id") or op.get("ref") or "").strip()
        object_type = str(op.get("type") or "").strip()
        if not object_id or not object_type:
            return []
        return [
            {
                "op": op_name,
                "object_id": object_id,
                "object_type": object_type,
                "object_tags": _extract_tags(data),
                "data": data,
            }
        ]

    if op_name == "object.update":
        object_row = _resolve_ref_to_object(session_row, op.get("object"))
        if object_row is None:
            return []
        patch = op.get("patch")
        patch_data = dict(patch) if isinstance(patch, dict) else {}
        data = dict(object_row.data or {})
        data.update(patch_data)
        return [
            {
                "op": op_name,
                "object_id": str(object_row.object_id),
                "object_type": str(object_row.type),
                "object_tags": _extract_tags(data),
                "data": data,
            }
        ]

    if op_name in {"link.create", "link.close"}:
        source_row = _resolve_ref_to_object(
            session_row,
            op.get("from") if op.get("from") is not None else op.get("from_ref"),
        )
        if source_row is None:
            return []
        target_row = _resolve_ref_to_object(
            session_row,
            op.get("to") if op.get("to") is not None else op.get("to_ref"),
        )
        source_data = dict(source_row.data or {})
        candidate_data: dict[str, Any] = dict(source_data)
        link_type = str(op.get("type") or "").strip()
        if link_type:
            candidate_data["link_type"] = link_type
        if target_row is not None:
            candidate_data["link_to_id"] = str(target_row.object_id)
        else:
            raw_target_ref = str(op.get("to") if op.get("to") is not None else op.get("to_ref") or "").strip()
            if raw_target_ref:
                candidate_data["link_to_id"] = raw_target_ref
        candidate_data["link_bidirectional"] = bool(op.get("bidirectional", False))
        return [
            {
                "op": op_name,
                "object_id": str(source_row.object_id),
                "object_type": str(source_row.type),
                "object_tags": _extract_tags(candidate_data),
                "data": candidate_data,
            }
        ]

    if op_name == "event.create":
        payload = op.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        target_row = None
        for key in ("object_id", "target_object_id", "npc_id", "faction_id", "zone_id", "item_id"):
            target_row = _resolve_ref_to_object(session_row, payload.get(key))
            if target_row is not None:
                break
        if target_row is None:
            target_row = _resolve_ref_to_object(session_row, op.get("scope"))
        if target_row is None:
            return []
        data = dict(target_row.data or {})
        return [
            {
                "op": op_name,
                "object_id": str(target_row.object_id),
                "object_type": str(target_row.type),
                "object_tags": _extract_tags(data),
                "data": data,
            }
        ]

    return []


def resolve_applied_op_refs(
    applied_ops: list[dict[str, Any]],
    ref_map: dict[str, str],
) -> list[dict[str, Any]]:
    if not ref_map:
        return [dict(op) for op in applied_ops if isinstance(op, dict)]

    resolved: list[dict[str, Any]] = []
    for raw_op in applied_ops:
        if not isinstance(raw_op, dict):
            continue
        op = dict(raw_op)
        op_name = str(op.get("op") or "").strip()
        if op_name == "object.create":
            raw_ref = str(op.get("ref") or "").strip()
            if raw_ref and raw_ref in ref_map:
                op["object_id"] = ref_map[raw_ref]
        elif op_name == "object.update":
            raw_object = str(op.get("object") or "").strip()
            if raw_object and raw_object in ref_map:
                op["object"] = ref_map[raw_object]
        elif op_name in {"link.create", "link.close"}:
            raw_from = str(op.get("from") or op.get("from_ref") or "").strip()
            if raw_from and raw_from in ref_map:
                op["from"] = ref_map[raw_from]
            raw_to = str(op.get("to") or op.get("to_ref") or "").strip()
            if raw_to and raw_to in ref_map:
                op["to"] = ref_map[raw_to]
        elif op_name == "event.create":
            raw_scope = str(op.get("scope") or "").strip()
            if raw_scope and raw_scope in ref_map:
                op["scope"] = ref_map[raw_scope]
            payload = op.get("payload")
            if isinstance(payload, dict):
                payload_copy = dict(payload)
                for key in ("object_id", "target_object_id", "npc_id", "faction_id", "zone_id", "item_id"):
                    raw_value = str(payload_copy.get(key) or "").strip()
                    if raw_value and raw_value in ref_map:
                        payload_copy[key] = ref_map[raw_value]
                op["payload"] = payload_copy
        resolved.append(op)
    return resolved


def detect_structural_signals(
    applied_ops: list[dict[str, Any]],
    session: models.SessionModel,
    turn_index: int,
) -> list[schemas.StructuralSignal]:
    """
    Reads world_constitution from objects.
    Matches applied_ops against structural_triggers by object_type + object_tags + data_flag.
    Returns raw signals — DOES NOT add to the queue.
    """
    constitution = get_object_by_type(session, "world_constitution")
    if constitution is None:
        return []

    triggers = (constitution.data or {}).get("structural_triggers")
    if not isinstance(triggers, list) or not triggers:
        return []

    signals: list[schemas.StructuralSignal] = []
    seen: set[tuple[str, str]] = set()
    for op in applied_ops:
        if not isinstance(op, dict):
            continue
        candidates = _build_candidates_from_op(op, session)
        for candidate in candidates:
            for raw_trigger in triggers:
                if not isinstance(raw_trigger, dict):
                    continue
                matched, trigger_name = _matches_trigger(candidate, raw_trigger)
                if not matched or not trigger_name:
                    continue
                object_id = str(candidate.get("object_id") or "").strip()
                if not object_id:
                    continue
                key = (trigger_name, object_id)
                if key in seen:
                    continue
                seen.add(key)
                signals.append(
                    schemas.StructuralSignal(
                        trigger=trigger_name,
                        object_id=object_id,
                        turn=int(turn_index),
                    )
                )
    return signals


__all__ = [
    "COOLDOWN_TURNS",
    "MAX_CONSEQUENCE_WINDOW_SPAN",
    "MAX_CONSEQUENCE_CHAIN_DEPTH",
    "PRIORITY_WEIGHT",
    "CONSEQUENCE_WINDOW_OBJECT_TYPE",
    "ensure_persisted_consequence_windows",
    "list_consequence_window_payloads",
    "materialize_consequence_windows",
    "select_due_consequence_window_payloads",
    "select_due_consequences_from_state_json",
    "enqueue_consequences",
    "select_due_consequences",
    "apply_consequence_results",
    "resolve_applied_op_refs",
    "detect_structural_signals",
    "get_object_by_type",
]
