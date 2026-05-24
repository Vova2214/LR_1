from __future__ import annotations

"""Claims module (Phase 1b implementation)."""

import uuid

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from . import crud_core as _core
from .crud_shared import _coerce_state_payload, _create_internal_turn_row, _extract_in_game_time
from . import models, schemas


def create_claim(
    db: Session,
    session_id: uuid.UUID,
    payload: schemas.ClaimCreateIn,
) -> schemas.ClaimCreateOut:
    result: schemas.ClaimCreateOut | None = None
    current_turn = 0
    in_game_day: int | None = None
    in_game_minute: int | None = None
    resolved_location_id: uuid.UUID | None = None
    listener_objects: dict[uuid.UUID, models.ObjectModel | object] = {}

    with db.begin():
        _core._acquire_session_turn_lock(db, session_id)
        session_row = _core._require_session(db, session_id, for_update=True)
        session_state = _core._recover_abandoned_pending_turn_locked(
            db=db,
            session_id=session_id,
            session_row=session_row,
            state_payload=_coerce_state_payload(getattr(session_row, "state_json", {})),
        )
        current_turn = max(_core._safe_int(getattr(session_row, "turn_index", 0)) or 0, 0) + 1
        in_game_day, in_game_minute = _extract_in_game_time(session_state)

        speaker_object = _core._require_object(db, session_id, payload.speaker_id)
        listener_ids = list(payload.listener_ids)
        if not listener_ids:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="At least one listener_id/listener_ids entry is required",
            )
        for listener_id in listener_ids:
            listener_objects[listener_id] = _core._require_object(db, session_id, listener_id)

        if payload.about_object_id is not None:
            _core._require_object(db, session_id, payload.about_object_id)
        if payload.location_id is not None:
            _core._require_object(db, session_id, payload.location_id)
            resolved_location_id = payload.location_id
        else:
            resolved_location_id = _core._infer_actor_zone_id(
                db,
                session_id,
                payload.speaker_id,
            )
            if resolved_location_id is None:
                _core.logger.warning(
                    "Unable to infer claim location_id for session_id=%s speaker_id=%s turn_index=%s",
                    session_id,
                    payload.speaker_id,
                    current_turn,
                )

        claim_data: dict[str, object] = {
            "text": payload.text,
            "confidence": payload.confidence,
            "about_object_id": str(payload.about_object_id) if payload.about_object_id else None,
            "location_id": str(resolved_location_id) if resolved_location_id else None,
        }

        claim_object = models.ObjectModel(
            session_id=session_id,
            type="claim",
            name="Claim",
            data=claim_data,
        )
        db.add(claim_object)
        db.flush()

        asserted_link = models.LinkModel(
            session_id=session_id,
            from_object_id=payload.speaker_id,
            to_object_id=claim_object.object_id,
            type="asserted",
            data={},
            valid_from_turn=current_turn,
            valid_to_turn=None,
        )
        heard_links = [
            models.LinkModel(
                session_id=session_id,
                from_object_id=listener_id,
                to_object_id=claim_object.object_id,
                type="heard",
                data={},
                valid_from_turn=current_turn,
                valid_to_turn=None,
            )
            for listener_id in listener_ids
        ]
        db.add_all([asserted_link, *heard_links])

        speaker_name = str(getattr(speaker_object, "name", "") or "").strip() or "Unknown"
        _create_internal_turn_row(
            db,
            session_id,
            session_row,
            turn_index=current_turn,
            user_input=_core._truncate_text(
                f"[internal claim.create] {speaker_name}: {payload.text}",
                _core.EMBED_SNIPPET_MAX_CHARS,
            ),
            ai_text=_core._truncate_text(
                f"(claim) {speaker_name}: {payload.text}",
                _core.EMBED_SNIPPET_MAX_CHARS,
            ),
            note="internal_claim_create",
            applied_ops=[
                {
                    "op": "object.create",
                    "ref": str(claim_object.object_id),
                    "type": "claim",
                    "name": "Claim",
                    "data": dict(claim_data),
                },
                {
                    "op": "link.create",
                    "from": str(payload.speaker_id),
                    "to": str(claim_object.object_id),
                    "type": "asserted",
                    "data": {},
                    "valid_from_turn": current_turn,
                    "valid_to_turn": None,
                },
                *[
                    {
                        "op": "link.create",
                        "from": str(listener_id),
                        "to": str(claim_object.object_id),
                        "type": "heard",
                        "data": {},
                        "valid_from_turn": current_turn,
                        "valid_to_turn": None,
                    }
                    for listener_id in listener_ids
                ],
            ],
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
        )
        db.flush()
        _core._entities._add_internal_object_created_event(
            db,
            session_id=session_id,
            turn_index=current_turn,
            object_row=claim_object,
            object_data=claim_data,
        )
        _core._entities._add_internal_link_event(
            db,
            session_id=session_id,
            turn_index=current_turn,
            event_type="LINK_CREATED",
            from_object_id=payload.speaker_id,
            to_object_id=claim_object.object_id,
            link_type="asserted",
            from_object=speaker_object,
            to_object=claim_object,
            link_data={},
            valid_from_turn=current_turn,
            valid_to_turn=None,
        )
        for listener_id in listener_ids:
            _core._entities._add_internal_link_event(
                db,
                session_id=session_id,
                turn_index=current_turn,
                event_type="LINK_CREATED",
                from_object_id=listener_id,
                to_object_id=claim_object.object_id,
                link_type="heard",
                from_object=listener_objects.get(listener_id),
                to_object=claim_object,
                link_data={},
                valid_from_turn=current_turn,
                valid_to_turn=None,
            )

        claim_event = models.EventModel(
            session_id=session_id,
            turn_index=current_turn,
            type="CLAIM_CREATED",
            scope_object_id=resolved_location_id,
            payload={
                "claim_object_id": str(claim_object.object_id),
                "speaker_id": str(payload.speaker_id),
                "listener_id": str(listener_ids[0]),
                "listener_ids": [str(listener_id) for listener_id in listener_ids],
                "text": payload.text,
                "confidence": payload.confidence,
            },
        )
        db.add(claim_event)

        db.flush()
        result = schemas.ClaimCreateOut(
            claim_object_id=claim_object.object_id,
            event_id=claim_event.event_id,
            turn_index=current_turn,
        )

        if _core.USE_EMBEDDINGS:
            _core._entities._enqueue_claim_text_refresh_event(
                db,
                session_id=session_id,
                object_id=claim_object.object_id,
            )
        _core._entities._enqueue_turn_chronicle_sync_event(
            db,
            session_id=session_id,
            turn_index=current_turn,
        )

    if result is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Claim creation failed")

    return result


__all__ = ["create_claim"]
