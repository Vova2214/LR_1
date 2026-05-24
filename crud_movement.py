from __future__ import annotations

"""Movement module (Phase 1b implementation).

Core movement logic lives here; ``crud_core.move_player`` delegates to this
module to preserve the public surface while reducing monolith size.
"""

from dataclasses import dataclass
import uuid
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from . import crud_core as _core
from . import models, schemas
from .constants import LOCATED_IN_LINK_TYPE
from .crud_shared import _build_internal_turn_ai_json, _coerce_state_payload
from .db import DEFAULT_SPAWN_ZONE_NAME
from .domain import geography_policy as geography_domain


@dataclass(slots=True)
class TravelApplyResult:
    final_node_id: uuid.UUID
    applied_ops: list[dict[str, Any]]
    move_event: models.EventModel | None
    created_node_id: uuid.UUID | None = None


def apply_travel_resolution(
    db: Session,
    session_id: uuid.UUID,
    *,
    player_object_id: uuid.UUID,
    current_node_id: uuid.UUID,
    resolution: geography_domain.TravelResolution,
    world_profile: geography_domain.GeographyWorldProfile,
    new_turn: int,
    in_game_day: int,
    in_game_minute: int,
) -> TravelApplyResult:
    if resolution.final_node_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Resolved travel has no final node",
        )

    applied_ops: list[dict[str, Any]] = []
    move_event: models.EventModel | None = None
    created_node_id: uuid.UUID | None = None

    current_node = _core._shared._get_object(db, session_id, current_node_id)
    if current_node is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Current travel node is missing",
        )

    target_node = _core._shared._get_object(db, session_id, resolution.final_node_id)
    if target_node is None and resolution.expanded_node_name:
        geo_payload = {
            "geo": {
                "node_kind": resolution.expanded_node_kind or world_profile.frontier_expansion_policy.default_node_kind,
                "parent_node_id": str(current_node_id),
                "discovery_state": "discovered",
                "frontier_capable": world_profile.frontier_expansion_policy.default_frontier_capable,
            }
        }
        target_node = models.ObjectModel(
            session_id=session_id,
            object_id=resolution.resolved_target or resolution.final_node_id,
            type="zone",
            name=resolution.expanded_node_name,
            data=geo_payload,
        )
        db.add(target_node)
        db.flush()
        created_node_id = target_node.object_id
        applied_ops.append(
            {
                "op": "object.create",
                "ref": str(target_node.object_id),
                "type": "zone",
                "name": target_node.name,
                "data": geo_payload,
            }
        )
        _core._entities._add_internal_object_created_event(
            db,
            session_id=session_id,
            turn_index=new_turn,
            object_row=target_node,
            object_data=geo_payload,
        )
        if _core.USE_EMBEDDINGS:
            _core._entities._enqueue_zone_profile_refresh_event(
                db,
                session_id=session_id,
                object_id=target_node.object_id,
            )

        route_kind = resolution.expanded_route_kind or world_profile.frontier_expansion_policy.default_route_kind
        route_data = {
            "geo": {
                "route_kind": route_kind,
                "travel_cost": world_profile.frontier_expansion_policy.default_travel_cost,
                "travel_cost_unit": world_profile.frontier_expansion_policy.default_travel_cost_unit,
                "bidirectional": world_profile.frontier_expansion_policy.default_bidirectional,
            }
        }
        forward_link = models.LinkModel(
            session_id=session_id,
            from_object_id=current_node_id,
            to_object_id=target_node.object_id,
            type=route_kind,
            data=route_data,
            valid_from_turn=new_turn,
            valid_to_turn=None,
        )
        db.add(forward_link)
        applied_ops.append(
            {
                "op": "link.create",
                "from": str(current_node_id),
                "to": str(target_node.object_id),
                "type": route_kind,
                "data": route_data,
            }
        )
        _core._entities._add_internal_link_event(
            db,
            session_id=session_id,
            turn_index=new_turn,
            event_type="LINK_CREATED",
            from_object_id=current_node_id,
            to_object_id=target_node.object_id,
            link_type=route_kind,
            from_object=current_node,
            to_object=target_node,
            link_data=route_data,
            valid_from_turn=new_turn,
            valid_to_turn=None,
            source="travel_control",
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
        )
        if world_profile.frontier_expansion_policy.default_bidirectional:
            reverse_link = models.LinkModel(
                session_id=session_id,
                from_object_id=target_node.object_id,
                to_object_id=current_node_id,
                type=route_kind,
                data=route_data,
                valid_from_turn=new_turn,
                valid_to_turn=None,
            )
            db.add(reverse_link)
            applied_ops.append(
                {
                    "op": "link.create",
                    "from": str(target_node.object_id),
                    "to": str(current_node_id),
                    "type": route_kind,
                    "data": route_data,
                }
            )
            _core._entities._add_internal_link_event(
                db,
                session_id=session_id,
                turn_index=new_turn,
                event_type="LINK_CREATED",
                from_object_id=target_node.object_id,
                to_object_id=current_node_id,
                link_type=route_kind,
                from_object=target_node,
                to_object=current_node,
                link_data=route_data,
                valid_from_turn=new_turn,
                valid_to_turn=None,
                source="travel_control",
                in_game_day=in_game_day,
                in_game_minute=in_game_minute,
            )

    if target_node is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Resolved travel target is missing",
        )

    if resolution.final_node_id == current_node_id:
        return TravelApplyResult(
            final_node_id=current_node_id,
            applied_ops=applied_ops,
            move_event=None,
            created_node_id=created_node_id,
        )

    active_locations = _core._shared._close_player_active_located_in_links(
        db,
        session_id,
        player_object_id,
        closed_at_turn=new_turn,
    )
    for active_location in active_locations:
        previous_zone_id = getattr(active_location, "to_object_id", None)
        if not isinstance(previous_zone_id, uuid.UUID):
            continue
        applied_ops.append(
            {
                "op": "link.close",
                "from": str(player_object_id),
                "to": str(previous_zone_id),
                "type": LOCATED_IN_LINK_TYPE,
            }
        )
        previous_zone = _core._shared._get_object(db, session_id, previous_zone_id)
        _core._entities._add_internal_link_event(
            db,
            session_id=session_id,
            turn_index=new_turn,
            event_type="LINK_CLOSED",
            from_object_id=player_object_id,
            to_object_id=previous_zone_id,
            link_type=LOCATED_IN_LINK_TYPE,
            from_object=None,
            to_object=previous_zone,
            source="travel_control",
            in_game_day=in_game_day,
            in_game_minute=in_game_minute,
        )

    new_location_link = models.LinkModel(
        session_id=session_id,
        from_object_id=player_object_id,
        to_object_id=target_node.object_id,
        type=LOCATED_IN_LINK_TYPE,
        data={},
        valid_from_turn=new_turn,
        valid_to_turn=None,
    )
    db.add(new_location_link)
    applied_ops.append(
        {
            "op": "link.create",
            "from": str(player_object_id),
            "to": str(target_node.object_id),
            "type": LOCATED_IN_LINK_TYPE,
            "data": {},
        }
    )
    _core._entities._add_internal_link_event(
        db,
        session_id=session_id,
        turn_index=new_turn,
        event_type="LINK_CREATED",
        from_object_id=player_object_id,
        to_object_id=target_node.object_id,
        link_type=LOCATED_IN_LINK_TYPE,
        from_object=None,
        to_object=target_node,
        link_data={},
        valid_from_turn=new_turn,
        valid_to_turn=None,
        source="travel_control",
        in_game_day=in_game_day,
        in_game_minute=in_game_minute,
    )

    move_event = models.EventModel(
        session_id=session_id,
        turn_index=new_turn,
        type="MOVE",
        scope_object_id=target_node.object_id,
        payload={
            "from_object_id": str(current_node_id),
            "from_name": current_node.name,
            "to_object_id": str(target_node.object_id),
            "to_name": target_node.name,
            "travel_outcome_mode": resolution.outcome_mode,
            "travel_cost": resolution.consumed_cost,
            "travel_cost_unit": resolution.cost_unit,
            "source": "travel_control",
            "in_game_time": {"day": in_game_day, "minute": in_game_minute},
        },
    )
    db.add(move_event)
    db.flush()
    return TravelApplyResult(
        final_node_id=target_node.object_id,
        applied_ops=applied_ops,
        move_event=move_event,
        created_node_id=created_node_id,
    )


def move_player(
    db: Session,
    session_id: uuid.UUID,
    payload: schemas.MoveIn,
) -> schemas.MoveOut:
    with _core._session_turn_runtime_lock(db, session_id):
        return _move_player_locked(db, session_id, payload)


def _move_player_locked(
    db: Session,
    session_id: uuid.UUID,
    payload: schemas.MoveIn,
) -> schemas.MoveOut:
    move_out: schemas.MoveOut | None = None
    target_zone_id: uuid.UUID | None = None
    next_day = 0
    next_minute = 0
    new_turn = 0
    with db.begin():
        _core._acquire_session_turn_lock(db, session_id)
        session_row = _core._require_session(db, session_id, for_update=True)
        state_payload = _coerce_state_payload(getattr(session_row, "state_json", {}))
        state_payload = _core._recover_abandoned_pending_turn_locked(
            db=db,
            session_id=session_id,
            session_row=session_row,
            state_payload=state_payload,
        )

        new_turn = session_row.turn_index + 1
        session_row.turn_index = new_turn

        next_day, next_minute, time_scale = _core._normalize_time_payload(state_payload)
        state_payload["time"] = {"day": next_day, "minute": next_minute}
        state_payload["time_scale"] = time_scale

        player_object_id_raw = state_payload.get("player_object_id")
        if player_object_id_raw is None:
            player_object_id = db.execute(
                select(models.ObjectModel.object_id)
                .where(
                    models.ObjectModel.session_id == session_id,
                    models.ObjectModel.type == "player",
                )
                .order_by(models.ObjectModel.created_at.asc())
                .limit(1)
            ).scalar_one_or_none()
            if player_object_id is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Session has no player object",
                )
            state_payload["player_object_id"] = str(player_object_id)
        else:
            try:
                player_object_id = uuid.UUID(str(player_object_id_raw))
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Invalid player_object_id in session state",
                ) from exc

        session_row.state_json = state_payload
        turn_row = models.TurnModel(
            session_id=session_id,
            turn_index=new_turn,
            user_input=f"(move to {payload.to_name} via {payload.relation})",
            ai_text=None,
            ai_json={},
        )
        db.add(turn_row)
        # Materialize the turn row before any side effect can emit events against
        # this turn_index (MOVE itself, or NPC TTL cleanup).
        db.flush([turn_row])
        ttl_cleanup = _core._coerce_ttl_cleanup_result(
            _core.cleanup_ephemeral_npcs(
                db,
                session_id,
                new_turn,
                in_game_day=next_day,
                in_game_minute=next_minute,
            )
        )

        active_locations = _core._shared._get_active_located_in_links(
            db,
            session_id,
            player_object_id,
        )
        had_active_player_location = bool(active_locations)
        current_location_link = (
            active_locations[-1]
            if active_locations
            else _core._shared._get_latest_located_in_link(db, session_id, player_object_id)
        )
        current_location = (
            _core._shared._get_object(db, session_id, current_location_link.to_object_id)
            if current_location_link is not None
            else None
        )
        current_location_created = False
        if current_location is None:
            current_location = db.execute(
                select(models.ObjectModel)
                .where(
                    models.ObjectModel.session_id == session_id,
                    models.ObjectModel.type == "zone",
                )
                .order_by(models.ObjectModel.created_at.asc())
                .limit(1)
            ).scalar_one_or_none()
            if current_location is None:
                current_location_created = True
                current_location = models.ObjectModel(
                    session_id=session_id,
                    type="zone",
                    name=DEFAULT_SPAWN_ZONE_NAME,
                    data={},
                )
                db.add(current_location)
                db.flush()
                if _core.USE_EMBEDDINGS:
                    _core._entities._enqueue_zone_profile_refresh_event(
                        db,
                        session_id=session_id,
                        object_id=current_location.object_id,
                    )
            _core.logger.warning(
                "Recovering move_player location anchor for session_id=%s player_object_id=%s turn=%s active_links=%s",
                session_id,
                player_object_id,
                new_turn,
                len(active_locations),
            )
        elif not active_locations:
            _core.logger.warning(
                "Recovering move_player with no active located_in link for session_id=%s player_object_id=%s turn=%s",
                session_id,
                player_object_id,
                new_turn,
            )

        current_location_id = current_location.object_id

        target_zone = db.execute(
            select(models.ObjectModel)
            .join(
                models.LinkModel,
                and_(
                    models.LinkModel.session_id == models.ObjectModel.session_id,
                    models.LinkModel.to_object_id == models.ObjectModel.object_id,
                ),
            )
            .where(
                models.ObjectModel.session_id == session_id,
                models.ObjectModel.type == "zone",
                models.ObjectModel.name == payload.to_name,
                models.LinkModel.session_id == session_id,
                models.LinkModel.from_object_id == current_location_id,
                models.LinkModel.type == payload.relation,
                models.LinkModel.valid_to_turn.is_(None),
            )
            .limit(1)
        ).scalar_one_or_none()

        if target_zone is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Move target must be an existing reachable location",
            )

        _core._shared._close_player_active_located_in_links(
            db,
            session_id,
            player_object_id,
            closed_at_turn=new_turn,
        )

        db.add(
            models.LinkModel(
                session_id=session_id,
                from_object_id=player_object_id,
                to_object_id=target_zone.object_id,
                type=LOCATED_IN_LINK_TYPE,
                data={},
                valid_from_turn=new_turn,
                valid_to_turn=None,
            )
        )

        applied_ops: list[dict[str, object]] = list(ttl_cleanup.applied_ops)
        if current_location_created:
            applied_ops.append(
                {
                    "op": "object.create",
                    "ref": str(current_location_id),
                    "type": "zone",
                    "name": current_location.name,
                    "data": dict(getattr(current_location, "data", {}) or {}),
                }
            )
        if had_active_player_location:
            applied_ops.append(
                {
                    "op": "link.close",
                    "from": str(player_object_id),
                    "to": str(current_location_id),
                    "type": LOCATED_IN_LINK_TYPE,
                }
            )
        applied_ops.append(
            {
                "op": "link.create",
                "from": str(player_object_id),
                "to": str(target_zone.object_id),
                "type": LOCATED_IN_LINK_TYPE,
                "data": {},
            }
        )

        turn_snippet = f"(move) {current_location.name} -> {target_zone.name} via {payload.relation}"
        turn_row.ai_text = turn_snippet
        turn_row.ai_json = _build_internal_turn_ai_json(
            note="debug_move",
            applied_ops=applied_ops,
            in_game_day=next_day,
            in_game_minute=next_minute,
            extra_ai_json={"ttl_cleaned": ttl_cleanup.cleaned_count},
        )

        if current_location_created:
            _core._entities._add_internal_object_created_event(
                db,
                session_id=session_id,
                turn_index=new_turn,
                object_row=current_location,
                object_data=dict(getattr(current_location, "data", {}) or {}),
            )
        if had_active_player_location:
            _core._entities._add_internal_link_event(
                db,
                session_id=session_id,
                turn_index=new_turn,
                event_type="LINK_CLOSED",
                from_object_id=player_object_id,
                to_object_id=current_location_id,
                link_type=LOCATED_IN_LINK_TYPE,
                from_object=None,
                to_object=current_location,
            )
        _core._entities._add_internal_link_event(
            db,
            session_id=session_id,
            turn_index=new_turn,
            event_type="LINK_CREATED",
            from_object_id=player_object_id,
            to_object_id=target_zone.object_id,
            link_type=LOCATED_IN_LINK_TYPE,
            from_object=None,
            to_object=target_zone,
            link_data={},
            valid_from_turn=new_turn,
            valid_to_turn=None,
        )

        move_event = models.EventModel(
            session_id=session_id,
            turn_index=new_turn,
            type="MOVE",
            scope_object_id=target_zone.object_id,
            payload={
                "from_object_id": str(current_location_id),
                "from_name": current_location.name,
                "to_object_id": str(target_zone.object_id),
                "to_name": target_zone.name,
                "relation": payload.relation,
                "in_game_time": {"day": next_day, "minute": next_minute},
            },
        )
        db.add(move_event)

        db.flush()
        target_zone_id = target_zone.object_id

        _core._entities._enqueue_turn_chronicle_sync_event(
            db,
            session_id=session_id,
            turn_index=new_turn,
        )

        move_out = schemas.MoveOut(
            new_location_id=target_zone.object_id,
            event_id=move_event.event_id,
            turn_index=new_turn,
        )

    if move_out is None or target_zone_id is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Move result is incomplete")

    return move_out


__all__ = ["move_player", "_move_player_locked"]
