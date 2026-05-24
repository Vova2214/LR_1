from __future__ import annotations

import uuid
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from .. import models, schemas
from .. import crud as crud_runtime
from .. import crud_context as context_ops
from .. import crud_core as core_ops
from ..domain import player_commands as player_command_domain
from ..lore_ux import claims_from_world_constitution_data
from ..persistence.player_command_repository import player_command_repository
from ..persistence.session_read_repository import session_read_repository


class PlayerCommandService:
    def _session_state(
        self,
        session_row: models.SessionModel,
        *,
        policy: player_command_domain.PlayerCommandPolicy = player_command_domain.PLAYER_COMMAND_POLICY,
    ) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], int]:
        state_payload = dict(session_row.state_json or {})
        corrections = player_command_domain.coerce_player_corrections(
            state_payload.get(player_command_domain.PLAYER_CORRECTIONS_KEY),
            policy=policy,
        )
        committed_turn = session_read_repository.resolve_committed_turn_upper_bound(session_row)
        return state_payload, corrections, committed_turn

    def _maybe_capture_snapshot(self, db: Session, *, session_id: uuid.UUID) -> None:
        try:
            player_command_repository.upsert_snapshot_for_current_state(db, session_id=session_id)
        except Exception:  # noqa: BLE001
            core_ops.logger.warning(
                "player_command_snapshot_capture_failed",
                extra={"session_id": str(session_id)},
                exc_info=True,
            )

    def _command_result(
        self,
        *,
        session_id: uuid.UUID,
        command: player_command_domain.PlayerCommandName,
        status_label: str,
        effect: str,
        affected_turns: list[int],
        updated_rules: list[str],
        next_step: str,
        details: dict[str, Any] | None = None,
        restored_turn: int | None = None,
        state_mutated: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "command": command,
            "status": status_label,
            "effect": effect,
            "affected_turns": affected_turns,
            "updated_rules": updated_rules,
            "next_step": next_step,
            "details": dict(details or {}),
            "state_mutated": state_mutated,
        }
        if restored_turn is not None:
            payload["restored_turn"] = restored_turn
        return payload

    def _handle_undo(
        self,
        db: Session,
        session_id: uuid.UUID,
        *,
        policy: player_command_domain.PlayerCommandPolicy,
    ) -> dict[str, Any]:
        with db.begin():
            session_row = core_ops._require_session(db, session_id, for_update=True)
            _state_payload, corrections, committed_turn = self._session_state(session_row, policy=policy)
            if committed_turn <= 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="No committed turn is available to undo.",
                )
            target_turn = committed_turn - 1
            snapshot_row = player_command_repository.get_snapshot_at_or_before(
                db,
                session_id=session_id,
                target_turn=target_turn,
            )
            if snapshot_row is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="No snapshot is available for undo.",
                )
            restored_turn = player_command_repository.restore_snapshot_in_place(
                db,
                session_row=session_row,
                snapshot_row=snapshot_row,
                preserved_player_corrections=player_command_domain.corrections_payload(corrections, policy=policy),
            )

        return self._command_result(
            session_id=session_id,
            command="undo",
            status_label="applied",
            effect="state_restored",
            affected_turns=list(range(restored_turn + 1, committed_turn + 1)),
            updated_rules=[],
            next_step="Continue from the restored turn or use /why to inspect the new latest outcome.",
            details={
                "requested_turn": target_turn,
                "restored_turn": restored_turn,
                "snapshot_turn": int(snapshot_row.turn_index),
            },
            restored_turn=restored_turn,
            state_mutated=True,
        )

    def _handle_rollback(
        self,
        db: Session,
        session_id: uuid.UUID,
        parsed: player_command_domain.ParsedPlayerCommand,
        *,
        policy: player_command_domain.PlayerCommandPolicy,
    ) -> dict[str, Any]:
        requested_turn = int(parsed.target_turn or 0)
        with db.begin():
            session_row = core_ops._require_session(db, session_id, for_update=True)
            _state_payload, corrections, committed_turn = self._session_state(session_row, policy=policy)
            if requested_turn >= committed_turn:
                return self._command_result(
                    session_id=session_id,
                    command="rollback",
                    status_label="applied",
                    effect="state_unchanged",
                    affected_turns=[],
                    updated_rules=[],
                    next_step="The session is already at or before that committed turn.",
                    details={
                        "requested_turn": requested_turn,
                        "restored_turn": committed_turn,
                        "snapshot_turn": committed_turn,
                    },
                    restored_turn=committed_turn,
                    state_mutated=False,
                )

            snapshot_row = player_command_repository.get_snapshot_at_or_before(
                db,
                session_id=session_id,
                target_turn=requested_turn,
            )
            if snapshot_row is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="No rollback snapshot is available at or before the requested turn.",
                )
            restored_turn = player_command_repository.restore_snapshot_in_place(
                db,
                session_row=session_row,
                snapshot_row=snapshot_row,
                preserved_player_corrections=player_command_domain.corrections_payload(corrections, policy=policy),
            )

        return self._command_result(
            session_id=session_id,
            command="rollback",
            status_label="applied",
            effect="state_restored",
            affected_turns=list(range(restored_turn + 1, committed_turn + 1)),
            updated_rules=[],
            next_step="Continue from the restored checkpoint or use /retcon to replay the latest action differently.",
            details={
                "requested_turn": requested_turn,
                "restored_turn": restored_turn,
                "snapshot_turn": int(snapshot_row.turn_index),
                "exact_match": int(snapshot_row.turn_index) == requested_turn,
            },
            restored_turn=restored_turn,
            state_mutated=restored_turn != committed_turn,
        )

    def _handle_fixrule(
        self,
        db: Session,
        session_id: uuid.UUID,
        parsed: player_command_domain.ParsedPlayerCommand,
        *,
        policy: player_command_domain.PlayerCommandPolicy,
    ) -> dict[str, Any]:
        with db.begin():
            session_row = core_ops._require_session(db, session_id, for_update=True)
            state_payload, corrections, committed_turn = self._session_state(session_row, policy=policy)
            corrections, amendment = player_command_domain.apply_rule_amendment(
                corrections,
                text=str(parsed.argument_text or ""),
                policy=policy,
            )
            state_payload[player_command_domain.PLAYER_CORRECTIONS_KEY] = corrections
            session_row.state_json = state_payload
            db.flush()

        self._maybe_capture_snapshot(db, session_id=session_id)
        return self._command_result(
            session_id=session_id,
            command="fixrule",
            status_label="applied",
            effect="rule_amendment_added",
            affected_turns=[committed_turn] if committed_turn >= 0 else [],
            updated_rules=[str(amendment.get("amendment_id") or "")],
            next_step="Future turns will treat this correction as a session-local runtime rule amendment.",
            details={"rule_amendment": amendment},
            state_mutated=True,
        )

    def _handle_fair(
        self,
        db: Session,
        session_id: uuid.UUID,
        parsed: player_command_domain.ParsedPlayerCommand,
        *,
        policy: player_command_domain.PlayerCommandPolicy,
    ) -> dict[str, Any]:
        with db.begin():
            session_row = core_ops._require_session(db, session_id, for_update=True)
            state_payload, corrections, committed_turn = self._session_state(session_row, policy=policy)
            corrections, guardrail = player_command_domain.apply_fairness_guardrail(
                corrections,
                text=str(parsed.argument_text or ""),
                policy=policy,
            )
            state_payload[player_command_domain.PLAYER_CORRECTIONS_KEY] = corrections
            session_row.state_json = state_payload
            db.flush()

        self._maybe_capture_snapshot(db, session_id=session_id)
        return self._command_result(
            session_id=session_id,
            command="fair",
            status_label="applied",
            effect="guardrail_added",
            affected_turns=[committed_turn] if committed_turn >= 0 else [],
            updated_rules=list(guardrail.get("rule_codes") or []),
            next_step="You can keep the outcome or use /retcon to replay the latest action with these guardrails active.",
            details={"fairness_guardrail": guardrail},
            state_mutated=True,
        )

    def _handle_retcon(
        self,
        db: Session,
        session_id: uuid.UUID,
        parsed: player_command_domain.ParsedPlayerCommand,
        *,
        policy: player_command_domain.PlayerCommandPolicy,
    ) -> dict[str, Any]:
        latest_turn_index = 0
        latest_user_input = ""
        restored_turn = 0

        with db.begin():
            session_row = core_ops._require_session(db, session_id, for_update=True)
            _state_payload, corrections, committed_turn = self._session_state(session_row, policy=policy)
            if committed_turn <= 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="No committed turn is available to retcon.",
                )

            latest_turn_index = committed_turn
            latest_turn = db.get(models.TurnModel, (session_id, latest_turn_index))
            if latest_turn is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="The latest committed turn could not be loaded for retcon.",
                )
            latest_user_input = str(latest_turn.user_input or "")
            snapshot_row = player_command_repository.get_snapshot_at_or_before(
                db,
                session_id=session_id,
                target_turn=latest_turn_index - 1,
            )
            if snapshot_row is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="No prior snapshot is available to replay the last turn.",
                )

            corrections = player_command_domain.mark_disputed_turn(
                corrections,
                turn_index=latest_turn_index,
                original_user_input=latest_user_input,
                replacement_user_input=str(parsed.argument_text or ""),
                replacement_turn_index=latest_turn_index,
                policy=policy,
            )
            restored_turn = player_command_repository.restore_snapshot_in_place(
                db,
                session_row=session_row,
                snapshot_row=snapshot_row,
                preserved_player_corrections=player_command_domain.corrections_payload(corrections, policy=policy),
            )

        replay_row = turn_application_service.run_turn_locked(
            db,
            session_id,
            schemas.TurnIn(user_input=str(parsed.argument_text or "")),
            allow_debug_patch=False,
        )
        self._maybe_capture_snapshot(db, session_id=session_id)
        refreshed_session = core_ops._require_session(db, session_id)
        refreshed_corrections = player_command_domain.coerce_player_corrections(
            dict(refreshed_session.state_json or {}).get(player_command_domain.PLAYER_CORRECTIONS_KEY),
            policy=policy,
        )
        return self._command_result(
            session_id=session_id,
            command="retcon",
            status_label="applied",
            effect="last_turn_replayed",
            affected_turns=[latest_turn_index],
            updated_rules=player_command_domain.command_rule_codes(refreshed_corrections, policy=policy),
            next_step="Review the replayed outcome or use /why to inspect what shaped it.",
            details={
                "replayed_turn": int(getattr(replay_row, "turn_index", 0) or 0),
                "replaced_turn": latest_turn_index,
                "restored_turn_before_replay": restored_turn,
                "original_user_input": latest_user_input,
                "replacement_user_input": str(parsed.argument_text or ""),
            },
            restored_turn=restored_turn,
            state_mutated=True,
        )

    def _handle_why(
        self,
        db: Session,
        session_id: uuid.UUID,
        *,
        policy: player_command_domain.PlayerCommandPolicy,
    ) -> dict[str, Any]:
        session_row = core_ops._require_session(db, session_id)
        _state_payload, corrections, committed_turn = self._session_state(session_row, policy=policy)
        if committed_turn <= 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No committed turn is available to explain.",
            )

        turn_row = db.get(models.TurnModel, (session_id, committed_turn))
        if turn_row is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The latest committed turn could not be loaded for explanation.",
            )

        _has_constitution, world_constitution_data = context_ops._get_latest_world_constitution_data(db, session_id)
        lore_rule_labels: list[str] = []
        for claim in claims_from_world_constitution_data(world_constitution_data):
            usage = str(getattr(claim, "usage", "") or "").strip()
            label = str(getattr(claim, "label", "") or "").strip()
            if usage not in {"runtime", "campaign", "special_case", "bookkeeping"} or not label:
                continue
            lore_rule_labels.append(label)
            if len(lore_rule_labels) >= policy.why_max_rule_labels:
                break

        explanation = player_command_domain.build_turn_explanation(
            turn_index=committed_turn,
            user_input=str(turn_row.user_input or ""),
            ai_json=dict(turn_row.ai_json or {}) if isinstance(turn_row.ai_json, dict) else {},
            lore_rule_labels=lore_rule_labels,
            corrections=corrections,
            policy=policy,
        )
        return self._command_result(
            session_id=session_id,
            command="why",
            status_label="explained",
            effect="explanation_returned",
            affected_turns=[committed_turn],
            updated_rules=[],
            next_step="Use /retcon to replay the turn or /fair to add a session-local fairness guardrail.",
            details={"explanation": explanation},
            state_mutated=False,
        )

    def run_command(
        self,
        db: Session,
        session_id: uuid.UUID,
        payload: schemas.PlayerCommandIn,
    ) -> dict[str, Any]:
        policy = player_command_domain.PLAYER_COMMAND_POLICY
        try:
            parsed = player_command_domain.parse_player_command_text(payload.command_text, policy=policy)
        except player_command_domain.PlayerCommandParseError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

        with crud_runtime._session_turn_runtime_lock(db, session_id):
            if parsed.command == "undo":
                return self._handle_undo(db, session_id, policy=policy)
            if parsed.command == "rollback":
                return self._handle_rollback(db, session_id, parsed, policy=policy)
            if parsed.command == "fixrule":
                return self._handle_fixrule(db, session_id, parsed, policy=policy)
            if parsed.command == "fair":
                return self._handle_fair(db, session_id, parsed, policy=policy)
            if parsed.command == "retcon":
                return self._handle_retcon(db, session_id, parsed, policy=policy)
            if parsed.command == "why":
                return self._handle_why(db, session_id, policy=policy)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unsupported player command '/{parsed.command}'.",
            )


player_command_service = PlayerCommandService()


from .turn_services import turn_application_service  # noqa: E402


__all__ = ["player_command_service"]
