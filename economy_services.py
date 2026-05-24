from __future__ import annotations

import uuid
from typing import Any, Mapping

from sqlalchemy.orm import Session

from ..domain.economy_policy import (
    DEFAULT_ECONOMY_POLICY,
    apply_economic_operations_payload,
    derive_economic_delta_payload,
    derive_economy_brief_payload,
    derive_economy_state_payload,
)
from ..persistence.economy_repository import economy_projection_repository

_OBJECT_ID_KEYS = (
    "object_id",
    "fact_object_id",
    "asset_object_id",
    "owner_object_id",
    "holder_object_id",
    "possessor_object_id",
    "actor_object_id",
    "counterparty_object_id",
    "location_object_id",
    "quest_object_id",
    "owner_entity_id",
    "possessor_entity_id",
    "debtor_entity_id",
    "creditor_entity_id",
    "business_entity_id",
    "employee_entity_id",
    "collector_entity_id",
    "last_owner_id",
)
_ECONOMY_HISTORY_LIMIT = max(int(DEFAULT_ECONOMY_POLICY.review_delta_history_limit), 1)
_MONETARY_PATCH_KEYS = ("cash_reserve", "currency_balance", "cash_on_hand")
_INVENTORY_PATCH_KEYS = ("inventory_balance", "stock_balance", "warehouse_units")
_STRUCTURED_OPERATION_KEYS = ("economic_operation", "operation", "economic_effect")
_SUPPORTED_OPERATION_TYPES = {
    "buy",
    "sell",
    "transfer",
    "loan",
    "repay",
    "hire",
    "fire",
    "pay_wage",
    "charge_rent",
    "restock",
    "craft",
    "collect_tax",
    "distribute_profit",
}


def _coerce_uuid_text(value: Any) -> str | None:
    if isinstance(value, uuid.UUID):
        return str(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return str(uuid.UUID(text))
    except (TypeError, ValueError, AttributeError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_mapping_list(raw_rows: Any) -> list[dict[str, Any]]:
    return [dict(row or {}) for row in list(raw_rows or []) if isinstance(row, Mapping)]


def _compiled_world_model(world_constitution_data: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(world_constitution_data or {})
    return dict(payload.get("compiled_world_model") or {})


def _identifier_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _coerce_id_list(values: Any) -> list[str]:
    normalized: list[str] = []
    for raw_value in list(values or []):
        text = _identifier_text(raw_value)
        if text and text not in normalized:
            normalized.append(text)
    return normalized


class EconomyDerivationService:
    def __init__(self) -> None:
        self._repository = economy_projection_repository

    def _collect_object_ids(self, *row_groups: Any, explicit_ids: list[Any] | None = None) -> list[str]:
        object_ids: list[str] = []
        for raw_value in list(explicit_ids or []):
            object_id = _coerce_uuid_text(raw_value)
            if object_id and object_id not in object_ids:
                object_ids.append(object_id)
        for group in row_groups:
            for row in _coerce_mapping_list(group):
                for key in _OBJECT_ID_KEYS:
                    object_id = _coerce_uuid_text(row.get(key))
                    if object_id and object_id not in object_ids:
                        object_ids.append(object_id)
        return object_ids

    def _object_profiles(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
        object_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        try:
            return self._repository.list_object_profiles(
                db,
                session_id=session_id,
                object_ids=object_ids,
            )
        except StopIteration:
            return {}

    def _recent_delta_history(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        try:
            rows = self._repository.list_recent_turn_economy_rows(
                db,
                session_id=session_id,
                limit=_ECONOMY_HISTORY_LIMIT,
            )
        except (StopIteration, TypeError, AttributeError):
            return []
        return self._delta_history_from_turn_rows(rows)

    def _delta_history_from_turn_rows(
        self,
        turn_rows: list[Mapping[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        delta_history: list[dict[str, Any]] = []
        seen_delta_keys: set[str] = set()
        ordered_rows = sorted(
            _coerce_mapping_list(turn_rows),
            key=lambda item: int(item.get("turn_index") or 0),
        )
        for row in ordered_rows:
            economy_payload = dict(row.get("economy") or {})
            raw_delta = dict(economy_payload.get("delta") or {})
            if not raw_delta:
                after_state = dict(economy_payload.get("after") or {})
                history_candidates = [
                    dict(item)
                    for item in list(after_state.get("delta_history") or [])
                    if isinstance(item, Mapping)
                ]
                if history_candidates:
                    raw_delta = history_candidates[-1]
            delta_key = str(raw_delta.get("delta_key") or "").strip()
            if not raw_delta or not delta_key or delta_key in seen_delta_keys:
                continue
            seen_delta_keys.add(delta_key)
            delta_history.append(raw_delta)
        return delta_history[-_ECONOMY_HISTORY_LIMIT:]

    def build_context_economy_state(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
        current_turn: int,
        turn_intent: str | None,
        world_constitution_data: Mapping[str, Any] | None,
        fact_rows: list[Mapping[str, Any]] | None,
        obligation_rows: list[Mapping[str, Any]] | None,
        inventory_rows: list[Mapping[str, Any]] | None,
        orphaned_items: list[Mapping[str, Any]] | None,
        supplemental_rows: list[Mapping[str, Any]] | None = None,
        explicit_object_ids: list[Any] | None = None,
    ) -> dict[str, Any]:
        compiled_world_model = _compiled_world_model(world_constitution_data)
        object_ids = self._collect_object_ids(
            fact_rows,
            obligation_rows,
            inventory_rows,
            orphaned_items,
            supplemental_rows,
            explicit_ids=explicit_object_ids,
        )
        object_profiles = self._object_profiles(
            db,
            session_id=session_id,
            object_ids=object_ids,
        )
        delta_history = self._recent_delta_history(
            db,
            session_id=session_id,
        )
        return derive_economy_state_payload(
            scope="turn_context",
            current_turn=max(int(current_turn), 0),
            compiled_world_model=compiled_world_model,
            fact_rows=_coerce_mapping_list(fact_rows),
            obligation_rows=_coerce_mapping_list(obligation_rows),
            object_profiles=object_profiles,
            inventory_rows=_coerce_mapping_list(inventory_rows),
            orphaned_items=_coerce_mapping_list(orphaned_items),
            delta_history=delta_history,
            turn_intent=turn_intent,
        )

    def build_review_economy_state(
        self,
        db: Session,
        *,
        session_id: uuid.UUID,
        current_turn: int,
        fact_rows: list[Mapping[str, Any]] | None,
        obligation_rows: list[Mapping[str, Any]] | None,
        turn_rows: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            world_constitution_data = self._repository.get_world_constitution_data(
                db,
                session_id=session_id,
            )
        except StopIteration:
            world_constitution_data = {}
        object_ids = self._collect_object_ids(
            fact_rows,
            obligation_rows,
        )
        object_profiles = self._object_profiles(
            db,
            session_id=session_id,
            object_ids=object_ids,
        )
        delta_history = self._delta_history_from_turn_rows(turn_rows)
        return derive_economy_state_payload(
            scope="review",
            current_turn=max(int(current_turn), 0),
            compiled_world_model=_compiled_world_model(world_constitution_data),
            fact_rows=_coerce_mapping_list(fact_rows),
            obligation_rows=_coerce_mapping_list(obligation_rows),
            object_profiles=object_profiles,
            delta_history=delta_history,
        )

    def _resolve_turn_ref(
        self,
        raw_ref: Any,
        *,
        ref_map: Mapping[str, str],
        player_object_id: uuid.UUID | None,
    ) -> str | None:
        if isinstance(raw_ref, Mapping):
            raw_ref = raw_ref.get("ref") or raw_ref.get("id") or raw_ref.get("object_id")
        text = str(raw_ref or "").strip()
        if not text:
            return None
        if text == "session_player" and player_object_id is not None:
            return str(player_object_id)
        if text in ref_map:
            return str(ref_map[text]).strip() or None
        return _coerce_uuid_text(text)

    def _asset_for_object_id(self, state_payload: Mapping[str, Any], object_id: str | None) -> dict[str, Any] | None:
        if not object_id:
            return None
        for asset in list(dict(state_payload or {}).get("assets") or []):
            if not isinstance(asset, dict):
                continue
            if str(asset.get("asset_object_id") or "").strip() == object_id:
                return dict(asset)
        return None

    def _business_for_entity(self, state_payload: Mapping[str, Any], entity_id: str | None) -> dict[str, Any] | None:
        if not entity_id:
            return None
        for business in list(dict(state_payload or {}).get("business_units") or []):
            if not isinstance(business, dict):
                continue
            if str(business.get("entity_id") or "").strip() == entity_id:
                return dict(business)
        return None

    def _money_balance_for_entity(self, state_payload: Mapping[str, Any], entity_id: str | None) -> float:
        if not entity_id:
            return 0.0
        for asset in list(dict(state_payload or {}).get("assets") or []):
            if not isinstance(asset, dict):
                continue
            if str(asset.get("asset_class") or "").strip() != "money":
                continue
            if str(asset.get("owner_entity_id") or asset.get("business_entity_id") or "").strip() != entity_id:
                continue
            return float(_safe_float(asset.get("quantity") or asset.get("liquid_value")) or 0.0)
        business = self._business_for_entity(state_payload, entity_id)
        return float(_safe_float((business or {}).get("cash_reserve")) or 0.0)

    def _inventory_balance_for_entity(self, state_payload: Mapping[str, Any], entity_id: str | None) -> float:
        if not entity_id:
            return 0.0
        balance = 0.0
        for asset in list(dict(state_payload or {}).get("assets") or []):
            if not isinstance(asset, dict):
                continue
            if str(asset.get("asset_class") or "").strip() not in {"goods", "rare_item", "warehouse_stock"}:
                continue
            if entity_id not in {
                str(asset.get("owner_entity_id") or "").strip(),
                str(asset.get("possessor_entity_id") or "").strip(),
                str(asset.get("business_entity_id") or "").strip(),
            }:
                continue
            balance += float(_safe_float(asset.get("quantity") or asset.get("liquid_value")) or 0.0)
        return round(balance, 6)

    def _matching_obligations(
        self,
        state_payload: Mapping[str, Any],
        *,
        obligation_type: str,
        debtor_entity_id: str | None = None,
        creditor_entity_id: str | None = None,
    ) -> list[dict[str, Any]]:
        matches = []
        for obligation in list(dict(state_payload or {}).get("obligations") or []):
            if not isinstance(obligation, dict):
                continue
            if str(obligation.get("status") or "").strip() != "open":
                continue
            if str(obligation.get("obligation_type") or "").strip() != obligation_type:
                continue
            if debtor_entity_id and str(obligation.get("debtor_entity_id") or "").strip() != debtor_entity_id:
                continue
            if creditor_entity_id and str(obligation.get("creditor_entity_id") or "").strip() != creditor_entity_id:
                continue
            matches.append(dict(obligation))
        matches.sort(
            key=lambda item: (
                int(item.get("due_turn") or 0),
                -float(_safe_float(item.get("accrued_amount") or item.get("amount") or item.get("quantity")) or 0.0),
                str(item.get("obligation_key") or ""),
            )
        )
        return matches

    def _resolve_patch_numeric(self, patch_payload: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
        for key in keys:
            value = _safe_float(patch_payload.get(key))
            if value is not None:
                return value
        return None

    def _normalize_operation_payload(
        self,
        raw_payload: Mapping[str, Any],
        *,
        operation_key: str,
        source_ref: str,
        state_payload: Mapping[str, Any],
        ref_map: Mapping[str, str],
        player_object_id: uuid.UUID | None,
        fallback_actor: str | None = None,
        fallback_counterparty: str | None = None,
        fallback_business: str | None = None,
    ) -> dict[str, Any] | None:
        operation_type = str(raw_payload.get("operation_type") or raw_payload.get("type") or "").strip()
        if operation_type not in _SUPPORTED_OPERATION_TYPES:
            return None
        actor = self._resolve_turn_ref(
            raw_payload.get("actor_entity_id")
            or raw_payload.get("actor")
            or raw_payload.get("source")
            or fallback_actor,
            ref_map=ref_map,
            player_object_id=player_object_id,
        ) or _identifier_text(raw_payload.get("actor_entity_id") or fallback_actor)
        counterparty = self._resolve_turn_ref(
            raw_payload.get("counterparty_entity_id")
            or raw_payload.get("counterparty")
            or raw_payload.get("target")
            or fallback_counterparty,
            ref_map=ref_map,
            player_object_id=player_object_id,
        ) or _identifier_text(raw_payload.get("counterparty_entity_id") or fallback_counterparty)
        business = self._resolve_turn_ref(
            raw_payload.get("business_entity_id")
            or raw_payload.get("business")
            or fallback_business,
            ref_map=ref_map,
            player_object_id=player_object_id,
        ) or _identifier_text(raw_payload.get("business_entity_id") or fallback_business)
        asset_object_id = self._resolve_turn_ref(
            raw_payload.get("asset_object_id")
            or raw_payload.get("asset")
            or raw_payload.get("asset_ref")
            or raw_payload.get("object")
            or raw_payload.get("object_ref"),
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        asset_payload = self._asset_for_object_id(state_payload, asset_object_id)
        asset_key = _identifier_text(raw_payload.get("asset_key")) or _identifier_text((asset_payload or {}).get("asset_key"))
        if asset_key is None and asset_object_id is not None:
            asset_key = f"{str((asset_payload or {}).get('asset_class') or 'goods').strip()}:{asset_object_id}"
        asset_class = _identifier_text(raw_payload.get("asset_class")) or _identifier_text((asset_payload or {}).get("asset_class"))
        metadata = dict(raw_payload.get("metadata") or {})
        metadata.update(
            {
                key: value
                for key, value in dict(raw_payload).items()
                if key
                not in {
                    "operation_type",
                    "type",
                    "actor_entity_id",
                    "actor",
                    "source",
                    "counterparty_entity_id",
                    "counterparty",
                    "target",
                    "business_entity_id",
                    "business",
                    "asset_object_id",
                    "asset",
                    "asset_ref",
                    "object",
                    "object_ref",
                    "asset_key",
                    "asset_class",
                    "obligation_type",
                    "amount",
                    "value",
                    "payment_amount",
                    "quantity",
                    "units",
                    "metadata",
                }
            }
        )
        return {
            "operation_key": operation_key,
            "operation_type": operation_type,
            "actor_entity_id": actor,
            "counterparty_entity_id": counterparty,
            "business_entity_id": business,
            "asset_key": asset_key,
            "asset_class": asset_class,
            "obligation_type": _identifier_text(raw_payload.get("obligation_type")),
            "amount": _safe_float(raw_payload.get("amount") or raw_payload.get("value") or raw_payload.get("payment_amount")),
            "quantity": _safe_float(raw_payload.get("quantity") or raw_payload.get("units")),
            "source_refs": [source_ref],
            "metadata": metadata,
        }

    def derive_turn_operations(
        self,
        *,
        base_state: Mapping[str, Any] | None,
        applied_ops: list[Mapping[str, Any]] | None,
        semantic_events: list[Any] | None,
        ref_map: Mapping[str, str] | None,
        player_object_id: uuid.UUID | None,
    ) -> list[dict[str, Any]]:
        state_payload = dict(base_state or {})
        normalized_ref_map = {str(key): str(value) for key, value in dict(ref_map or {}).items()}
        operations: list[dict[str, Any]] = []
        seen_operation_keys: set[str] = set()
        seen_signatures: set[tuple[Any, ...]] = set()
        pending_previous_holders: dict[str, str] = {}

        def _append_operation(payload: Mapping[str, Any] | None) -> None:
            if not isinstance(payload, Mapping):
                return
            operation_key = str(payload.get("operation_key") or "").strip()
            operation_type = str(payload.get("operation_type") or "").strip()
            if not operation_key or operation_key in seen_operation_keys or operation_type not in _SUPPORTED_OPERATION_TYPES:
                return
            signature = (
                operation_type,
                str(payload.get("actor_entity_id") or ""),
                str(payload.get("counterparty_entity_id") or ""),
                str(payload.get("business_entity_id") or ""),
                str(payload.get("asset_key") or ""),
                round(float(_safe_float(payload.get("amount")) or 0.0), 6),
                round(float(_safe_float(payload.get("quantity")) or 0.0), 6),
            )
            if signature in seen_signatures:
                return
            seen_operation_keys.add(operation_key)
            seen_signatures.add(signature)
            operations.append(dict(payload))

        for index, raw_op in enumerate(_coerce_mapping_list(applied_ops)):
            op_name = str(raw_op.get("op") or "").strip()
            link_type = str(raw_op.get("type") or "").strip()
            if op_name == "link.close" and link_type == "carried_by":
                asset_object_id = _coerce_uuid_text(raw_op.get("from"))
                previous_holder = _coerce_uuid_text(raw_op.get("to"))
                if asset_object_id and previous_holder:
                    pending_previous_holders[asset_object_id] = previous_holder
                continue
            if op_name != "link.create" or link_type != "carried_by":
                continue
            asset_object_id = _coerce_uuid_text(raw_op.get("from"))
            new_holder = _coerce_uuid_text(raw_op.get("to"))
            if asset_object_id is None or new_holder is None:
                continue
            asset_payload = self._asset_for_object_id(state_payload, asset_object_id)
            _append_operation(
                {
                    "operation_key": f"applied:{index}:carried_by:{asset_object_id}",
                    "operation_type": "transfer",
                    "actor_entity_id": pending_previous_holders.pop(asset_object_id, None)
                    or str((asset_payload or {}).get("owner_entity_id") or "").strip()
                    or None,
                    "counterparty_entity_id": new_holder,
                    "asset_key": str((asset_payload or {}).get("asset_key") or f"goods:{asset_object_id}").strip(),
                    "asset_class": str((asset_payload or {}).get("asset_class") or "goods").strip(),
                    "quantity": (asset_payload or {}).get("quantity") if asset_payload else 1.0,
                    "source_refs": [f"applied_op:{index}"],
                    "metadata": {"link_type": "carried_by", "asset_object_id": asset_object_id},
                }
            )

        for index, raw_op in enumerate(_coerce_mapping_list(applied_ops)):
            op_name = str(raw_op.get("op") or "").strip()
            if op_name == "event.create":
                payload = dict(raw_op.get("payload") or {})
                structured_payload: dict[str, Any] | None = None
                for key in _STRUCTURED_OPERATION_KEYS:
                    candidate = payload.get(key)
                    if isinstance(candidate, Mapping):
                        structured_payload = dict(candidate)
                        break
                if structured_payload is None and str(payload.get("operation_type") or "").strip() in _SUPPORTED_OPERATION_TYPES:
                    structured_payload = dict(payload)
                if structured_payload is not None:
                    _append_operation(
                        self._normalize_operation_payload(
                            structured_payload,
                            operation_key=f"applied:{index}:event_create",
                            source_ref=f"applied_op:{index}",
                            state_payload=state_payload,
                            ref_map=normalized_ref_map,
                            player_object_id=player_object_id,
                            fallback_business=self._resolve_turn_ref(
                                raw_op.get("scope"),
                                ref_map=normalized_ref_map,
                                player_object_id=player_object_id,
                            ),
                        )
                    )
                continue
            if op_name != "object.update":
                continue
            object_id = self._resolve_turn_ref(
                raw_op.get("object"),
                ref_map=normalized_ref_map,
                player_object_id=player_object_id,
            ) or _identifier_text(raw_op.get("object"))
            patch_payload = dict(raw_op.get("patch") or {})
            if not object_id or not patch_payload:
                continue
            existing_business = self._business_for_entity(state_payload, object_id)
            existing_employee_ids = set(_coerce_id_list((existing_business or {}).get("employee_ids")))
            patched_employee_ids = [
                resolved
                for resolved in (
                    self._resolve_turn_ref(item, ref_map=normalized_ref_map, player_object_id=player_object_id)
                    or _identifier_text(item)
                    for item in list(patch_payload.get("employee_ids") or [])
                )
                if resolved
            ]
            if "employee_ids" in patch_payload:
                for employee_id in sorted(set(patched_employee_ids) - existing_employee_ids):
                    wage_matches = self._matching_obligations(
                        state_payload,
                        obligation_type="wage",
                        debtor_entity_id=object_id,
                        creditor_entity_id=employee_id,
                    )
                    _append_operation(
                        {
                            "operation_key": f"applied:{index}:hire:{employee_id}",
                            "operation_type": "hire",
                            "actor_entity_id": object_id,
                            "counterparty_entity_id": employee_id,
                            "business_entity_id": object_id,
                            "amount": _safe_float((wage_matches[0] if wage_matches else {}).get("amount")),
                            "source_refs": [f"applied_op:{index}"],
                            "metadata": {"patch_key": "employee_ids"},
                        }
                    )
                for employee_id in sorted(existing_employee_ids - set(patched_employee_ids)):
                    _append_operation(
                        {
                            "operation_key": f"applied:{index}:fire:{employee_id}",
                            "operation_type": "fire",
                            "actor_entity_id": object_id,
                            "counterparty_entity_id": employee_id,
                            "business_entity_id": object_id,
                            "source_refs": [f"applied_op:{index}"],
                            "metadata": {"patch_key": "employee_ids"},
                        }
                    )

            cash_new = self._resolve_patch_numeric(patch_payload, _MONETARY_PATCH_KEYS)
            cash_current = self._money_balance_for_entity(state_payload, object_id)
            cash_delta = (
                round(float(cash_new) - float(cash_current), 6)
                if cash_new is not None
                else 0.0
            )
            inventory_new = self._resolve_patch_numeric(patch_payload, _INVENTORY_PATCH_KEYS)
            inventory_current = self._inventory_balance_for_entity(state_payload, object_id)
            inventory_delta = (
                round(float(inventory_new) - float(inventory_current), 6)
                if inventory_new is not None
                else 0.0
            )
            if cash_new is not None and inventory_new is not None:
                if inventory_delta > 0.0 and cash_delta < 0.0:
                    _append_operation(
                        {
                            "operation_key": f"applied:{index}:restock",
                            "operation_type": "restock",
                            "actor_entity_id": object_id,
                            "business_entity_id": object_id,
                            "asset_key": f"warehouse_stock:{object_id}",
                            "asset_class": "warehouse_stock",
                            "amount": abs(cash_delta),
                            "quantity": inventory_delta,
                            "source_refs": [f"applied_op:{index}"],
                            "metadata": {"patch_keys": list(patch_payload)},
                        }
                    )
                elif inventory_delta < 0.0 and cash_delta > 0.0:
                    _append_operation(
                        {
                            "operation_key": f"applied:{index}:sell",
                            "operation_type": "sell",
                            "actor_entity_id": object_id,
                            "asset_key": f"warehouse_stock:{object_id}",
                            "asset_class": "warehouse_stock",
                            "amount": cash_delta,
                            "quantity": abs(inventory_delta),
                            "source_refs": [f"applied_op:{index}"],
                            "metadata": {"patch_keys": list(patch_payload)},
                        }
                    )
            if cash_new is None:
                continue
            if cash_delta < 0.0:
                outgoing_amount = abs(cash_delta)
                for obligation_type, operation_type in (
                    ("debt", "repay"),
                    ("wage", "pay_wage"),
                    ("profit_share", "distribute_profit"),
                ):
                    matches = self._matching_obligations(
                        state_payload,
                        obligation_type=obligation_type,
                        debtor_entity_id=object_id,
                    )
                    for match in matches:
                        obligation_amount = float(
                            _safe_float(match.get("accrued_amount") or match.get("amount") or match.get("quantity")) or 0.0
                        )
                        if obligation_amount <= 0.0 or outgoing_amount <= 0.0:
                            continue
                        if obligation_amount > outgoing_amount:
                            continue
                        outgoing_amount = round(outgoing_amount - obligation_amount, 6)
                        _append_operation(
                            {
                                "operation_key": f"applied:{index}:{operation_type}:{match.get('obligation_key')}",
                                "operation_type": operation_type,
                                "actor_entity_id": object_id,
                                "counterparty_entity_id": _identifier_text(match.get("creditor_entity_id")),
                                "business_entity_id": object_id if obligation_type in {"wage", "profit_share"} else None,
                                "asset_key": _identifier_text(match.get("related_asset_key")),
                                "amount": obligation_amount,
                                "source_refs": [f"applied_op:{index}"],
                                "metadata": {
                                    "target_obligation_key": _identifier_text(match.get("obligation_key")),
                                    "patch_keys": list(patch_payload),
                                },
                            }
                        )
            elif cash_delta > 0.0:
                incoming_amount = cash_delta
                matches = self._matching_obligations(
                    state_payload,
                    obligation_type="tax",
                    creditor_entity_id=object_id,
                )
                for match in matches:
                    obligation_amount = float(
                        _safe_float(match.get("accrued_amount") or match.get("amount") or match.get("quantity")) or 0.0
                    )
                    if obligation_amount <= 0.0 or incoming_amount <= 0.0:
                        continue
                    if obligation_amount > incoming_amount:
                        continue
                    incoming_amount = round(incoming_amount - obligation_amount, 6)
                    _append_operation(
                        {
                            "operation_key": f"applied:{index}:collect_tax:{match.get('obligation_key')}",
                            "operation_type": "collect_tax",
                            "actor_entity_id": object_id,
                            "counterparty_entity_id": _identifier_text(match.get("debtor_entity_id")),
                            "amount": obligation_amount,
                            "source_refs": [f"applied_op:{index}"],
                            "metadata": {
                                "target_obligation_key": _identifier_text(match.get("obligation_key")),
                                "patch_keys": list(patch_payload),
                            },
                        }
                    )

        for index, raw_event in enumerate(list(semantic_events or [])):
            if hasattr(raw_event, "model_dump"):
                event = raw_event.model_dump(mode="json")
            elif isinstance(raw_event, Mapping):
                event = dict(raw_event)
            else:
                continue
            event_type = str(event.get("type") or "").strip()
            payload = dict(event.get("payload") or {})
            actor = self._resolve_turn_ref(
                event.get("source") or event.get("subject"),
                ref_map=normalized_ref_map,
                player_object_id=player_object_id,
            )
            counterparty = self._resolve_turn_ref(
                event.get("target") or event.get("object"),
                ref_map=normalized_ref_map,
                player_object_id=player_object_id,
            )
            asset_object_id = self._resolve_turn_ref(
                payload.get("object")
                or payload.get("object_ref")
                or payload.get("item")
                or event.get("object"),
                ref_map=normalized_ref_map,
                player_object_id=player_object_id,
            )
            asset_payload = self._asset_for_object_id(state_payload, asset_object_id)
            amount = _safe_float(
                payload.get("amount")
                or payload.get("value")
                or payload.get("payment_amount")
            )
            quantity = _safe_float(payload.get("quantity") or payload.get("units"))
            operation_type: str | None = None
            obligation_type: str | None = None
            if event_type == "debt":
                operation_type = "loan"
                obligation_type = "debt"
            elif event_type in {"gift", "item_transfer"}:
                operation_type = "transfer"
            elif event_type == "promise":
                operation_type = "transfer" if asset_object_id else None
            else:
                structured_payload = None
                for key in _STRUCTURED_OPERATION_KEYS:
                    candidate = payload.get(key)
                    if isinstance(candidate, Mapping):
                        structured_payload = dict(candidate)
                        break
                if structured_payload is None and str(payload.get("operation_type") or "").strip() in _SUPPORTED_OPERATION_TYPES:
                    structured_payload = dict(payload)
                if structured_payload is not None:
                    _append_operation(
                        self._normalize_operation_payload(
                            structured_payload,
                            operation_key=f"semantic:{index}:structured",
                            source_ref=f"semantic_event:{index}:{event_type}",
                            state_payload=state_payload,
                            ref_map=normalized_ref_map,
                            player_object_id=player_object_id,
                            fallback_actor=actor,
                            fallback_counterparty=counterparty,
                        )
                    )
                    continue
            if operation_type is None:
                continue
            operation_payload = {
                "operation_key": f"semantic:{index}:{event_type}",
                "operation_type": operation_type,
                "actor_entity_id": actor,
                "counterparty_entity_id": counterparty,
                "asset_key": str((asset_payload or {}).get("asset_key") or (f"goods:{asset_object_id}" if asset_object_id else "")).strip() or None,
                "asset_class": str((asset_payload or {}).get("asset_class") or "goods").strip() if asset_object_id else None,
                "obligation_type": obligation_type,
                "amount": amount,
                "quantity": quantity,
                "source_refs": [f"semantic_event:{index}:{event_type}"],
                "metadata": {"semantic_event_type": event_type},
            }
            if operation_payload["asset_key"] is None and operation_type == "transfer":
                continue
            _append_operation(operation_payload)

        return operations

    def build_turn_observability_economy(
        self,
        *,
        base_state: Mapping[str, Any] | None,
        current_turn: int,
        applied_ops: list[Mapping[str, Any]] | None,
        semantic_events: list[Any] | None,
        ref_map: Mapping[str, str] | None,
        player_object_id: uuid.UUID | None,
    ) -> dict[str, Any]:
        before_state = dict(base_state or {})
        if not before_state:
            before_state = derive_economy_state_payload(
                scope="turn_observability",
                current_turn=max(int(current_turn) - 1, 0),
                compiled_world_model={},
            )
        operations = self.derive_turn_operations(
            base_state=before_state,
            applied_ops=applied_ops,
            semantic_events=semantic_events,
            ref_map=ref_map,
            player_object_id=player_object_id,
        )
        after_state = apply_economic_operations_payload(
            before_state,
            operations,
            current_turn=max(int(current_turn), 0),
        )
        delta_payload = derive_economic_delta_payload(
            before_state,
            after_state,
            operations,
            current_turn=max(int(current_turn), 0),
        )
        after_state = dict(after_state)
        history = [
            dict(item)
            for item in list((dict(before_state or {})).get("delta_history") or [])
            if isinstance(item, Mapping)
        ]
        history.append(dict(delta_payload))
        after_state["delta_history"] = history[-_ECONOMY_HISTORY_LIMIT:]
        after_state["continuity_summary"] = dict(after_state.get("continuity_summary") or {})
        after_state["continuity_summary"]["delta_count"] = len(after_state["delta_history"])
        after_state["brief"] = derive_economy_brief_payload(
            after_state,
            policy=DEFAULT_ECONOMY_POLICY,
        )
        return {
            "before": before_state,
            "operations": operations,
            "after": after_state,
            "delta": delta_payload,
        }


economy_service = EconomyDerivationService()
