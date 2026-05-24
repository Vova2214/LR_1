from __future__ import annotations

from typing import TypedDict


class NpcData(TypedDict, total=False):
    short_desc: str
    role: str
    attitude: str
    ephemeral: bool
    pinned: bool
    despawned_turn: int
    spawn: dict[str, int]


class ZoneData(TypedDict, total=False):
    description: str
    atmosphere: str
    exits: list[str]


class QuestData(TypedDict, total=False):
    status: str
    summary: str
    objective: str
    started_turn: int
    completed_turn: int


class ClaimData(TypedDict, total=False):
    text: str
    confidence: float
    speaker_id: str
    listener_ids: list[str]
    location_id: str
    about_object_id: str


__all__ = [
    "NpcData",
    "ZoneData",
    "QuestData",
    "ClaimData",
]
