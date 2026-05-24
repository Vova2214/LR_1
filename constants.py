from __future__ import annotations

QUEST_TERMINAL_STATUSES: tuple[str, ...] = (
    "inactive",
    "completed",
    "done",
    "closed",
    "failed",
    "archived",
)

NPC_OFFSTAGE_STATUS = "offstage"
TRACKING_QUEST_LINK_TYPE = "tracking_quest"
LOCATED_IN_LINK_TYPE = "located_in"
CARRIED_BY_LINK_TYPE = "carried_by"
ADJACENT_LINK_TYPE = "adjacent"
SESSION_PLAYER_REF = "session_player"

NPC_SOCIAL_LINK_TYPES: set[str] = {
    "knows",
    "friends_with",
    "allied_with",
    "hostile_to",
    "family",
    "employer",
    "employee",
    "rival",
}

REACTION_SUPPORT_LINK_TYPES: set[str] = {
    "friends_with",
    "allied_with",
    "family",
    "employer",
    "employee",
}

REACTION_CONFLICT_LINK_TYPES: set[str] = {"hostile_to", "rival"}

NPC_DEATH_PRESERVED_LINK_TYPES: set[str] = {"family", "knows"}

RECIPROCAL_SOCIAL_LINK_TYPES: set[str] = {"friends_with", "hostile_to", "allied_with", "rival", "family"}

ORPHANED_ITEMS_LOOKBACK_TURNS = 2

__all__ = [
    "QUEST_TERMINAL_STATUSES",
    "NPC_OFFSTAGE_STATUS",
    "TRACKING_QUEST_LINK_TYPE",
    "LOCATED_IN_LINK_TYPE",
    "CARRIED_BY_LINK_TYPE",
    "ADJACENT_LINK_TYPE",
    "SESSION_PLAYER_REF",
    "NPC_SOCIAL_LINK_TYPES",
    "REACTION_SUPPORT_LINK_TYPES",
    "REACTION_CONFLICT_LINK_TYPES",
    "NPC_DEATH_PRESERVED_LINK_TYPES",
    "RECIPROCAL_SOCIAL_LINK_TYPES",
    "ORPHANED_ITEMS_LOOKBACK_TURNS",
]
