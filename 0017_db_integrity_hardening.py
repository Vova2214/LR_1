"""Harden DB integrity checks for events, links, and session refs.

Revision ID: 0017_db_integrity_hardening
Revises: 0016_internal_summary
Create Date: 2026-03-03 16:40:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0017_db_integrity_hardening"
down_revision: Union[str, None] = "0016_internal_summary"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())

    if "events" in table_names:
        op.execute(
            """
            UPDATE events
            SET payload = '{}'::jsonb
            WHERE payload IS NULL OR jsonb_typeof(payload) <> 'object'
            """
        )
        op.execute("ALTER TABLE events DROP CONSTRAINT IF EXISTS ck_events_payload_is_object")
        op.create_check_constraint(
            "ck_events_payload_is_object",
            "events",
            "jsonb_typeof(payload) = 'object'",
        )

    if "links" in table_names:
        op.execute(
            """
            UPDATE links
            SET valid_from_turn = 0
            WHERE valid_from_turn < 0
            """
        )
        op.execute(
            """
            UPDATE links
            SET valid_to_turn = valid_from_turn
            WHERE valid_to_turn IS NOT NULL
              AND valid_to_turn < valid_from_turn
            """
        )
        op.execute("ALTER TABLE links DROP CONSTRAINT IF EXISTS ck_links_valid_from_turn_non_negative")
        op.create_check_constraint(
            "ck_links_valid_from_turn_non_negative",
            "links",
            "valid_from_turn >= 0",
        )

    if {"sessions", "objects", "turns"}.issubset(table_names):
        op.execute(
            """
            CREATE OR REPLACE FUNCTION fn_sessions_state_json_refs_validate()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            DECLARE
                player_raw text;
                player_uuid uuid;
                pending_raw text;
                pending_turn_int integer;
            BEGIN
                player_raw := COALESCE(NEW.state_json ->> 'player_object_id', '');
                IF player_raw <> '' THEN
                    BEGIN
                        player_uuid := player_raw::uuid;
                    EXCEPTION
                        WHEN invalid_text_representation THEN
                            RAISE EXCEPTION
                                'sessions.state_json.player_object_id must be a UUID, got: %',
                                player_raw;
                    END;

                    IF NOT EXISTS (
                        SELECT 1
                        FROM objects o
                        WHERE o.session_id = NEW.id
                          AND o.object_id = player_uuid
                          AND o.type = 'player'
                    ) THEN
                        RAISE EXCEPTION
                            'sessions.state_json.player_object_id (%) does not reference a player object in session %',
                            player_uuid, NEW.id;
                    END IF;
                END IF;

                pending_raw := COALESCE(NEW.state_json ->> 'pending_turn', '');
                IF pending_raw <> '' THEN
                    BEGIN
                        pending_turn_int := pending_raw::integer;
                    EXCEPTION
                        WHEN invalid_text_representation THEN
                            RAISE EXCEPTION
                                'sessions.state_json.pending_turn must be integer, got: %',
                                pending_raw;
                    END;

                    IF pending_turn_int < 0 THEN
                        RAISE EXCEPTION
                            'sessions.state_json.pending_turn must be >= 0, got: %',
                            pending_turn_int;
                    END IF;

                    IF NOT EXISTS (
                        SELECT 1
                        FROM turns t
                        WHERE t.session_id = NEW.id
                          AND t.turn_index = pending_turn_int
                    ) THEN
                        RAISE EXCEPTION
                            'sessions.state_json.pending_turn (%) does not reference a turn in session %',
                            pending_turn_int, NEW.id;
                    END IF;
                END IF;

                RETURN NEW;
            END;
            $$;
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())

    if "links" in table_names:
        op.execute("ALTER TABLE links DROP CONSTRAINT IF EXISTS ck_links_valid_from_turn_non_negative")
    if "events" in table_names:
        op.execute("ALTER TABLE events DROP CONSTRAINT IF EXISTS ck_events_payload_is_object")

    if {"sessions", "objects", "turns"}.issubset(table_names):
        op.execute(
            """
            CREATE OR REPLACE FUNCTION fn_sessions_state_json_refs_validate()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            DECLARE
                player_raw text;
                player_uuid uuid;
                pending_raw text;
                pending_turn_int integer;
            BEGIN
                player_raw := COALESCE(NEW.state_json ->> 'player_object_id', '');
                IF player_raw <> '' THEN
                    BEGIN
                        player_uuid := player_raw::uuid;
                    EXCEPTION
                        WHEN invalid_text_representation THEN
                            RAISE EXCEPTION
                                'sessions.state_json.player_object_id must be a UUID, got: %',
                                player_raw;
                    END;

                    IF NOT EXISTS (
                        SELECT 1
                        FROM objects o
                        WHERE o.session_id = NEW.id
                          AND o.object_id = player_uuid
                    ) THEN
                        RAISE EXCEPTION
                            'sessions.state_json.player_object_id (%) does not reference an object in session %',
                            player_uuid, NEW.id;
                    END IF;
                END IF;

                pending_raw := COALESCE(NEW.state_json ->> 'pending_turn', '');
                IF pending_raw <> '' THEN
                    BEGIN
                        pending_turn_int := pending_raw::integer;
                    EXCEPTION
                        WHEN invalid_text_representation THEN
                            RAISE EXCEPTION
                                'sessions.state_json.pending_turn must be integer, got: %',
                                pending_raw;
                    END;

                    IF pending_turn_int < 0 THEN
                        RAISE EXCEPTION
                            'sessions.state_json.pending_turn must be >= 0, got: %',
                            pending_turn_int;
                    END IF;

                    IF NOT EXISTS (
                        SELECT 1
                        FROM turns t
                        WHERE t.session_id = NEW.id
                          AND t.turn_index = pending_turn_int
                    ) THEN
                        RAISE EXCEPTION
                            'sessions.state_json.pending_turn (%) does not reference a turn in session %',
                            pending_turn_int, NEW.id;
                    END IF;
                END IF;

                RETURN NEW;
            END;
            $$;
            """
        )
