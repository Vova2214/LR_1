"""Guard sessions.state_json refs to objects/turns

Revision ID: 0009_state_json_refs
Revises: 0008_set_null_object_scope_fks
Create Date: 2026-03-02 05:25:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009_state_json_refs"
down_revision: Union[str, None] = "0008_set_null_object_scope_fks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
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

    op.execute("DROP TRIGGER IF EXISTS trg_sessions_state_json_refs_validate ON sessions")
    op.execute(
        """
        CREATE CONSTRAINT TRIGGER trg_sessions_state_json_refs_validate
        AFTER INSERT OR UPDATE OF state_json ON sessions
        DEFERRABLE INITIALLY DEFERRED
        FOR EACH ROW
        EXECUTE FUNCTION fn_sessions_state_json_refs_validate()
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION fn_objects_clear_session_player_object_ref()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            UPDATE sessions
            SET state_json = jsonb_set(
                COALESCE(state_json, '{}'::jsonb),
                '{player_object_id}',
                'null'::jsonb,
                true
            )
            WHERE id = OLD.session_id
              AND COALESCE(state_json ->> 'player_object_id', '') = OLD.object_id::text;
            RETURN OLD;
        END;
        $$;
        """
    )

    op.execute("DROP TRIGGER IF EXISTS trg_objects_clear_session_player_object_ref ON objects")
    op.execute(
        """
        CREATE TRIGGER trg_objects_clear_session_player_object_ref
        AFTER DELETE ON objects
        FOR EACH ROW
        EXECUTE FUNCTION fn_objects_clear_session_player_object_ref()
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION fn_turns_clear_session_pending_turn_ref()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            UPDATE sessions
            SET state_json = COALESCE(state_json, '{}'::jsonb)
                - 'pending_turn'
                - 'pending_turn_started_at'
            WHERE id = OLD.session_id
              AND COALESCE(state_json ->> 'pending_turn', '') = OLD.turn_index::text;
            RETURN OLD;
        END;
        $$;
        """
    )

    op.execute("DROP TRIGGER IF EXISTS trg_turns_clear_session_pending_turn_ref ON turns")
    op.execute(
        """
        CREATE TRIGGER trg_turns_clear_session_pending_turn_ref
        AFTER DELETE ON turns
        FOR EACH ROW
        EXECUTE FUNCTION fn_turns_clear_session_pending_turn_ref()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_turns_clear_session_pending_turn_ref ON turns")
    op.execute("DROP FUNCTION IF EXISTS fn_turns_clear_session_pending_turn_ref()")

    op.execute("DROP TRIGGER IF EXISTS trg_objects_clear_session_player_object_ref ON objects")
    op.execute("DROP FUNCTION IF EXISTS fn_objects_clear_session_player_object_ref()")

    op.execute("DROP TRIGGER IF EXISTS trg_sessions_state_json_refs_validate ON sessions")
    op.execute("DROP FUNCTION IF EXISTS fn_sessions_state_json_refs_validate()")
