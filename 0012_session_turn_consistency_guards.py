"""Drop state_json.current_turn and enforce session/turn consistency

Revision ID: 0012_session_turn_consistency
Revises: 0011_active_link_edge_uq
Create Date: 2026-03-02 14:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0012_session_turn_consistency"
down_revision: Union[str, None] = "0011_active_link_edge_uq"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_STATE_JSON_DEFAULT_NO_CURRENT_TURN = sa.text(
    "'{\"world_name\":\"Unnamed\",\"difficulty\":\"normal\",\"rules\":{},\"player_object_id\": null}'::jsonb"
)
_STATE_JSON_DEFAULT_WITH_CURRENT_TURN = sa.text(
    "'{\"world_name\":\"Unnamed\",\"difficulty\":\"normal\",\"rules\":{},\"player_object_id\": null,\"current_turn\": 0}'::jsonb"
)


def upgrade() -> None:
    op.execute(
        """
        UPDATE sessions
        SET state_json = COALESCE(state_json, '{}'::jsonb) - 'current_turn'
        WHERE COALESCE(state_json, '{}'::jsonb) ? 'current_turn'
        """
    )

    op.execute(
        """
        UPDATE sessions AS s
        SET turn_index = COALESCE(t.max_turn, 0)
        FROM (
            SELECT session_id, MAX(turn_index) AS max_turn
            FROM turns
            GROUP BY session_id
        ) AS t
        WHERE s.id = t.session_id
          AND s.turn_index IS DISTINCT FROM COALESCE(t.max_turn, 0)
        """
    )
    op.execute(
        """
        UPDATE sessions AS s
        SET turn_index = 0
        WHERE NOT EXISTS (
            SELECT 1
            FROM turns t
            WHERE t.session_id = s.id
        )
          AND s.turn_index <> 0
        """
    )

    # Flush deferred constraint triggers fired by UPDATE statements above
    # before altering the sessions table definition in the same transaction.
    op.execute("SET CONSTRAINTS ALL IMMEDIATE")

    op.alter_column("sessions", "state_json", server_default=_STATE_JSON_DEFAULT_NO_CURRENT_TURN)
    op.create_check_constraint(
        "ck_sessions_state_json_no_current_turn",
        "sessions",
        "NOT (state_json ? 'current_turn')",
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION fn_assert_session_turn_index_consistency()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            sid uuid;
            session_turn integer;
            max_turn integer;
        BEGIN
            IF TG_TABLE_NAME = 'sessions' THEN
                sid := COALESCE(NEW.id, OLD.id);
            ELSE
                sid := COALESCE(NEW.session_id, OLD.session_id);
            END IF;

            IF sid IS NULL THEN
                RETURN NULL;
            END IF;

            SELECT s.turn_index
            INTO session_turn
            FROM sessions s
            WHERE s.id = sid;

            IF session_turn IS NULL THEN
                RETURN NULL;
            END IF;

            SELECT COALESCE(MAX(t.turn_index), 0)
            INTO max_turn
            FROM turns t
            WHERE t.session_id = sid;

            IF session_turn <> max_turn THEN
                RAISE EXCEPTION
                    'sessions.turn_index (%) must equal max(turns.turn_index) (%) for session %',
                    session_turn, max_turn, sid
                    USING ERRCODE = '23514';
            END IF;

            RETURN NULL;
        END;
        $$;
        """
    )

    op.execute("DROP TRIGGER IF EXISTS trg_turns_assert_session_turn_index_consistency ON turns")
    op.execute(
        """
        CREATE CONSTRAINT TRIGGER trg_turns_assert_session_turn_index_consistency
        AFTER INSERT OR UPDATE OR DELETE ON turns
        DEFERRABLE INITIALLY DEFERRED
        FOR EACH ROW
        EXECUTE FUNCTION fn_assert_session_turn_index_consistency()
        """
    )

    op.execute("DROP TRIGGER IF EXISTS trg_sessions_assert_session_turn_index_consistency ON sessions")
    op.execute(
        """
        CREATE CONSTRAINT TRIGGER trg_sessions_assert_session_turn_index_consistency
        AFTER INSERT OR UPDATE OF turn_index ON sessions
        DEFERRABLE INITIALLY DEFERRED
        FOR EACH ROW
        EXECUTE FUNCTION fn_assert_session_turn_index_consistency()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_sessions_assert_session_turn_index_consistency ON sessions")
    op.execute("DROP TRIGGER IF EXISTS trg_turns_assert_session_turn_index_consistency ON turns")
    op.execute("DROP FUNCTION IF EXISTS fn_assert_session_turn_index_consistency()")

    op.drop_constraint("ck_sessions_state_json_no_current_turn", "sessions", type_="check")
    op.alter_column("sessions", "state_json", server_default=_STATE_JSON_DEFAULT_WITH_CURRENT_TURN)

    op.execute(
        """
        UPDATE sessions
        SET state_json = jsonb_set(
            COALESCE(state_json, '{}'::jsonb),
            '{current_turn}',
            to_jsonb(turn_index),
            true
        )
        """
    )
