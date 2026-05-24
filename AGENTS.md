## Project Skills

### Available skills

- `rpg-api-backend`: Develop and maintain the FastAPI backend in this repository. Use when changing routes, schemas, CRUD modules, config, background workers, or backend tests. (file: /Users/vova/Desktop/rpg/apps/api/.codex/skills/rpg-api-backend/SKILL.md)
- `rpg-turn-pipeline`: Work on turn execution, memory seeds, durable facts, context assembly, prompt resolution, LLM telemetry, or outbox scheduling linked to turns. (file: /Users/vova/Desktop/rpg/apps/api/.codex/skills/rpg-turn-pipeline/SKILL.md)
- `rpg-schema-migrations`: Change SQLAlchemy models, Alembic revisions, indexes, constraints, pgvector settings, or persisted JSON shapes. (file: /Users/vova/Desktop/rpg/apps/api/.codex/skills/rpg-schema-migrations/SKILL.md)

### How to use project skills

- Use the narrowest matching project skill before editing code in this repository.
- Combine `rpg-api-backend` with `rpg-schema-migrations` when a schema change also alters API behavior.
- Combine `rpg-turn-pipeline` with `rpg-schema-migrations` when a turn-flow change affects persisted turn, memory, or outbox data.
- Read only the specific `SKILL.md` files needed for the current task.
