# Architecture Decisions

## ADR-001: Session-Isolated Memory Model (Gap 3)

### Context
Current data model is intentionally session-scoped (`session_id` across objects, links, events, turns, embeddings). There is no shared world namespace across sessions.

### Decision
Keep memory/session isolation as the default architecture. Do not introduce `world_id` or cross-session shared object namespaces in this batch.

### Consequences
- New sessions stay deterministic and independent.
- No implicit data leakage between players/sessions.
- Multi-session continuity requires future explicit design work.

### Revisit Trigger
Revisit when product requirements include persistent shared worlds across multiple player sessions.

## ADR-002: World Prompt Indexing Scope (Gap 5)

### Context
World prompt chunk retrieval is indexed from raw `world_prompt`, while runtime narrator constitution merges `world_prompt_for_system` with `lore_profile_for_system`.

### Decision
Keep chunk indexing prompt-only. Keep lore-profile merge as a separate runtime stage for system prompt composition.

### Consequences
- Index invalidation remains simple and stable.
- Retrieval quality remains centered on base world prompt semantics.
- Lore profile still affects narration via merged constitution, but not chunk similarity search.

### Revisit Trigger
Revisit if lore adaptation payload grows and retrieval quality metrics show prompt-only indexing is insufficient.

## ADR-003: NPC Knowledge Source of Truth (Gap 7)

### Context
Narrator memory seeds are global context, while NPC-specific knowledge boundaries are built from claim graph links (`asserted`/`heard` relationships and validity windows).

### Decision
Keep claim graph as the canonical source for NPC-scoped knowledge. Do not add seed-to-claim semantic cross-linking in this batch.

### Consequences
- NPC epistemic boundaries stay explicit and auditable.
- No extra fuzzy matching complexity in hot context path.
- Some potentially useful seed hints remain global-only until future enhancement.

### Revisit Trigger
Revisit if NPC dialogue quality requires memory-seed provenance mapping to claims.
