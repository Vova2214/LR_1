from __future__ import annotations

import json
import os
from collections.abc import Generator
from typing import Callable, Concatenate, ParamSpec, TypeVar

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

P = ParamSpec("P")
T = TypeVar("T")


def _read_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _read_optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _clamp_float(value: float, *, min_value: float, max_value: float) -> float:
    return min(max(float(value), min_value), max_value)


def _read_str_env(primary: str, *, legacy: str | None = None, default: str = "") -> str:
    raw = os.getenv(primary)
    if raw is not None and raw.strip():
        return raw.strip()
    if legacy:
        legacy_raw = os.getenv(legacy)
        if legacy_raw is not None and legacy_raw.strip():
            return legacy_raw.strip()
    return default


def _read_choice_env(name: str, *, allowed: set[str], default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in allowed:
        return normalized
    return default


def _read_csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name)
    source = default if raw is None else raw
    values = [part.strip() for part in source.split(",") if part.strip()]
    if values:
        return values
    fallback = default.strip()
    return [fallback] if fallback else []


def _read_json_object_env(name: str, default: str = "{}") -> dict[str, object]:
    raw = os.getenv(name)
    source = raw if raw is not None else default
    try:
        parsed = json.loads(source)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/rpg",
)
ENABLE_ALL_FLAGS = _read_bool_env("ENABLE_ALL_FLAGS", default=True)


def _read_feature_flag(name: str, *, default: bool | None = None) -> bool:
    # Global switch for local/dev runs: explicit per-flag env still wins.
    fallback = ENABLE_ALL_FLAGS if default is None else bool(default)
    return _read_bool_env(name, default=fallback)


def _is_free_model_name(model_name: str) -> bool:
    return str(model_name or "").strip().lower().endswith(":free")


ALLOW_DEBUG_PATCH = _read_feature_flag("ALLOW_DEBUG_PATCH")
ENABLE_DEBUG_ROUTER = _read_feature_flag("ENABLE_DEBUG_ROUTER")
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")
# ============================================================
# Только DeepSeek V4 Flash через OpenRouter — для ВСЕХ текстовых этапов
# (Narrator, Librarian, Assistant, Planning, Memory и т.д.)
# ============================================================
DEEPSEEK_V4_FLASH_MODEL = "deepseek/deepseek-v4-flash"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = _read_str_env("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

# Принудительно используем только DeepSeek V4 Flash на всех этапах
OPENROUTER_CHAT_MODEL = DEEPSEEK_V4_FLASH_MODEL
OPENROUTER_NARRATOR_MODEL = DEEPSEEK_V4_FLASH_MODEL
OPENROUTER_LIBRARIAN_MODEL = DEEPSEEK_V4_FLASH_MODEL
OPENROUTER_ASSISTANT_MODEL = DEEPSEEK_V4_FLASH_MODEL

OPENROUTER_CHAT_TIMEOUT_SECONDS = _read_float_env("OPENROUTER_CHAT_TIMEOUT_SECONDS", default=120.0)

# ============================================================
# Embeddings — оставлены на OpenRouter (Qwen), как было изначально
# ============================================================
USE_EMBEDDINGS = _read_feature_flag("USE_EMBEDDINGS")
OPENROUTER_EMBED_MODEL = "qwen/qwen3-embedding-8b"   # явно, без зависимости от .env

USE_REDIS_CACHE = _read_bool_env("USE_REDIS_CACHE", default=False)
REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_CACHE_NAMESPACE = os.getenv("REDIS_CACHE_NAMESPACE", "rpg-cache")
REDIS_CACHE_TTL_SECONDS = _read_int_env("REDIS_CACHE_TTL_SECONDS", default=1800)
REDIS_CACHE_CONNECT_TIMEOUT_SECONDS = _read_float_env(
    "REDIS_CACHE_CONNECT_TIMEOUT_SECONDS",
    default=0.2,
)
UVICORN_WORKERS = _read_int_env("UVICORN_WORKERS", default=2)
DB_POOL_SIZE = _read_int_env("DB_POOL_SIZE", default=10)
DB_MAX_OVERFLOW = _read_int_env("DB_MAX_OVERFLOW", default=10)
DB_POOL_TIMEOUT_SECONDS = _read_int_env("DB_POOL_TIMEOUT_SECONDS", default=30)
DB_POOL_RECYCLE_SECONDS = _read_int_env("DB_POOL_RECYCLE_SECONDS", default=1800)
DOCS_AUTH_ENABLED = _read_bool_env("DOCS_AUTH_ENABLED", default=True)
_FREE_MODEL_OPTIONAL_LLM_DEFAULT = ENABLE_ALL_FLAGS and False
USE_QUERY_REFORMULATOR = _read_feature_flag(
    "USE_QUERY_REFORMULATOR",
    default=_FREE_MODEL_OPTIONAL_LLM_DEFAULT,
)
USE_CHRONICLE_SUMMARIZER = _read_feature_flag(
    "USE_CHRONICLE_SUMMARIZER",
    default=_FREE_MODEL_OPTIONAL_LLM_DEFAULT,
)
# Memory Seed Normalizer включается тем же флагом что и Chronicle Summarizer
USE_CONTEXT_COMPRESSOR = _read_feature_flag("USE_CONTEXT_COMPRESSOR")
USE_CONTEXT_TRIMMER = _read_feature_flag("USE_CONTEXT_TRIMMER")
USE_UNIFIED_CONTEXT_SCORING = _read_bool_env("USE_UNIFIED_CONTEXT_SCORING", default=False)
USE_ELASTIC_ENTROPY_THRESHOLD = _read_bool_env("USE_ELASTIC_ENTROPY_THRESHOLD", default=False)
ELASTIC_MIN_RELEVANCE_THRESHOLD = _clamp_float(
    _read_float_env("ELASTIC_MIN_RELEVANCE_THRESHOLD", default=0.15),
    min_value=0.0,
    max_value=1.0,
)
USE_PROMPT_CACHE_LAYOUT = _read_bool_env("USE_PROMPT_CACHE_LAYOUT", default=False)
USE_CTX_WEIGHT_DECAY = _read_bool_env("USE_CTX_WEIGHT_DECAY", default=False)
CTX_WEIGHT_DECAY_LAMBDA = _clamp_float(
    _read_float_env("CTX_WEIGHT_DECAY_LAMBDA", default=0.10),
    min_value=0.0,
    max_value=0.99,
)
USE_REACTION_ENRICHER = _read_feature_flag(
    "USE_REACTION_ENRICHER",
    default=_FREE_MODEL_OPTIONAL_LLM_DEFAULT,
)
USE_WORLD_PROMPT_SUMMARIZER = _read_feature_flag(
    "USE_WORLD_PROMPT_SUMMARIZER",
    default=_FREE_MODEL_OPTIONAL_LLM_DEFAULT,
)
USE_SPLIT_NARRATOR_PATCHES = _read_feature_flag("USE_SPLIT_NARRATOR_PATCHES")
USE_STATE_FIRST_PIPELINE = _read_feature_flag("USE_STATE_FIRST_PIPELINE", default=True)
USE_WORLD_DIRECTOR = _read_feature_flag("USE_WORLD_DIRECTOR", default=USE_STATE_FIRST_PIPELINE)
USE_PROFILE_SYNTHESIZER = _read_feature_flag(
    "USE_PROFILE_SYNTHESIZER",
    default=_FREE_MODEL_OPTIONAL_LLM_DEFAULT,
)
USE_MEMORY_SEED_DEDUP = _read_bool_env("USE_MEMORY_SEED_DEDUP", default=False)
USE_MEMORY_IMPORTANCE_SCORER = _read_feature_flag("USE_MEMORY_IMPORTANCE_SCORER")
USE_STALE_MEMORY_DETECTOR = _read_feature_flag("USE_STALE_MEMORY_DETECTOR")
STALE_MEMORY_DETECTOR_SEED_LIMIT = _read_int_env("STALE_MEMORY_DETECTOR_SEED_LIMIT", default=15)
USE_MEMORY_CONSOLIDATOR = _read_feature_flag("USE_MEMORY_CONSOLIDATOR")
USE_CONSEQUENCES = _read_bool_env("USE_CONSEQUENCES", default=False)
USE_LORE_ADAPTATION = _read_feature_flag("USE_LORE_ADAPTATION")
LORE_ADAPTATION_MAX_CHARS = _read_int_env("LORE_ADAPTATION_MAX_CHARS", default=100_000)
LORE_ADAPTATION_TIMEOUT_SECONDS = _read_int_env("LORE_ADAPTATION_TIMEOUT_SECONDS", default=120)
LORE_ADAPTATION_RETRY_AFTER_SECONDS = _read_int_env("LORE_ADAPTATION_RETRY_AFTER_SECONDS", default=5)
CORS_ALLOW_ORIGINS = _read_csv_env("CORS_ALLOW_ORIGINS", default="*")
CORS_ALLOW_METHODS = _read_csv_env("CORS_ALLOW_METHODS", default="*")
CORS_ALLOW_HEADERS = _read_csv_env("CORS_ALLOW_HEADERS", default="*")
CORS_ALLOW_CREDENTIALS = _read_bool_env("CORS_ALLOW_CREDENTIALS", default=False)
MEMORY_CONSOLIDATOR_THRESHOLD = _read_int_env("MEMORY_CONSOLIDATOR_THRESHOLD", default=50)
MEMORY_CONSOLIDATOR_INTERVAL_TURNS = _read_int_env("MEMORY_CONSOLIDATOR_INTERVAL_TURNS", default=10)
SCHEMA_EMBEDDING_DIM = 4096
_configured_embedding_dim = _read_int_env("EMBEDDING_DIM", default=SCHEMA_EMBEDDING_DIM)
if _configured_embedding_dim != SCHEMA_EMBEDDING_DIM:
    raise RuntimeError(
        "EMBEDDING_DIM is incompatible with database schema: "
        f"configured={_configured_embedding_dim}, expected={SCHEMA_EMBEDDING_DIM}"
    )
EMBEDDING_DIM = SCHEMA_EMBEDDING_DIM
EMBED_SNIPPET_MAX_CHARS = _read_int_env("EMBED_SNIPPET_MAX_CHARS", default=1200)
RETRIEVAL_TOP_K = _read_int_env("RETRIEVAL_TOP_K", default=8)
DEDUP_SIM_THRESHOLD = _read_float_env("DEDUP_SIM_THRESHOLD", default=0.88)
ZONE_GLOBAL_DEDUP_THRESHOLD = _read_float_env("ZONE_GLOBAL_DEDUP_THRESHOLD", default=0.93)
USE_DEDUP_ARBITER = _read_feature_flag("USE_DEDUP_ARBITER")
DEDUP_ARBITER_MIN_SIM = _read_float_env("DEDUP_ARBITER_MIN_SIM", default=0.72)
WORLD_PROMPT_CHUNK_MAX_CHARS = _read_int_env("WORLD_PROMPT_CHUNK_MAX_CHARS", default=1400)
WORLD_PROMPT_TOP_K = _read_int_env("WORLD_PROMPT_TOP_K", default=3)
WORLD_PROMPT_FALLBACK_MAX_CHARS = _read_int_env("WORLD_PROMPT_FALLBACK_MAX_CHARS", default=1200)
TURN_CONTEXT_MAX_CHARS = _read_int_env("TURN_CONTEXT_MAX_CHARS", default=7500)
LEGACY_CONTEXT_CHARS_PER_TOKEN = 3.125
_configured_turn_context_max_tokens = _read_optional_int_env("TURN_CONTEXT_MAX_TOKENS")
if _configured_turn_context_max_tokens is None:
    TURN_CONTEXT_MAX_TOKENS = max(round(TURN_CONTEXT_MAX_CHARS / LEGACY_CONTEXT_CHARS_PER_TOKEN), 1)
else:
    TURN_CONTEXT_MAX_TOKENS = max(_configured_turn_context_max_tokens, 1)
TURN_CONTEXT_TOKEN_RESERVE = max(_read_int_env("TURN_CONTEXT_TOKEN_RESERVE", default=200), 0)
CONTEXT_TOKENIZER_ENCODING = _read_str_env("CONTEXT_TOKENIZER_ENCODING", default="o200k_base")
TURN_CONTEXT_TURNS_LIMIT = _read_int_env("TURN_CONTEXT_TURNS_LIMIT", default=8)
TURN_CONTEXT_SEMANTIC_TURNS_LIMIT = _read_int_env("TURN_CONTEXT_SEMANTIC_TURNS_LIMIT", default=3)
TURN_CONTEXT_EVENTS_LIMIT = _read_int_env("TURN_CONTEXT_EVENTS_LIMIT", default=16)
PENDING_TURN_TIMEOUT_SECONDS = _read_int_env("PENDING_TURN_TIMEOUT_SECONDS", default=120)
OUTBOX_PROCESSED_RETENTION_DAYS = _read_int_env("OUTBOX_PROCESSED_RETENTION_DAYS", default=7)
LLM_TELEMETRY_RETENTION_DAYS = _read_int_env("LLM_TELEMETRY_RETENTION_DAYS", default=30)
TELEMETRY_ROLLUP_ENABLED = _read_bool_env("TELEMETRY_ROLLUP_ENABLED", default=True)
RETENTION_JANITOR_INTERVAL_SECONDS = _read_int_env("RETENTION_JANITOR_INTERVAL_SECONDS", default=3600)
MEMORY_REVIEW_INTERVAL_SECONDS = _read_int_env("MEMORY_REVIEW_INTERVAL_SECONDS", default=86400)
OUTBOX_POLL_INTERVAL_SECONDS = _read_float_env("OUTBOX_POLL_INTERVAL_SECONDS", default=1.0)
OUTBOX_BATCH_SIZE = max(_read_int_env("OUTBOX_BATCH_SIZE", default=16), 1)
OUTBOX_PROCESSING_LEASE_SECONDS = max(_read_int_env("OUTBOX_PROCESSING_LEASE_SECONDS", default=300), 30)
LLM_PRICING_REVISION = _read_str_env("LLM_PRICING_REVISION", default="2026-03-04-r1")
LLM_MODEL_PRICING_JSON = _read_json_object_env("LLM_MODEL_PRICING_JSON", default="{}")
DEFAULT_SPAWN_ZONE_NAME = _read_str_env("DEFAULT_SPAWN_ZONE_NAME", default="Трактир")


class Base(DeclarativeBase):
    pass


engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT_SECONDS,
    pool_recycle=DB_POOL_RECYCLE_SECONDS,
)
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


def run_with_new_session(
    func: Callable[Concatenate[Session, P], T],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    db = SessionLocal()
    try:
        return func(db, *args, **kwargs)
    except Exception:
        db.rollback()
        raise
    finally:
        # ``Session.close()`` already tears down any open transaction for this
        # short-lived helper session. Avoid an explicit success-path rollback
        # here because SQLAlchemy expires loaded ORM state on rollback, which
        # breaks read routes that intentionally return detached model instances
        # for immediate serialization.
        db.close()


def get_db() -> Generator[Session, None, None]:
    """Yield a database session for the duration of a request.

    .. warning::

        This dependency does **not** auto-commit at request end. Every write
        path must use explicit ``with db.begin():`` transaction blocks.
        Any leftover transaction at request teardown is rolled back.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        if db.in_transaction():
            db.rollback()
        db.close()
