# recap_db.py
# -*- coding: utf-8 -*-
"""
PostgreSQL / pgvector persistence layer for the recap bot.

Provides:
- is_enabled()      — True when PG_DATABASE is set
- init_pool()       — open the connection pool and bootstrap schema
- close_pool()      — graceful shutdown
- get_write_queue() — asyncio.Queue fed by _add_to_history
- run_writer(q)     — background task that drains the write queue
- upsert_message()  — idempotent message INSERT … ON CONFLICT
- get/insert/set helpers for the indexing pipeline
- hybrid_search()   — vector + lexical RRF retrieval
- count_* helpers   — for /init_status
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger("recap-bot.db")

# ---------------------------------------------------------------------------
# Configuration (read once at import time; set before importing this module)
# ---------------------------------------------------------------------------

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DATABASE = os.getenv("PG_DATABASE")
PG_USERNAME = os.getenv("PG_USERNAME", "")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
EMBEDDING_DIM = int(os.getenv("RECAP_EMBEDDING_DIM", "1536"))

# ---------------------------------------------------------------------------
# State (module-level singletons)
# ---------------------------------------------------------------------------

_pool = None  # set by init_pool()
_write_queue: Optional[asyncio.Queue] = None
_wake_event: Optional[asyncio.Event] = None


def is_enabled() -> bool:
    """Returns True when a PostgreSQL database name is configured."""
    return bool(PG_DATABASE)


# ---------------------------------------------------------------------------
# Pool lifecycle
# ---------------------------------------------------------------------------

async def init_pool() -> None:
    """Open the async connection pool and bootstrap the database schema."""
    global _pool, _write_queue, _wake_event
    if not is_enabled():
        return

    try:
        from psycopg_pool import AsyncConnectionPool  # noqa: F401
    except ImportError:
        logger.error(
            "psycopg-pool is not installed — PostgreSQL persistence disabled. "
            "Install it with: pip install 'psycopg[binary,pool]'"
        )
        return

    from psycopg_pool import AsyncConnectionPool

    conninfo = (
        f"host={PG_HOST} port={PG_PORT} dbname={PG_DATABASE}"
        f" user={PG_USERNAME} password={PG_PASSWORD}"
    )

    _pool = AsyncConnectionPool(
        conninfo,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True},
        open=False,
    )
    await _pool.open()
    _write_queue = asyncio.Queue()
    _wake_event = asyncio.Event()

    await _bootstrap_schema()
    logger.info(
        "PostgreSQL pool initialised (host=%s db=%s embedding_dim=%s)",
        PG_HOST, PG_DATABASE, EMBEDDING_DIM,
    )


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_write_queue() -> Optional[asyncio.Queue]:
    return _write_queue


def get_wake_event() -> Optional[asyncio.Event]:
    """
    Event the indexer waits on while idle. Set by wake_indexer() whenever a
    message is written, so a fresh live message or /init upload is picked up
    immediately instead of waiting out the full INDEX_INTERVAL.
    """
    return _wake_event


def wake_indexer() -> None:
    """Signal the indexer to stop waiting and run another pass right away."""
    if _wake_event is not None:
        _wake_event.set()


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

_SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS messages (
    chat_id                    BIGINT       NOT NULL,
    message_id                 BIGINT       NOT NULL,
    chat_username              TEXT,
    user_id                    BIGINT,
    user_name                  TEXT         NOT NULL DEFAULT '',
    text                       TEXT         NOT NULL DEFAULT '',
    content_kind               TEXT         NOT NULL DEFAULT 'text',
    date                       TIMESTAMPTZ  NOT NULL,
    is_bot                     BOOLEAN      NOT NULL DEFAULT FALSE,
    reply_to_message_id        BIGINT,
    reply_to_user_name         TEXT,
    reply_to_text              TEXT,
    is_forwarded               BOOLEAN      NOT NULL DEFAULT FALSE,
    forward_from_name          TEXT,
    forward_from_chat_title    TEXT,
    forward_from_chat_username TEXT,
    forward_from_message_id    BIGINT,
    media_source_message_id    BIGINT,
    chunk_id                   BIGINT,
    PRIMARY KEY (chat_id, message_id)
);

CREATE INDEX IF NOT EXISTS messages_unindexed
    ON messages (chat_id, date, message_id)
    WHERE chunk_id IS NULL;

CREATE TABLE IF NOT EXISTS chunks (
    id                     BIGSERIAL    PRIMARY KEY,
    chat_id                BIGINT       NOT NULL,
    summary                TEXT         NOT NULL DEFAULT '',
    keywords               TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],
    message_ids            BIGINT[]     NOT NULL DEFAULT ARRAY[]::BIGINT[],
    important_message_ids  BIGINT[]     NOT NULL DEFAULT ARRAY[]::BIGINT[],
    first_message_id       BIGINT,
    start_date             TIMESTAMPTZ,
    end_date               TIMESTAMPTZ,
    embedding              vector({EMBEDDING_DIM}),
    tsv                    tsvector,
    created_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunks_chat_id ON chunks (chat_id);
CREATE INDEX IF NOT EXISTS chunks_tsv     ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_dates   ON chunks (chat_id, start_date, end_date);

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS
    important_message_ids BIGINT[] NOT NULL DEFAULT ARRAY[]::BIGINT[];
"""


async def _bootstrap_schema() -> None:
    async with _pool.connection() as conn:
        await conn.execute(_SCHEMA_SQL)
    logger.info("DB schema bootstrapped (embedding_dim=%s)", EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vec_str(v: List[float]) -> str:
    """Format a Python list as the pgvector string literal '[x,y,z,...]'."""
    return "[" + ",".join(str(float(x)) for x in v) + "]"


# ---------------------------------------------------------------------------
# Message persistence
# ---------------------------------------------------------------------------

_UPSERT_MSG_SQL = """
INSERT INTO messages (
    chat_id, message_id, chat_username, user_id, user_name,
    text, content_kind, date, is_bot,
    reply_to_message_id, reply_to_user_name, reply_to_text,
    is_forwarded, forward_from_name, forward_from_chat_title,
    forward_from_chat_username, forward_from_message_id,
    media_source_message_id
) VALUES (
    %(chat_id)s, %(message_id)s, %(chat_username)s, %(user_id)s, %(user_name)s,
    %(text)s, %(content_kind)s, %(date)s, %(is_bot)s,
    %(reply_to_message_id)s, %(reply_to_user_name)s, %(reply_to_text)s,
    %(is_forwarded)s, %(forward_from_name)s, %(forward_from_chat_title)s,
    %(forward_from_chat_username)s, %(forward_from_message_id)s,
    %(media_source_message_id)s
)
ON CONFLICT (chat_id, message_id) DO UPDATE SET
    text         = EXCLUDED.text,
    content_kind = EXCLUDED.content_kind,
    user_name    = EXCLUDED.user_name,
    media_source_message_id = COALESCE(
        messages.media_source_message_id,
        EXCLUDED.media_source_message_id
    )
"""


async def upsert_message(row: dict) -> None:
    """Upsert one message dict (as produced by _cm_to_db_row) into messages."""
    if not _pool:
        return
    async with _pool.connection() as conn:
        await conn.execute(_UPSERT_MSG_SQL, row)
    wake_indexer()


async def wipe_chat(chat_id: int) -> None:
    """
    Permanently delete all stored messages and chunks for one chat.

    Used by /init when a fresh export is uploaded, so the whole history gets
    cleanly reindexed instead of merging with (possibly stale/partial) data
    from a previous import or from live traffic.
    """
    if not _pool:
        return
    async with _pool.connection() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM chunks WHERE chat_id = %s", (chat_id,))
            await conn.execute("DELETE FROM messages WHERE chat_id = %s", (chat_id,))
    logger.info("Wiped stored history for chat_id=%s", chat_id)


# ---------------------------------------------------------------------------
# Indexing pipeline helpers
# ---------------------------------------------------------------------------

async def get_chats_with_unindexed() -> List[int]:
    """Return chat_ids that have at least one message not yet assigned to a chunk."""
    if not _pool:
        return []
    async with _pool.connection() as conn:
        cur = await conn.execute(
            "SELECT DISTINCT chat_id FROM messages WHERE chunk_id IS NULL"
        )
        return [r[0] for r in await cur.fetchall()]


async def get_unindexed_batch(chat_id: int, limit: int = 250) -> List[dict]:
    """Return up to `limit` oldest unindexed messages for a chat, ordered by date."""
    if not _pool:
        return []
    from psycopg.rows import dict_row
    async with _pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT * FROM messages
                WHERE  chat_id  = %s
                  AND  chunk_id IS NULL
                ORDER  BY date, message_id
                LIMIT  %s
                """,
                (chat_id, limit),
            )
            return await cur.fetchall()


async def insert_chunk(
    chat_id: int,
    summary: str,
    keywords: List[str],
    message_ids: List[int],
    first_message_id: int,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    embedding: Optional[List[float]],
    text_for_search: str = "",
    important_message_ids: Optional[List[int]] = None,
) -> int:
    """Insert a finalised chunk and return its new id.

    The full-text index (tsv) is built from the summary, keywords AND the raw
    message text so that exact words that appear in messages (but not in the
    LLM summary) are still lexically searchable. `embedding` may be None — the
    chunk is then still stored and remains lexically searchable.
    """
    if not _pool:
        raise RuntimeError("DB pool is not initialised")

    tsv_source = " ".join(
        part for part in (summary, " ".join(keywords), text_for_search) if part
    ).strip()
    emb_literal = _vec_str(embedding) if embedding else None

    async with _pool.connection() as conn:
        cur = await conn.execute(
            """
            INSERT INTO chunks (
                chat_id, summary, keywords, message_ids, important_message_ids,
                first_message_id, start_date, end_date,
                embedding, tsv
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s::vector,
                to_tsvector('russian', %s)
            ) RETURNING id
            """,
            (
                chat_id, summary, keywords, message_ids, important_message_ids or [],
                first_message_id, start_date, end_date,
                emb_literal, tsv_source,
            ),
        )
        row = await cur.fetchone()
        return row[0]


async def set_chunk_id(chat_id: int, message_ids: List[int], chunk_id: int) -> None:
    """Mark a set of messages as belonging to a finished chunk."""
    if not _pool:
        return
    async with _pool.connection() as conn:
        await conn.execute(
            "UPDATE messages SET chunk_id = %s WHERE chat_id = %s AND message_id = ANY(%s)",
            (chunk_id, chat_id, message_ids),
        )


async def get_chunk_messages(chat_id: int, message_ids: List[int]) -> List[dict]:
    """Load specific messages from DB, ordered chronologically."""
    if not _pool:
        return []
    from psycopg.rows import dict_row
    async with _pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT * FROM messages
                WHERE  chat_id    = %s
                  AND  message_id = ANY(%s)
                ORDER  BY date, message_id
                """,
                (chat_id, message_ids),
            )
            return await cur.fetchall()


# ---------------------------------------------------------------------------
# Hybrid retrieval (vector + lexical, reciprocal-rank fusion)
# ---------------------------------------------------------------------------

async def hybrid_search(
    chat_id: int,
    query_embedding: List[float],
    query_text: str,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    limit: int = 10,
) -> List[dict]:
    """
    Return chunks for *chat_id* ranked by a reciprocal-rank fusion of:
      • cosine-distance vector similarity  (embedding <=> query)
      • ts_rank full-text similarity       (tsv @@ websearch_to_tsquery)

    Optional date filters narrow results to chunks whose [start_date, end_date]
    interval overlaps the requested range.
    """
    if not _pool:
        return []

    base_conditions = ["chat_id = %(chat_id)s"]
    params: dict = {
        "chat_id": chat_id,
        "emb": _vec_str(query_embedding),
        "qtext": query_text,
        "limit": limit,
    }

    if date_from:
        base_conditions.append("end_date >= %(date_from)s")
        params["date_from"] = date_from
    if date_to:
        base_conditions.append("start_date <= %(date_to)s")
        params["date_to"] = date_to

    where = " AND ".join(base_conditions)

    sql = f"""
    WITH
    vec AS (
        SELECT id,
               ROW_NUMBER() OVER (ORDER BY embedding <=> %(emb)s::vector) AS rn
        FROM   chunks
        WHERE  {where}
          AND  embedding IS NOT NULL
        LIMIT  %(limit)s
    ),
    lex AS (
        SELECT id,
               ROW_NUMBER() OVER (
                   ORDER BY ts_rank(tsv, websearch_to_tsquery('russian', %(qtext)s)) DESC
               ) AS rn
        FROM   chunks
        WHERE  {where}
          AND  tsv @@ websearch_to_tsquery('russian', %(qtext)s)
        LIMIT  %(limit)s
    ),
    fused AS (
        SELECT
            COALESCE(vec.id, lex.id)                          AS id,
            COALESCE(1.0 / (60.0 + vec.rn), 0.0)
            + COALESCE(1.0 / (60.0 + lex.rn), 0.0)           AS score
        FROM vec
        FULL OUTER JOIN lex ON vec.id = lex.id
    )
    SELECT c.*
    FROM   fused
    JOIN   chunks c ON c.id = fused.id
    ORDER  BY fused.score DESC
    LIMIT  %(limit)s
    """

    from psycopg.rows import dict_row
    async with _pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return await cur.fetchall()


# ---------------------------------------------------------------------------
# Status / count helpers (for /init_status)
# ---------------------------------------------------------------------------

async def count_messages(chat_id: int) -> int:
    if not _pool:
        return 0
    async with _pool.connection() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM messages WHERE chat_id = %s", (chat_id,)
        )
        return (await cur.fetchone())[0]


async def count_chunks(chat_id: int) -> int:
    if not _pool:
        return 0
    async with _pool.connection() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE chat_id = %s", (chat_id,)
        )
        return (await cur.fetchone())[0]


async def count_unindexed(chat_id: int) -> int:
    if not _pool:
        return 0
    async with _pool.connection() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM messages WHERE chat_id = %s AND chunk_id IS NULL",
            (chat_id,),
        )
        return (await cur.fetchone())[0]


# ---------------------------------------------------------------------------
# Background writer task
# ---------------------------------------------------------------------------

async def run_writer(queue: asyncio.Queue) -> None:
    """
    Drain the write queue produced by _add_to_history and upsert each row.
    A sentinel value of None signals a clean shutdown.
    """
    logger.info("DB writer task started")
    while True:
        row = await queue.get()
        if row is None:
            queue.task_done()
            break
        try:
            await upsert_message(row)
        except Exception as exc:
            logger.exception("DB upsert failed: %s", exc)
        finally:
            queue.task_done()
    logger.info("DB writer task stopped")
