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
- search_messages()  — message-level full-text search (messages.tsv)
- get_context_window() — discussion around a set of hits
- vector_search_chunks() — semantic fallback over indexed chunks
- count_* helpers   — for /init_status
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
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
_search_enabled_cache: dict[int, bool] = {}


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
    _search_enabled_cache.clear()

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
    _search_enabled_cache.clear()


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

CREATE TABLE IF NOT EXISTS chat_settings (
    chat_id         BIGINT      PRIMARY KEY,
    search_enabled  BOOLEAN     NOT NULL DEFAULT FALSE,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS
    important_message_ids BIGINT[] NOT NULL DEFAULT ARRAY[]::BIGINT[];

-- Message-level full-text search. A generated column is backfilled by
-- PostgreSQL itself for already stored rows, so no reindex is needed.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('russian'::regconfig, coalesce(text, ''))) STORED;
CREATE INDEX IF NOT EXISTS messages_tsv ON messages USING GIN (tsv);
"""


async def _bootstrap_schema() -> None:
    async with _pool.connection() as conn:
        await conn.execute(_SCHEMA_SQL)
    logger.info("DB schema bootstrapped (embedding_dim=%s)", EMBEDDING_DIM)


# ---------------------------------------------------------------------------
# Per-chat search/indexing opt-in
# ---------------------------------------------------------------------------

async def is_search_enabled(chat_id: int) -> bool:
    """Return whether persistent storage, indexing and search are enabled."""
    if not _pool:
        return False
    if chat_id in _search_enabled_cache:
        return _search_enabled_cache[chat_id]

    async with _pool.connection() as conn:
        cur = await conn.execute(
            "SELECT search_enabled FROM chat_settings WHERE chat_id = %s",
            (chat_id,),
        )
        row = await cur.fetchone()

    enabled = bool(row and row[0])
    _search_enabled_cache[chat_id] = enabled
    return enabled


async def set_search_enabled(chat_id: int, enabled: bool) -> None:
    """Persist the per-chat opt-in flag and update the local fast-path cache."""
    if not _pool:
        raise RuntimeError("DB pool is not initialised")
    async with _pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO chat_settings (chat_id, search_enabled, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (chat_id) DO UPDATE SET
                search_enabled = EXCLUDED.search_enabled,
                updated_at = NOW()
            """,
            (chat_id, enabled),
        )
    _search_enabled_cache[chat_id] = enabled
    if enabled:
        wake_indexer()


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


async def upsert_message(row: dict) -> bool:
    """Upsert one message dict (as produced by _cm_to_db_row) into messages."""
    if not _pool or not await is_search_enabled(row["chat_id"]):
        return False
    async with _pool.connection() as conn:
        await conn.execute(_UPSERT_MSG_SQL, row)
    wake_indexer()
    return True


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
    """Return opted-in chat_ids with messages not yet assigned to a chunk."""
    if not _pool:
        return []
    async with _pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT DISTINCT m.chat_id
            FROM messages AS m
            JOIN chat_settings AS s ON s.chat_id = m.chat_id
            WHERE m.chunk_id IS NULL AND s.search_enabled = TRUE
            """
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
# Retrieval: message-level full-text search + chunk-level vector fallback
# ---------------------------------------------------------------------------

# Search requests (/search, /s, /п, ? …) and other bot commands are never
# useful search results — they only echo the question back. Filtering them at
# query time also cleans up histories imported before the import skipped them.
_NOT_COMMAND_SQL = r"text !~ '^\s*(/|\?)'"


async def search_messages(
    chat_id: int,
    tsquery: str,
    term_queries: List[str],
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    exclude_user_id: Optional[int] = None,
    limit: int = 300,
) -> List[dict]:
    """
    Full-text search over individual messages of *chat_id*.

    *tsquery* is a to_tsquery() expression (terms OR-ed together, already
    sanitised by recap_search.build_or_tsquery). *term_queries* are the
    individual OR-branches; each returned row carries ``matched_terms`` — the
    0-based indexes of the branches it matches — so the caller can measure how
    much of the query a discussion covers.

    Bot messages (live or the bot's own id from an import) and search/command
    texts are excluded.
    """
    if not _pool or not tsquery:
        return []

    conditions = [
        "chat_id = %(chat_id)s",
        "tsv @@ q",
        "NOT is_bot",
        "user_id IS DISTINCT FROM %(bot_id)s",
        _NOT_COMMAND_SQL,
    ]
    params: dict = {
        "chat_id": chat_id,
        "q": tsquery,
        "terms": term_queries,
        "bot_id": exclude_user_id,
        "limit": limit,
    }
    if date_from:
        conditions.append("date >= %(date_from)s")
        params["date_from"] = date_from
    if date_to:
        conditions.append("date <= %(date_to)s")
        params["date_to"] = date_to

    sql = f"""
    SELECT message_id, date, user_id, user_name, text, reply_to_message_id,
           media_source_message_id,
           ts_rank_cd(tsv, q) AS rank,
           ARRAY(
               SELECT (t.i - 1)::int
               FROM   unnest(%(terms)s::text[]) WITH ORDINALITY AS t(term, i)
               WHERE  tsv @@ to_tsquery('russian', t.term)
           ) AS matched_terms
    FROM   messages, to_tsquery('russian', %(q)s) AS q
    WHERE  {" AND ".join(conditions)}
    ORDER  BY rank DESC, date DESC
    LIMIT  %(limit)s
    """

    from psycopg.rows import dict_row
    async with _pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return await cur.fetchall()


async def get_context_window(
    chat_id: int,
    start_date: datetime,
    end_date: datetime,
    before: int = 3,
    after: int = 15,
    cap: int = 40,
    before_span: timedelta = timedelta(hours=1),
    after_span: timedelta = timedelta(hours=2),
    exclude_user_id: Optional[int] = None,
) -> List[dict]:
    """
    Load the discussion around [start_date, end_date]: a few messages before
    the first hit and more after the last one (answers follow questions),
    ordered chronologically and capped at *cap* messages. Neighbours are only
    taken within *before_span* / *after_span*, so an unrelated conversation
    hours later never leaks into the window. Bot messages and search/command
    texts are skipped, as in search_messages().
    """
    if not _pool:
        return []
    clean = (
        f"chat_id = %(chat_id)s AND NOT is_bot "
        f"AND user_id IS DISTINCT FROM %(bot_id)s AND {_NOT_COMMAND_SQL}"
    )
    sql = f"""
    SELECT * FROM (
        (SELECT * FROM messages
         WHERE {clean} AND date < %(start)s AND date >= %(from)s
         ORDER BY date DESC, message_id DESC LIMIT %(before)s)
        UNION ALL
        (SELECT * FROM messages
         WHERE {clean} AND date >= %(start)s AND date <= %(end)s
         ORDER BY date, message_id LIMIT %(cap)s)
        UNION ALL
        (SELECT * FROM messages
         WHERE {clean} AND date > %(end)s AND date <= %(until)s
         ORDER BY date, message_id LIMIT %(after)s)
    ) AS w
    ORDER BY date, message_id
    """
    params = {
        "chat_id": chat_id, "bot_id": exclude_user_id,
        "start": start_date, "end": end_date,
        "from": start_date - before_span, "until": end_date + after_span,
        "before": before, "after": after, "cap": cap,
    }
    from psycopg.rows import dict_row
    async with _pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
    return rows[:cap]


async def vector_search_chunks(
    chat_id: int,
    query_embedding: List[float],
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    limit: int = 10,
) -> List[dict]:
    """
    Semantic fallback: chunks of *chat_id* closest to *query_embedding* by
    cosine distance. Optional date filters keep chunks whose
    [start_date, end_date] interval overlaps the requested range.
    """
    if not _pool:
        return []

    conditions = ["chat_id = %(chat_id)s", "embedding IS NOT NULL"]
    params: dict = {
        "chat_id": chat_id,
        "emb": _vec_str(query_embedding),
        "limit": limit,
    }
    if date_from:
        conditions.append("end_date >= %(date_from)s")
        params["date_from"] = date_from
    if date_to:
        conditions.append("start_date <= %(date_to)s")
        params["date_to"] = date_to

    sql = f"""
    SELECT id, summary, message_ids, important_message_ids, first_message_id,
           start_date, end_date,
           embedding <=> %(emb)s::vector AS distance
    FROM   chunks
    WHERE  {" AND ".join(conditions)}
    ORDER  BY distance
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
