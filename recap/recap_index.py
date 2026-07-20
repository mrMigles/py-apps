# recap_index.py
# -*- coding: utf-8 -*-
"""
Periodic LLM-based indexing pipeline for the recap bot.

Background loop (run_indexer):
  1. Find chats with unindexed messages.
  2. Load up to INDEX_BATCH oldest unindexed messages per chat.
  3. Ask RECAP_MODEL to split them into coherent discussion chunks (strict JSON).
  4. Validate every returned message ID against the batch.
  5. Embed each complete chunk (summary + keywords) via the embedding model.
  6. Store the chunk in PostgreSQL and mark its member messages as indexed.
  7. Carry the trailing open chunk into the next run (with a force-close
     safeguard when the batch is full and yields only one incomplete chunk).

Pure helper functions (parse_chunk_json, build_index_text) are kept free of
I/O so they can be unit-tested without a database or LLM.
"""

import asyncio
import json
import logging
import os
import re
from typing import List, Optional, Set

import recap_db

logger = logging.getLogger("recap-bot.index")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RECAP_MODEL = os.getenv("RECAP_OPENAI_MODEL", "google/gemini-2.5-flash-lite")
RECAP_LLM_URL = os.getenv("RECAP_LLM_URL", "https://api.openai.com/v1")
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
EMBEDDING_MODEL = os.getenv("RECAP_EMBEDDING_MODEL", "openai/text-embedding-3-small")
INDEX_INTERVAL = int(os.getenv("RECAP_INDEX_INTERVAL_SECONDS", "120"))
# A batch of messages must fit in one LLM response as strict JSON. Keeping this
# small prevents the response from overflowing max_tokens (which would truncate
# the JSON, fail parsing and stall the pipeline on the same batch forever).
INDEX_BATCH = int(os.getenv("RECAP_INDEX_BATCH", "60"))
LLM_TIMEOUT = float(os.getenv("RECAP_LLM_TIMEOUT_SECONDS", "120"))

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(
            api_key=OPENAI_TOKEN, base_url=RECAP_LLM_URL, timeout=LLM_TIMEOUT
        )
    return _client


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable)
# ---------------------------------------------------------------------------

def build_index_text(rows: List[dict]) -> str:
    """
    Build the compact numbered conversation text sent to the chunking LLM.
    Each row is a messages-table dict (same shape as get_unindexed_batch output).
    """
    lines: List[str] = []
    for r in rows:
        kind = r.get("content_kind", "text")
        user = r.get("user_name") or f"User {r.get('user_id', '?')}"
        text = (r.get("text") or "").strip()[:240]
        mid = r.get("message_id")

        if kind == "voice_transcript":
            line = f"[id={mid}] {user} [voice transcript]: {text}"
        elif kind == "video_note_transcript":
            line = f"[id={mid}] {user} [video note transcript]: {text}"
        elif kind == "image_description":
            line = f"[id={mid}] {user} [image]: {text}"
        else:
            line = f"[id={mid}] {user}: {text}"

        if r.get("is_forwarded"):
            origin = (
                r.get("forward_from_name")
                or r.get("forward_from_chat_title")
                or "Unknown"
            )
            line += f" (FORWARDED from {origin})"
        if r.get("reply_to_message_id"):
            line += f" (reply_to_id={r['reply_to_message_id']})"

        lines.append(line)
    return "\n".join(lines)


def parse_chunk_json(json_str: str, valid_ids: Set[int]) -> List[dict]:
    """
    Parse and validate the LLM's chunking response.

    - Strips markdown code fences if present.
    - Drops any message ID not in valid_ids.
    - Drops chunks that have no valid IDs after filtering.
    - Caps summary at 500 chars, keywords at 20 items of 50 chars each.
    - Returns a list of dicts with keys:
        message_ids, summary, keywords, important_message_ids, is_complete
    """
    text = json_str.strip()
    # Strip markdown code fences
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text.rstrip())
    text = text.strip()

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Chunk LLM returned non-JSON: %.200s", text)
        return []

    if not isinstance(raw, list):
        logger.warning("Chunk LLM response is not a JSON array")
        return []

    result: List[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        raw_ids = item.get("message_ids", [])
        ids = [i for i in raw_ids if isinstance(i, int) and i in valid_ids]
        if not ids:
            continue

        raw_imp = item.get("important_message_ids", [])
        imp_ids = [i for i in raw_imp if isinstance(i, int) and i in valid_ids]

        result.append({
            "message_ids": ids,
            "summary": str(item.get("summary", ""))[:500],
            "keywords": [
                str(k)[:50]
                for k in item.get("keywords", [])
                if isinstance(k, str)
            ][:20],
            "important_message_ids": imp_ids,
            "is_complete": bool(item.get("is_complete", True)),
        })

    return result


# ---------------------------------------------------------------------------
# LLM calls (synchronous, run in thread)
# ---------------------------------------------------------------------------

_CHUNK_SYSTEM = (
    "Ты анализируешь переписку в Telegram и разбиваешь её на смысловые блоки. "
    "Отвечай ТОЛЬКО валидным JSON-массивом без markdown, без HTML, без ссылок, без пояснений."
)

_CHUNK_USER_TMPL = """\
Ниже идут сообщения чата. Разбей их на смысловые блоки (темы, ветки обсуждений).
Формат строки: [id=N] Автор: текст  (возможны пометки FORWARDED, reply_to_id)

Для каждого блока верни JSON-объект:
{{
  "message_ids": [список целых чисел id ИЗ ПРЕДОСТАВЛЕННЫХ СООБЩЕНИЙ],
  "summary": "краткое описание по-русски, 2–3 предложения",
  "keywords": ["ключевое слово", ...],
  "important_message_ids": [подмножество message_ids с ключевыми сообщениями],
  "is_complete": true
}}

Правила:
- Используй ТОЛЬКО id из этого набора: {valid_ids_preview}
- Каждое сообщение — ровно в одном блоке, ни одно не пропускай
- "summary" — одно короткое предложение; "keywords" — не более 5 слов
- "is_complete": false ТОЛЬКО для последнего блока, если обсуждение явно не завершено
- Верни ТОЛЬКО компактный JSON-массив в ОДНУ строку, без переносов строк, отступов и лишнего текста

Сообщения:
{conv_text}"""


def _call_chunk_llm_sync(conv_text: str, valid_ids: Set[int]) -> str:
    preview = repr(sorted(valid_ids)[:30])
    user_prompt = _CHUNK_USER_TMPL.format(
        valid_ids_preview=preview,
        conv_text=conv_text,
    )
    resp = _get_client().chat.completions.create(
        model=RECAP_MODEL,
        messages=[
            {"role": "system", "content": _CHUNK_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=8000,
    )
    return (resp.choices[0].message.content or "").strip()


def _embed_sync(text: str) -> List[float]:
    resp = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


# ---------------------------------------------------------------------------
# Per-chat indexing
# ---------------------------------------------------------------------------

def _fallback_chunks(rows: List[dict], window: int = 15) -> List[dict]:
    """Deterministically split a batch into fixed-size windows.

    Used when the LLM fails to return usable JSON, so the pipeline always makes
    progress with reasonably-sized (searchable) chunks instead of one giant one.
    """
    chunks: List[dict] = []
    for i in range(0, len(rows), window):
        group = rows[i:i + window]
        chunks.append({
            "message_ids": [r["message_id"] for r in group],
            "summary": build_index_text(group)[:500],
            "keywords": [],
            "important_message_ids": [],
            "is_complete": True,
        })
    return chunks


async def _index_chat(chat_id: int) -> int:
    """Index one batch for a chat. Returns the number of messages indexed."""
    rows = await recap_db.get_unindexed_batch(chat_id, INDEX_BATCH)
    if not rows:
        return 0

    batch_full = len(rows) >= INDEX_BATCH
    valid_ids: Set[int] = {r["message_id"] for r in rows}
    id_to_row = {r["message_id"]: r for r in rows}

    conv_text = build_index_text(rows)

    try:
        raw_json = await asyncio.to_thread(_call_chunk_llm_sync, conv_text, valid_ids)
    except Exception as exc:
        logger.exception("Chunk LLM error for chat_id=%s: %s", chat_id, exc)
        return 0

    chunks = parse_chunk_json(raw_json, valid_ids)
    if not chunks:
        logger.warning("No valid chunks from LLM for chat_id=%s", chat_id)
        if not batch_full:
            # Small/incomplete backlog: likely a transient LLM hiccup or an
            # ongoing discussion — retry later without forcing a chunk.
            return 0
        # Full batch yielded nothing usable (e.g. truncated JSON). Fall back to
        # deterministic fixed-size windows so the pipeline always makes progress
        # instead of retrying the same batch forever.
        logger.info("Falling back to windowed chunks for chat_id=%s", chat_id)
        chunks = _fallback_chunks(rows)

    # Carry-over: the last chunk may be marked incomplete if the discussion
    # continues beyond this batch.  We skip it so its messages are retried
    # with newer context next run.
    #
    # Safeguard: if the whole batch produced only one chunk and it is still
    # open, force-close it — otherwise we'd loop forever on the same batch.
    if not chunks[-1]["is_complete"]:
        if batch_full and len(chunks) == 1:
            logger.info(
                "Force-closing single open chunk (full batch) for chat_id=%s", chat_id
            )
            chunks[-1]["is_complete"] = True
        else:
            # Drop the last incomplete chunk; its messages stay unindexed.
            chunks = chunks[:-1]

    indexed_count = 0
    for chunk in chunks:
        if not chunk["is_complete"]:
            continue

        mids = chunk["message_ids"]
        chunk_rows = [id_to_row[m] for m in mids if m in id_to_row]
        if not chunk_rows:
            continue

        dates = [r["date"] for r in chunk_rows if r.get("date")]
        start_date = min(dates) if dates else None
        end_date = max(dates) if dates else None
        first_mid = min(mids)

        # Concatenated message text keeps exact words (that may be absent from
        # the LLM summary) lexically searchable.
        text_for_search = " ".join(
            (r.get("text") or "").strip() for r in chunk_rows
        ).strip()[:8000]

        embed_text = (
            chunk["summary"] + " " + " ".join(chunk["keywords"]) + " " + text_for_search
        ).strip()
        # Embedding failures must NOT stall indexing: store the chunk without an
        # embedding — it stays lexically searchable and won't be retried forever.
        try:
            embedding = await asyncio.to_thread(_embed_sync, embed_text)
        except Exception as exc:
            logger.warning(
                "Embedding error for chunk in chat_id=%s (storing without embedding): %s",
                chat_id, exc,
            )
            embedding = None

        try:
            chunk_id = await recap_db.insert_chunk(
                chat_id=chat_id,
                summary=chunk["summary"],
                keywords=chunk["keywords"],
                message_ids=mids,
                first_message_id=first_mid,
                start_date=start_date,
                end_date=end_date,
                embedding=embedding,
                text_for_search=text_for_search,
                important_message_ids=chunk["important_message_ids"],
            )
            await recap_db.set_chunk_id(chat_id, mids, chunk_id)
            indexed_count += len(mids)
            logger.info(
                "Indexed chunk id=%s for chat_id=%s (%s messages)",
                chunk_id, chat_id, len(mids),
            )
        except Exception as exc:
            logger.exception(
                "DB error storing chunk for chat_id=%s: %s", chat_id, exc
            )

    return indexed_count


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------

async def run_indexer() -> None:
    """
    Background coroutine that keeps chats fully indexed.

    While there is a backlog it processes batches back-to-back (continuous
    indexing — important during a large /init import), and only sleeps for
    INDEX_INTERVAL once every chat is fully caught up.
    """
    logger.info(
        "Indexer started (interval=%ss, batch_size=%s)", INDEX_INTERVAL, INDEX_BATCH
    )
    while True:
        if not recap_db.is_enabled():
            try:
                await asyncio.sleep(INDEX_INTERVAL)
                continue
            except asyncio.CancelledError:
                break

        did_work = False
        try:
            chat_ids = await recap_db.get_chats_with_unindexed()
            for chat_id in chat_ids:
                try:
                    if await _index_chat(chat_id):
                        did_work = True
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception(
                        "Indexing failed for chat_id=%s: %s", chat_id, exc
                    )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.exception("Indexer loop error: %s", exc)
            did_work = False

        # Keep draining immediately while progress is being made; otherwise idle
        # until the next interval. asyncio.sleep(0) yields control between passes.
        try:
            await asyncio.sleep(0 if did_work else INDEX_INTERVAL)
        except asyncio.CancelledError:
            break

    logger.info("Indexer stopped")
