# recap_search.py
# -*- coding: utf-8 -*-
"""
/search command for the recap bot.

Flow:
  1. Guard: persistence must be enabled.
  2. LLM normalises the natural-language query to JSON filters
     (query, date_from, date_to, participants, exact_terms).
  3. Embed the normalised query string.
  4. Hybrid retrieval: vector (cosine) + lexical (tsvector) with RRF merging,
     scoped to the current chat_id and optional date range.
  5. Optional LLM rerank of the top-5 candidate chunks.
  6. Load the best chunk's original messages from DB.
  7. LLM generates a grounded answer using only [id=N] markers (no HTML,
     no SQL, no Telegram links).
  8. Application converts [id=N] markers to validated HTML links
     (unknown IDs are silently dropped).
  9. Reply to the first message of the selected discussion.

Pure helpers (parse_search_filters, convert_id_markers_to_links) have no
I/O and can be unit-tested without a database or LLM.
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

import recap_db

logger = logging.getLogger("recap-bot.search")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RECAP_MODEL = os.getenv("RECAP_OPENAI_MODEL", "google/gemini-2.5-flash-lite")
RECAP_LLM_URL = os.getenv("RECAP_LLM_URL", "https://api.openai.com/v1")
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
EMBEDDING_MODEL = os.getenv("RECAP_EMBEDDING_MODEL", "openai/text-embedding-3-small")
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

def _link_prefix(chat_id: int, chat_username: Optional[str]) -> Optional[str]:
    if chat_username:
        return f"https://t.me/{chat_username}/"
    s = str(chat_id)
    if s.startswith("-100"):
        return f"https://t.me/c/{s[4:]}/"
    return None


def parse_search_filters(json_str: str) -> dict:
    """
    Parse the LLM-normalised search filter JSON into a plain dict.

    Returned keys (all optional except 'query'):
      query        str   — normalised query text
      date_from    datetime (UTC) — lower bound
      date_to      datetime (UTC) — upper bound
      participants list[str]
      exact_terms  list[str]

    On any parse error, returns {'query': original_input[:200]}.
    """
    text = json_str.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text.rstrip())
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"query": json_str.strip()[:200]}

    if not isinstance(data, dict):
        return {"query": json_str.strip()[:200]}

    result: dict = {"query": str(data.get("query", ""))[:500]}

    for field in ("date_from", "date_to"):
        val = data.get(field)
        if val and isinstance(val, str):
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
                try:
                    result[field] = datetime.strptime(val, fmt).replace(
                        tzinfo=timezone.utc
                    )
                    break
                except ValueError:
                    pass

    participants = data.get("participants")
    if isinstance(participants, list):
        result["participants"] = [str(p)[:100] for p in participants if p][:10]

    exact_terms = data.get("exact_terms")
    if isinstance(exact_terms, list):
        result["exact_terms"] = [str(t)[:100] for t in exact_terms if t][:10]

    return result


def convert_id_markers_to_links(
    text: str,
    link_prefix: Optional[str],
    valid_ids: set,
) -> str:
    """
    Replace [id=N] markers in *text* with HTML anchor tags.

    Rules:
    - ID must be in valid_ids, otherwise the marker is removed entirely.
    - If link_prefix is None (no stable permalink), the marker is removed too.
    - The generated href is: link_prefix + str(id)
    """
    def replace(m: re.Match) -> str:
        mid = int(m.group(1))
        if mid not in valid_ids or not link_prefix:
            return ""
        return f'<a href="{link_prefix}{mid}">#{mid}</a>'

    return re.sub(r"\[id=(\d+)\]", replace, text)


# ---------------------------------------------------------------------------
# LLM calls (synchronous, wrapped with asyncio.to_thread)
# ---------------------------------------------------------------------------

_NORMALISE_SYSTEM = (
    "Ты нормализуешь поисковые запросы по истории Telegram-чата. "
    "Верни ТОЛЬКО JSON-объект без markdown, без HTML, без пояснений."
)

_NORMALISE_USER = """\
Нормализуй запрос и извлеки фильтры. Верни JSON:
{{
  "query": "нормализованный запрос по-русски (обязательно)",
  "date_from": "YYYY-MM-DD или null",
  "date_to":   "YYYY-MM-DD или null",
  "participants": ["имя1", "имя2"] или null,
  "exact_terms": ["точная фраза"] или null
}}

Запрос пользователя: {query}"""


def _normalise_query_sync(query_text: str) -> str:
    resp = _get_client().chat.completions.create(
        model=RECAP_MODEL,
        messages=[
            {"role": "system", "content": _NORMALISE_SYSTEM},
            {"role": "user", "content": _NORMALISE_USER.format(query=query_text)},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    return (resp.choices[0].message.content or "").strip()


def _embed_sync(text: str) -> List[float]:
    resp = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


_RERANK_SYSTEM = (
    "Ты ранжируешь результаты поиска по релевантности к запросу. "
    "Верни ТОЛЬКО JSON-массив chunk_id (целые числа) в порядке убывания релевантности. "
    "Не более 3 элементов. Без markdown, без пояснений."
)


def _rerank_sync(chunks: List[dict], query_text: str) -> List[dict]:
    summaries = "\n".join(
        f"[chunk_id={c['id']}] {c['summary']}" for c in chunks
    )
    user = f"Запрос: {query_text}\n\nРезультаты:\n{summaries}"
    try:
        resp = _get_client().chat.completions.create(
            model=RECAP_MODEL,
            messages=[
                {"role": "system", "content": _RERANK_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.rstrip())
        ids = json.loads(raw.strip())
        if isinstance(ids, list):
            id_map = {c["id"]: c for c in chunks}
            reranked = [id_map[i] for i in ids if i in id_map]
            seen = {c["id"] for c in reranked}
            rest = [c for c in chunks if c["id"] not in seen]
            return reranked + rest
    except Exception as exc:
        logger.warning("Rerank failed (%s), using original order", exc)
    return chunks


_ANSWER_SYSTEM = (
    "Ты отвечаешь на вопросы по истории Telegram-чата. "
    "Ответ — по-русски, кратко и по делу. "
    "Ссылайся на источники через маркеры [id=N]. "
    "Не выдумывай — используй только предоставленные сообщения. "
    "Не генерируй HTML, SQL, Telegram-ссылки или код — только текст с маркерами [id=N]."
)

_ANSWER_USER = """\
Вопрос: {query}

Сообщения чата (хронологически):
{messages}"""


def _generate_answer_sync(
    messages: List[dict],
    query_text: str,
) -> str:
    lines: List[str] = []
    for m in messages:
        # Use the canonical citation ID (original media message when applicable)
        mid = m.get("media_source_message_id") or m.get("message_id")
        user = m.get("user_name", "?")
        text = (m.get("text") or "")[:300]
        lines.append(f"[id={mid}] {user}: {text}")

    resp = _get_client().chat.completions.create(
        model=RECAP_MODEL,
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM},
            {
                "role": "user",
                "content": _ANSWER_USER.format(
                    query=query_text,
                    messages="\n".join(lines),
                ),
            },
        ],
        temperature=0.3,
        max_tokens=700,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Telegram command handler
# ---------------------------------------------------------------------------

async def cmd_search(update, context) -> None:
    """Handle /search <query>."""
    from telegram.constants import ParseMode

    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return

    if not recap_db.is_enabled():
        await msg.reply_text(
            "Постоянная история не настроена — PostgreSQL не подключён. "
            "Задайте PG_DATABASE и перезапустите бот."
        )
        return

    if not context.args:
        await msg.reply_text("Использование: /search <запрос>")
        return

    query_text = " ".join(context.args)
    chat_id = chat.id
    chat_username = getattr(chat, "username", None)
    prefix = _link_prefix(chat_id, chat_username)

    processing_msg = await msg.reply_text("Ищу…")

    try:
        # 1. Normalise the query
        try:
            raw_filters = await asyncio.to_thread(_normalise_query_sync, query_text)
            filters = parse_search_filters(raw_filters)
        except Exception as exc:
            logger.warning("Query normalisation failed (%s), using raw query", exc)
            filters = {"query": query_text}

        norm_query = filters.get("query") or query_text
        date_from: Optional[datetime] = filters.get("date_from")
        date_to: Optional[datetime] = filters.get("date_to")

        # 2. Embed
        try:
            query_embedding = await asyncio.to_thread(_embed_sync, norm_query)
        except Exception as exc:
            logger.exception("Embedding failed: %s", exc)
            await processing_msg.edit_text("Не удалось обработать запрос (embedding).")
            return

        # 3. Hybrid retrieval
        chunks = await recap_db.hybrid_search(
            chat_id=chat_id,
            query_embedding=query_embedding,
            query_text=norm_query,
            date_from=date_from,
            date_to=date_to,
            limit=10,
        )

        if not chunks:
            await processing_msg.edit_text("По этому запросу ничего не найдено.")
            return

        # 4. Rerank top candidates
        if len(chunks) > 1:
            try:
                chunks = await asyncio.to_thread(_rerank_sync, chunks[:5], norm_query)
            except Exception as exc:
                logger.warning("Rerank error: %s", exc)

        best = chunks[0]
        chunk_message_ids: List[int] = list(best.get("message_ids") or [])

        # 5. Load messages
        messages = await recap_db.get_chunk_messages(chat_id, chunk_message_ids)
        if not messages:
            await processing_msg.edit_text("Не удалось загрузить сообщения фрагмента.")
            return

        # Build the set of valid citation IDs (canonical: media source if set)
        valid_ids: set = set()
        for m in messages:
            valid_ids.add(m.get("media_source_message_id") or m.get("message_id"))

        # 6. Generate grounded answer
        try:
            raw_answer = await asyncio.to_thread(
                _generate_answer_sync, messages, norm_query
            )
        except Exception as exc:
            logger.exception("Answer generation failed: %s", exc)
            await processing_msg.edit_text("Не удалось сформировать ответ.")
            return

        # 7. Convert [id=N] markers to validated HTML links
        answer_html = convert_id_markers_to_links(raw_answer, prefix, valid_ids)
        answer_html = answer_html.strip() or "Ответ не сформирован."

        # 8. Delete "Ищу…" message and send the answer
        try:
            await processing_msg.delete()
        except Exception:
            pass

        first_mid = best.get("first_message_id")
        send_kwargs = dict(
            chat_id=chat_id,
            text=answer_html,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        if first_mid:
            send_kwargs["reply_to_message_id"] = first_mid

        try:
            await context.bot.send_message(**send_kwargs)
        except Exception as exc:
            logger.warning(
                "Could not send reply to message_id=%s (%s); sending without reply",
                first_mid, exc,
            )
            send_kwargs.pop("reply_to_message_id", None)
            await context.bot.send_message(**send_kwargs)

    except Exception as exc:
        logger.exception("Search error: %s", exc)
        try:
            await processing_msg.edit_text("Произошла ошибка при поиске.")
        except Exception:
            pass
