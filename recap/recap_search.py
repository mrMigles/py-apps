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
  8. Application converts [id=N] markers to validated HTML links, or to
     plain-text "(#N)" references when the chat has no stable permalink
     format (e.g. a basic, non-super group); unknown IDs are dropped.
  9. Reply to the message the answer actually cites (falling back through
     important/first/any chunk message) — old imported messages may no
     longer exist in the live chat, so several candidates are tried.

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

SEARCH_TEXT_ALIAS_PATTERN = re.compile(
    r"^(?:/п(?:@[A-Za-z0-9_]+)?|\?)(?:\s|$)",
    re.IGNORECASE,
)
_SEARCH_QUERY_PATTERN = re.compile(
    r"^(?:/(?:search|s|п)(?:@[A-Za-z0-9_]+)?|\?)(?:\s+(?P<query>.*))?$",
    re.IGNORECASE | re.DOTALL,
)

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


def ensure_citation(
    answer: str,
    valid_ids: List[int],
) -> str:
    """
    Guarantee the answer carries at least one [id=N] citation marker.

    Some models produce a well-grounded answer but forget the [id=N] marker
    convention entirely — in that case convert_id_markers_to_links() has
    nothing to turn into a link, so the user gets an answer with no source
    at all. If *answer* has no marker already, append markers for the given
    *valid_ids* (already ordered by relevance/priority, most important first;
    at most 3 are appended) so a link always makes it into the final message.
    """
    if re.search(r"\[id=\d+\]", answer):
        return answer
    if not valid_ids:
        return answer
    markers = " ".join(f"[id={mid}]" for mid in valid_ids[:3])
    return f"{answer} {markers}".strip()


def extract_cited_ids(text: str, valid_ids: set) -> List[int]:
    """
    Return the [id=N] message ids cited in *text*, in order of first
    appearance, deduplicated and restricted to valid_ids.

    Used to pick reply-to-message targets that the answer actually grounds
    itself in — more relevant than always using the chunk's chronologically
    first message, which may since have been deleted from the live chat.
    """
    seen = set()
    ids: List[int] = []
    for m in re.finditer(r"\[id=(\d+)\]", text):
        mid = int(m.group(1))
        if mid in valid_ids and mid not in seen:
            seen.add(mid)
            ids.append(mid)
    return ids


def convert_id_markers_to_links(
    text: str,
    link_prefix: Optional[str],
    valid_ids: set,
) -> str:
    """
    Replace [id=N] markers in *text* with HTML anchor tags.

    Rules:
    - ID must be in valid_ids, otherwise the marker is removed entirely.
    - If link_prefix is None (no stable permalink exists for this chat — e.g.
      a basic, non-super group has no t.me/c/... URL scheme at all), the
      marker becomes a plain-text "(#N)" reference instead of a link, so the
      answer still shows *something* traceable rather than silently losing
      every citation.
    - The generated href is: link_prefix + str(id)
    """
    def replace(m: re.Match) -> str:
        mid = int(m.group(1))
        if mid not in valid_ids:
            return ""
        if not link_prefix:
            return f"(#{mid})"
        return f'<a href="{link_prefix}{mid}">#{mid}</a>'

    text = re.sub(r"\[id=(\d+)\]", replace, text)

    # Some models ignore the [id=N] convention and instead dump a raw list of
    # message IDs like "[315950, 315957, ...]". Strip any leftover bracketed
    # groups that contain only digits, commas and whitespace — they are never
    # meaningful prose and only leak internal IDs to the user.
    text = re.sub(r"\[\s*\d[\d,\s]*\]", "", text)

    # Collapse whitespace left behind by removed markers.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    return text.strip()


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
    "Ссылайся на источники, вставляя маркер [id=N] сразу после соответствующего "
    "факта, где N — id одного сообщения. "
    "НЕ перечисляй id списком и НЕ выводи id, если по теме ничего не нашлось. "
    "Если ответа в предоставленных сообщениях нет — просто скажи об этом одним "
    "предложением без каких-либо id. "
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

async def _can_manage_search(update, context) -> bool:
    """Allow private-chat users and group administrators/owners."""
    msg = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if not msg or not chat:
        return False
    if chat.type == "private":
        return True
    if not user:
        await msg.reply_text("Не удалось определить пользователя.")
        return False
    try:
        member = await context.bot.get_chat_member(chat.id, user.id)
    except Exception as exc:
        logger.warning("get_chat_member failed: %s", exc)
        await msg.reply_text("Не удалось проверить права. Попробуй позже.")
        return False
    if member.status not in ("administrator", "creator"):
        await msg.reply_text("Эта команда доступна только администраторам чата.")
        return False
    return True


async def _set_search_enabled(update, context, enabled: bool) -> None:
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
    if not await _can_manage_search(update, context):
        return

    try:
        await recap_db.set_search_enabled(chat.id, enabled)
    except Exception as exc:
        logger.exception("Failed to update search setting for chat_id=%s: %s", chat.id, exc)
        await msg.reply_text("Не удалось сохранить настройку поиска. Попробуй позже.")
        return
    if enabled:
        await msg.reply_text(
            "Индексация и поиск включены. Новые сообщения этого чата будут "
            "сохраняться; поиск доступен через /search."
        )
    else:
        await msg.reply_text(
            "Индексация и поиск выключены. Новые сообщения не сохраняются. "
            "Уже сохранённая история не удалена."
        )


async def cmd_search_on(update, context) -> None:
    """Enable persistent history indexing and search for the current chat."""
    await _set_search_enabled(update, context, True)


async def cmd_search_off(update, context) -> None:
    """Disable persistent history indexing and search for the current chat."""
    await _set_search_enabled(update, context, False)

def extract_search_query(text: str) -> Optional[str]:
    """Return the query from any supported search command spelling."""
    match = _SEARCH_QUERY_PATTERN.match(text.strip())
    if not match:
        return None
    return (match.group("query") or "").strip()


async def cmd_search(update, context) -> None:
    """Handle /search, /s, /п and ? search requests."""
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

    if not await recap_db.is_search_enabled(chat.id):
        await msg.reply_text(
            "Поиск и индексация в этом чате выключены. "
            "Администратор может включить их командой /search_on."
        )
        return

    message_text = getattr(msg, "text", None)
    query_text = (
        extract_search_query(message_text)
        if isinstance(message_text, str)
        else None
    )
    if query_text is None:
        query_text = " ".join(getattr(context, "args", None) or []).strip()

    if not query_text:
        await msg.reply_text(
            "Использование: /search <запрос> (также /s, /п и ?)"
        )
        return
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
        canonical_id_by_message_id: dict = {}
        for m in messages:
            canonical = m.get("media_source_message_id") or m.get("message_id")
            valid_ids.add(canonical)
            canonical_id_by_message_id[m.get("message_id")] = canonical

        # 6. Generate grounded answer
        try:
            raw_answer = await asyncio.to_thread(
                _generate_answer_sync, messages, norm_query
            )
        except Exception as exc:
            logger.exception("Answer generation failed: %s", exc)
            await processing_msg.edit_text("Не удалось сформировать ответ.")
            return

        # 6b. Guarantee at least one citation, using the chunk's LLM-flagged
        # important messages first, falling back to the discussion's first
        # message, so the answer never ends up without a source link.
        fallback_ids: List[int] = [
            canonical_id_by_message_id[m]
            for m in (best.get("important_message_ids") or [])
            if m in canonical_id_by_message_id
        ]
        first_mid_fallback = canonical_id_by_message_id.get(best.get("first_message_id"))
        if first_mid_fallback and first_mid_fallback not in fallback_ids:
            fallback_ids.append(first_mid_fallback)
        raw_answer = ensure_citation(raw_answer, fallback_ids)

        # 7. Reply-to candidates: the ids the answer actually cites (most
        # relevant), then the fallback ids, then the chunk's first message,
        # then any other message in the chunk — tried in this order below
        # since imported/historical messages can be missing from the live
        # chat (deleted, or the chat never actually reaches back that far).
        cited_ids = extract_cited_ids(raw_answer, valid_ids)
        first_mid_canonical = canonical_id_by_message_id.get(best.get("first_message_id"))
        reply_candidates: List[int] = []
        seen_candidates: set = set()

        def _add_candidate(mid: Optional[int]) -> None:
            if mid and mid not in seen_candidates:
                seen_candidates.add(mid)
                reply_candidates.append(mid)

        for mid in cited_ids:
            _add_candidate(mid)
        for mid in fallback_ids:
            _add_candidate(mid)
        _add_candidate(first_mid_canonical)
        for m in messages:
            _add_candidate(m.get("media_source_message_id") or m.get("message_id"))

        # 8. Convert [id=N] markers to validated HTML links (or plain-text
        # "(#N)" references when this chat has no stable permalink format).
        answer_html = convert_id_markers_to_links(raw_answer, prefix, valid_ids)
        answer_html = answer_html.strip() or "Ответ не сформирован."

        # 9. Delete "Ищу…" message and send the answer
        try:
            await processing_msg.delete()
        except Exception:
            pass

        send_kwargs = dict(
            chat_id=chat_id,
            text=answer_html,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

        sent = False
        for candidate in reply_candidates[:6]:
            try:
                await context.bot.send_message(reply_to_message_id=candidate, **send_kwargs)
                sent = True
                break
            except Exception as exc:
                logger.debug(
                    "Reply to message_id=%s failed (%s); trying next candidate",
                    candidate, exc,
                )

        if not sent:
            if reply_candidates:
                logger.warning(
                    "Could not reply to any of %s candidate messages for chat_id=%s; "
                    "sending without reply",
                    len(reply_candidates[:6]), chat_id,
                )
            await context.bot.send_message(**send_kwargs)

    except Exception as exc:
        logger.exception("Search error: %s", exc)
        try:
            await processing_msg.edit_text("Произошла ошибка при поиске.")
        except Exception:
            pass
