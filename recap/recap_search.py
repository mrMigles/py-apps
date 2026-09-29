# recap_search.py
# -*- coding: utf-8 -*-
"""
/search command for the recap bot.

Flow:
  1. Guard: persistence and per-chat search must be enabled.
  2. LLM normalises the natural-language query to JSON filters
     (query, keywords, date_from, date_to, participants, exact_terms).
  3. Full-text search over individual messages (messages.tsv): keywords are
     sanitised into prefix terms and OR-ed together, so a message does not
     have to contain every word of the question. Search commands and bot
     messages are excluded — they only echo the question back.
  4. Hits are clustered into discussions (hits close in time). A discussion
     with several authors and good keyword coverage outranks a lone question
     that merely mentions the same words.
  5. Semantic fallback: nearest indexed chunks by embedding fill the list up
     when full-text search finds few discussions.
  6. One result at a time: the discussion window around the hits is loaded,
     the LLM writes a short grounded answer using [id=N] markers, and an
     excerpt of key messages with links is appended.
  7. The result carries inline buttons «OK» (removes the buttons) and «Ещё»
     (edits the same message with the next discussion). Only the user who
     started the search may press them.

Pure helpers (parse_search_filters, build_term_queries, cluster_hits,
build_excerpt, convert_id_markers_to_links, …) have no I/O and can be
unit-tested without a database or LLM.
"""

import asyncio
import html
import json
import logging
import os
import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

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

    keywords = data.get("keywords")
    if isinstance(keywords, list):
        result["keywords"] = [str(k)[:100] for k in keywords if k][:10]

    return result


# Words that carry no topic: question words, "search verbs" and the most
# common function words. Used for the no-LLM keyword fallback and to keep
# them out of full-text terms (they would match nearly every message).
_STOPWORDS = frozenset("""
а без бы был была были было в вам вас ведь во вот все всё вы где да даже для
до его ее её если есть еще ещё же за и из или им их к как какая какие какой
когда кого кто ли либо мне мы на над не нет ни но ну о об обо он она они оно от
по под при про с со так там тебе то тоже только ты у уже чем что чтобы эта эти
это этот я
зачем почему сколько откуда куда чей чья чьё чьи каким какую каком
обсуждали обсуждал обсуждала обсуждение говорили говорил говорила писали писал
писала упоминали упоминал спрашивал спрашивали речь было найди найти покажи
вспомни напомни чате чат
""".split())

_QUESTION_WORDS = frozenset(
    "кто что где когда как почему зачем какой какая какие какое сколько "
    "куда откуда чей чья чьё чьи ли подскажите подскажи кто-нибудь "
    "кто-то посоветуйте посоветуй".split()
)

_WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*")


def extract_keywords(query: str, limit: int = 8) -> List[str]:
    """No-LLM fallback: the query's words minus stopwords/question words."""
    words: List[str] = []
    for w in _WORD_RE.findall(query.lower()):
        if len(w) < 2 or w in _STOPWORDS or w in words:
            continue
        words.append(w)
    return words[:limit]


def build_term_queries(terms: List[str], limit: int = 10) -> List[str]:
    """
    Turn keywords/phrases into safe to_tsquery() fragments.

    Each term is reduced to plain word tokens (letters/digits only, so no
    tsquery operator or quote can ever reach PostgreSQL). Words of 3+ chars
    become prefix terms ("поездку" → "поездку:*", stemmed by PostgreSQL to
    'поездк':* and matching every form); a multi-word phrase becomes an AND
    group. Duplicates and stopword-only terms are dropped.
    """
    fragments: List[str] = []
    for term in terms:
        tokens = [
            t for t in re.findall(r"[^\W_]+", str(term).lower())
            if len(t) >= 2 and t not in _STOPWORDS
        ]
        if not tokens:
            continue
        frag = " & ".join(f"{t}:*" if len(t) >= 3 else t for t in tokens)
        if frag not in fragments:
            fragments.append(frag)
    return fragments[:limit]


def build_or_tsquery(fragments: List[str]) -> str:
    """OR the fragments together: a message needs only one of them to match."""
    return " | ".join(f"({f})" for f in fragments)


def is_search_command_text(text: Optional[str]) -> bool:
    """True for bot commands and search requests ("/…", "? …")."""
    s = (text or "").lstrip()
    return s.startswith("/") or s.startswith("?")


def looks_like_question(text: Optional[str]) -> bool:
    """Heuristic: the message asks something rather than discusses it."""
    s = (text or "").strip().lower()
    if not s:
        return False
    if s.rstrip(")( .!").endswith("?"):
        return True
    first = _WORD_RE.match(s)
    return bool(first and first.group(0) in _QUESTION_WORDS)


def _canonical_id(m: dict) -> int:
    return m.get("media_source_message_id") or m.get("message_id")


@dataclass
class Candidate:
    """One found discussion: a time range plus the messages that matched."""
    start: datetime
    end: datetime
    score: float = 0.0
    hits: List[dict] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    source: str = "fts"  # "fts" | "vector"

    @property
    def hit_ids(self) -> List[int]:
        """Hit ids, best first: discussion messages before questions, then rank."""
        ordered = sorted(
            self.hits,
            key=lambda h: (looks_like_question(h.get("text")), -(h.get("rank") or 0.0)),
        )
        return [_canonical_id(h) for h in ordered]


QUESTION_PENALTY = 0.3


def cluster_hits(
    hits: List[dict],
    n_terms: int,
    gap: timedelta = timedelta(minutes=45),
    participants: Optional[List[str]] = None,
) -> List[Candidate]:
    """
    Group full-text hits into discussions and rank them.

    Hits closer than *gap* to the previous hit belong to the same discussion.
    Score = Σ rank (question-like hits count ×0.3)
            × keyword coverage (distinct matched terms / n_terms)
            × (1 + 0.5 per extra author)
            × 0.3 if every hit is a question
            × 1.5 if a requested participant took part.
    So a real discussion outranks a lone "а кто знает про X?" message.
    """
    if not hits:
        return []
    ordered = sorted(hits, key=lambda h: (h["date"], h["message_id"]))
    groups: List[List[dict]] = [[ordered[0]]]
    for h in ordered[1:]:
        if h["date"] - groups[-1][-1]["date"] > gap:
            groups.append([h])
        else:
            groups[-1].append(h)

    wanted = [p.lower() for p in (participants or []) if p]
    candidates: List[Candidate] = []
    for group in groups:
        questions = [looks_like_question(h.get("text")) for h in group]
        rank_sum = sum(
            (h.get("rank") or 0.0) * (QUESTION_PENALTY if q else 1.0)
            for h, q in zip(group, questions)
        )
        matched = set()
        for h in group:
            matched.update(h.get("matched_terms") or [])
        coverage = len(matched) / n_terms if n_terms else 1.0
        coverage = max(coverage, 1.0 / max(n_terms, 1))

        authors: List[str] = []
        for h in group:
            name = h.get("user_name") or str(h.get("user_id") or "?")
            if name not in authors:
                authors.append(name)

        score = rank_sum * coverage * (1.0 + 0.5 * (len(authors) - 1))
        if all(questions):
            score *= QUESTION_PENALTY
        if wanted and any(
            w in a.lower() for a in authors for w in wanted
        ):
            score *= 1.5

        candidates.append(Candidate(
            start=group[0]["date"],
            end=group[-1]["date"],
            score=score,
            hits=group,
            authors=authors,
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _message_ref(mid: int, link_prefix: Optional[str]) -> str:
    if not link_prefix:
        return f"(#{mid})"
    return f'<a href="{link_prefix}{mid}">#{mid}</a>'


def build_excerpt(
    window: List[dict],
    hit_ids: List[int],
    cited_ids: List[int],
    link_prefix: Optional[str],
    n: int = 4,
    max_len: int = 200,
) -> str:
    """
    HTML excerpt of the key messages of a discussion, in chronological order.

    Picks cited messages first, then hits (discussion before questions, as
    ordered by the caller), then — if still short — other window messages
    after the first hit, i.e. the likely answers.
    """
    by_id = {_canonical_id(m): m for m in window}
    picked: List[int] = []
    for mid in list(cited_ids) + list(hit_ids):
        if mid in by_id and mid not in picked:
            picked.append(mid)
        if len(picked) >= n:
            break
    if len(picked) < n and picked:
        first = min(picked)
        for m in window:
            mid = _canonical_id(m)
            if mid > first and mid not in picked:
                picked.append(mid)
                if len(picked) >= n:
                    break
    picked = picked[:n]

    lines: List[str] = []
    for m in window:
        mid = _canonical_id(m)
        if mid not in picked:
            continue
        text = " ".join((m.get("text") or "").split())
        if len(text) > max_len:
            text = text[: max_len - 1] + "…"
        name = html.escape(m.get("user_name") or "?", quote=False)
        lines.append(
            f"▫️ <b>{name}</b>: {html.escape(text, quote=False)} "
            f"{_message_ref(mid, link_prefix)}"
        )
    return "\n".join(lines)


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
        return _message_ref(mid, link_prefix)

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

_NORMALISE_USER = """Нормализуй запрос и извлеки фильтры. Верни JSON:
{{
  "query": "нормализованный запрос по-русски (обязательно)",
  "keywords": ["3–8 значимых слов или коротких фраз по теме, включая синонимы и другие написания (рус./англ.); БЕЗ вопросительных и служебных слов вроде «когда», «где», «кто», «обсуждали», «говорили»"],
  "date_from": "YYYY-MM-DD или null",
  "date_to":   "YYYY-MM-DD или null",
  "participants": ["имя1", "имя2"] или null,
  "exact_terms": ["точная фраза"] или null
}}

Сегодня: {today}
Запрос пользователя: {query}"""


def _normalise_query_sync(query_text: str) -> str:
    resp = _get_client().chat.completions.create(
        model=RECAP_MODEL,
        messages=[
            {"role": "system", "content": _NORMALISE_SYSTEM},
            {"role": "user", "content": _NORMALISE_USER.format(
                query=query_text,
                today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            )},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return (resp.choices[0].message.content or "").strip()


def _embed_sync(text: str) -> List[float]:
    resp = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


_ANSWER_SYSTEM = (
    "Ты отвечаешь на вопросы по истории Telegram-чата. "
    "Тебе дан фрагмент обсуждения, найденный поиском. "
    "Ответ — по-русски, 1–3 коротких предложения, по делу. "
    "Опирайся на сообщения, где тему ОБСУЖДАЛИ или ДАЛИ ОТВЕТ, а не на "
    "сообщения, где вопрос только задали. "
    "Ссылайся на источники, вставляя маркер [id=N] сразу после соответствующего "
    "факта, где N — id одного сообщения. "
    "НЕ перечисляй id списком и НЕ выводи id, если по теме ничего не нашлось. "
    "Если во фрагменте тему только упомянули или спросили, но ответа нет — "
    "так и скажи одним предложением. "
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
# Search on/off commands
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


# ---------------------------------------------------------------------------
# Retrieval and rendering
# ---------------------------------------------------------------------------

MAX_RESULTS = 10
MAX_ANSWER_CHARS = 1800


async def find_discussions(
    chat_id: int,
    filters: dict,
    query_text: str,
    bot_id: Optional[int] = None,
) -> List[Candidate]:
    """
    Full-text search over messages → discussion clusters, topped up with
    semantic chunk matches that do not overlap an already found discussion.
    """
    norm_query = filters.get("query") or query_text
    terms = list(filters.get("keywords") or []) + list(filters.get("exact_terms") or [])
    fragments = build_term_queries(terms)
    if not fragments:
        fragments = build_term_queries(
            extract_keywords(norm_query) or extract_keywords(query_text)
        )

    date_from: Optional[datetime] = filters.get("date_from")
    date_to: Optional[datetime] = filters.get("date_to")

    candidates: List[Candidate] = []
    if fragments:
        hits = await recap_db.search_messages(
            chat_id=chat_id,
            tsquery=build_or_tsquery(fragments),
            term_queries=fragments,
            date_from=date_from,
            date_to=date_to,
            exclude_user_id=bot_id,
        )
        candidates = cluster_hits(
            hits, len(fragments), participants=filters.get("participants"),
        )[:MAX_RESULTS]

    if len(candidates) < MAX_RESULTS:
        try:
            embedding = await asyncio.to_thread(_embed_sync, norm_query)
            chunks = await recap_db.vector_search_chunks(
                chat_id, embedding, date_from=date_from, date_to=date_to,
                limit=MAX_RESULTS,
            )
        except Exception as exc:
            logger.warning("Semantic fallback failed (%s); full-text results only", exc)
            chunks = []
        for ch in chunks:
            start, end = ch.get("start_date"), ch.get("end_date")
            if not start or not end:
                continue
            if any(start <= c.end and end >= c.start for c in candidates):
                continue
            candidates.append(Candidate(start=start, end=end, source="vector"))
            if len(candidates) >= MAX_RESULTS:
                break

    return candidates


async def render_candidate(
    chat_id: int,
    candidate: Candidate,
    query_text: str,
    link_prefix: Optional[str],
    position: int,
    total: int,
    bot_id: Optional[int] = None,
) -> str:
    """Load the discussion window and build the HTML text of one result."""
    window = await recap_db.get_context_window(
        chat_id, candidate.start, candidate.end, exclude_user_id=bot_id
    )
    if not window:
        return f"🔎 Результат {position}/{total}: не удалось загрузить сообщения."

    valid_ids = {_canonical_id(m) for m in window}
    hit_ids = [mid for mid in candidate.hit_ids if mid in valid_ids]
    if not hit_ids:
        # Semantic candidates have no hits: prefer the window's non-questions.
        hit_ids = [
            _canonical_id(m) for m in sorted(
                window, key=lambda m: looks_like_question(m.get("text"))
            )
        ]

    try:
        raw_answer = await asyncio.to_thread(_generate_answer_sync, window, query_text)
    except Exception as exc:
        logger.warning("Answer generation failed (%s); showing excerpt only", exc)
        raw_answer = ""
    raw_answer = (raw_answer or "")[:MAX_ANSWER_CHARS]
    if raw_answer:
        raw_answer = ensure_citation(raw_answer, hit_ids)
    cited_ids = extract_cited_ids(raw_answer, valid_ids)
    answer_html = convert_id_markers_to_links(
        html.escape(raw_answer, quote=False), link_prefix, valid_ids
    )

    authors = list(candidate.authors)
    if not authors:
        for m in window:
            name = m.get("user_name") or "?"
            if name not in authors:
                authors.append(name)
    header = (
        f"🔎 <b>Результат {position}/{total}</b> · "
        f"{candidate.start.strftime('%d.%m.%Y')} · "
        f"{html.escape(', '.join(authors[:3]), quote=False)}"
    )
    parts = [header]
    if answer_html:
        parts.append(answer_html)
    excerpt = build_excerpt(window, hit_ids, cited_ids, link_prefix)
    if excerpt:
        parts.append(excerpt)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Result sessions (pagination state behind the inline buttons)
# ---------------------------------------------------------------------------

SESSION_TTL_SECONDS = 24 * 3600
MAX_SESSIONS = 200
CALLBACK_PREFIX = "srch"


@dataclass
class SearchSession:
    token: str
    chat_id: int
    user_id: Optional[int]
    query: str
    link_prefix: Optional[str]
    candidates: List[Candidate]
    bot_id: Optional[int] = None
    index: int = 0
    pages: Dict[int, str] = field(default_factory=dict)
    busy: bool = False
    created: float = field(default_factory=time.monotonic)


# In memory only: after a restart the buttons answer «Поиск устарел».
_SESSIONS: "OrderedDict[str, SearchSession]" = OrderedDict()


def _new_session(**kwargs) -> SearchSession:
    now = time.monotonic()
    for token in [t for t, s in _SESSIONS.items() if now - s.created > SESSION_TTL_SECONDS]:
        _SESSIONS.pop(token, None)
    while len(_SESSIONS) >= MAX_SESSIONS:
        _SESSIONS.popitem(last=False)
    token = secrets.token_urlsafe(6)
    while token in _SESSIONS:
        token = secrets.token_urlsafe(6)
    session = SearchSession(token=token, **kwargs)
    _SESSIONS[token] = session
    return session


def _get_session(token: str) -> Optional[SearchSession]:
    session = _SESSIONS.get(token)
    if session and time.monotonic() - session.created > SESSION_TTL_SECONDS:
        _SESSIONS.pop(token, None)
        return None
    return session


def _keyboard(session: SearchSession):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    buttons = [InlineKeyboardButton(
        "👌 OK", callback_data=f"{CALLBACK_PREFIX}:{session.token}:ok"
    )]
    if session.index < len(session.candidates) - 1:
        buttons.append(InlineKeyboardButton(
            "➡️ Ещё", callback_data=f"{CALLBACK_PREFIX}:{session.token}:more"
        ))
    return InlineKeyboardMarkup([buttons])


async def _render_page(session: SearchSession, index: int) -> str:
    if index not in session.pages:
        session.pages[index] = await render_candidate(
            session.chat_id,
            session.candidates[index],
            session.query,
            session.link_prefix,
            index + 1,
            len(session.candidates),
            bot_id=session.bot_id,
        )
    return session.pages[index]


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

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
    prefix = _link_prefix(chat_id, getattr(chat, "username", None))
    user_id = getattr(getattr(update, "effective_user", None), "id", None)
    bot_id = getattr(context.bot, "id", None)
    bot_id = bot_id if isinstance(bot_id, int) else None

    processing_msg = await msg.reply_text("Ищу…")

    try:
        try:
            raw_filters = await asyncio.to_thread(_normalise_query_sync, query_text)
            filters = parse_search_filters(raw_filters)
        except Exception as exc:
            logger.warning("Query normalisation failed (%s), using raw query", exc)
            filters = {"query": query_text}

        candidates = await find_discussions(
            chat_id, filters, query_text,
            bot_id=bot_id,
        )
        if not candidates:
            await processing_msg.edit_text("По этому запросу ничего не найдено.")
            return

        session = _new_session(
            chat_id=chat_id,
            user_id=user_id if isinstance(user_id, int) else None,
            query=query_text,
            link_prefix=prefix,
            candidates=candidates,
            bot_id=bot_id,
        )
        text = await _render_page(session, 0)
        await processing_msg.edit_text(
            text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=_keyboard(session),
        )

    except Exception as exc:
        logger.exception("Search error: %s", exc)
        try:
            await processing_msg.edit_text("Произошла ошибка при поиске.")
        except Exception:
            pass


async def on_search_callback(update, context) -> None:
    """Handle the «OK» / «Ещё» buttons under a search result."""
    from telegram.constants import ParseMode

    query = update.callback_query
    if not query:
        return
    parts = (query.data or "").split(":")
    if len(parts) != 3 or parts[0] != CALLBACK_PREFIX:
        await query.answer()
        return
    _, token, action = parts

    session = _get_session(token)
    message = getattr(query, "message", None)
    if session is None or (
        message is not None and getattr(message, "chat_id", None) != session.chat_id
    ):
        await query.answer("Поиск устарел, повторите запрос.", show_alert=True)
        return

    from_user = getattr(query, "from_user", None)
    if session.user_id is not None and getattr(from_user, "id", None) != session.user_id:
        await query.answer("Кнопки доступны только автору поиска.", show_alert=True)
        return

    if action == "ok":
        await query.answer()
        _SESSIONS.pop(token, None)
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception as exc:
            logger.debug("Could not remove search keyboard: %s", exc)
        return

    if action != "more":
        await query.answer()
        return

    if session.busy:
        await query.answer("Ищу, подождите…")
        return
    if session.index >= len(session.candidates) - 1:
        await query.answer("Больше результатов нет.")
        return

    session.busy = True
    try:
        await query.answer()
        next_index = session.index + 1
        text = await _render_page(session, next_index)
        session.index = next_index
        await query.edit_message_text(
            text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            reply_markup=_keyboard(session),
        )
    except Exception as exc:
        logger.exception("Search pagination error: %s", exc)
    finally:
        session.busy = False
