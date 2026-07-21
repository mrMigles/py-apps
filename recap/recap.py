# recap_bot.py
# -*- coding: utf-8 -*-
"""
Telegram recap-bot: summarizes chat discussion for the last 24 hours
starting from a specific message (/recap as reply).

Now:
- Recap is a lively 1–2 paragraph story (not bullet chapters)
- Links are embedded inline inside the story (HTML <a href="...">)
- Model is allowed to generate full links, BUT we validate them:
  - href must start with our computed LINK_PREFIX
  - message_id must exist in the provided messages set
  - otherwise link is stripped to plain text
- Forwarded messages are explicitly marked in model input (FORWARDED from ...)
- Persistent semantic search via PostgreSQL/pgvector (optional)

Env:
- TELEGRAM_BOT_TOKEN
- OPENAI_TOKEN
- RECAP_OPENAI_MODEL (optional, default: google/gemini-2.5-flash-lite)
- RECAP_TRANSCRIPTION_MODEL (optional, default: openai/whisper-large-v3)
- RECAP_TRANSCRIPTION_LANGUAGE (optional, e.g. ru)
- RECAP_IMAGE_MODEL_BIG (optional, default: google/gemini-3.1-flash-lite) — for single images
- RECAP_IMAGE_MODEL_SIMPLE (optional, default: google/gemini-2.5-flash-lite) — for image albums (2–3 photos)
- PG_DATABASE, PG_USERNAME, PG_PASSWORD, PG_HOST, PG_PORT — PostgreSQL (optional)
- RECAP_EMBEDDING_MODEL (optional, default: openai/text-embedding-3-small)
- RECAP_EMBEDDING_DIM (optional, default: 1536)
- RECAP_INDEX_INTERVAL_SECONDS (optional, default: 120)
- RECAP_INDEX_BATCH (optional, default: 250)
"""

import asyncio
import base64
import html
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from telegram import Message, Update
from telegram.constants import ParseMode
from telegram.ext import (
  ApplicationBuilder,
  CommandHandler,
  ContextTypes,
  MessageHandler,
  filters,
)

import recap_db
import recap_index
import recap_search
import recap_import

# ==========================
# Logging
# ==========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("recap-bot")

logging.getLogger("telegram").setLevel(logging.INFO)
logging.getLogger("telegram.ext").setLevel(logging.INFO)

# ==========================
# Configuration and globals
# ==========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
RECAP_MODEL = os.getenv("RECAP_OPENAI_MODEL", "google/gemini-2.5-flash-lite")
RECAP_LLM_URL = os.getenv("RECAP_LLM_URL", "https://api.openai.com/v1")
TRANSCRIPTION_MODEL = os.getenv("RECAP_TRANSCRIPTION_MODEL", "openai/whisper-large-v3")
TRANSCRIPTION_LANGUAGE = os.getenv("RECAP_TRANSCRIPTION_LANGUAGE")
IMAGE_MODEL_BIG = os.getenv("RECAP_IMAGE_MODEL_BIG", "google/gemini-3.1-flash-lite")
IMAGE_MODEL_SIMPLE = os.getenv("RECAP_IMAGE_MODEL_SIMPLE", "google/gemini-2.5-flash-lite")

HISTORY_RETENTION = timedelta(days=1)
MAX_MESSAGES_FOR_SUMMARY = 250
MAX_TRANSCRIPT_REPLY_LENGTH = 3600
MAX_IMAGES_PER_MESSAGE = 3
MEDIA_GROUP_DEBOUNCE_SECONDS = 2.0

IMAGE_PROMPT = (
  "Кратко опиши изображение или набор изображений на русском языке. "
  "Передай только суть: что изображено и в чём основной смысл. "
  "Для нескольких изображений кратко опиши каждое и затем укажи их общую связь. "
  "Не переписывай текст целиком, не перечисляй интерфейсные элементы, метрики и незначительные детали. "
  "Для мема или шутки кратко объясни смысл без подробного разбора. "
  "Если есть подпись автора — используй её как контекст, но не повторяй дословно. "
  "Ответ — один короткий абзац, не более 2 предложений и 250 символов. "
  "Не используй заголовки и списки. Не выдумывай детали."
)

client: Optional[OpenAI] = None

history_lock = asyncio.Lock()
chat_history: Dict[int, List["ChatMessage"]] = {}

media_group_lock = asyncio.Lock()
media_group_buffer: Dict[str, List] = {}
media_group_tasks: Dict[str, asyncio.Task] = {}

# ==========================
# Data model
# ==========================


@dataclass
class ChatMessage:
  chat_id: int
  chat_username: Optional[str]
  message_id: int
  user_id: Optional[int]
  user_name: str
  text: str
  content_kind: str
  date: datetime
  is_bot: bool
  reply_to_message_id: Optional[int]
  reply_to_user_name: Optional[str]
  reply_to_text: Optional[str]

  # Forward metadata
  is_forwarded: bool
  forward_from_name: Optional[str]
  forward_from_chat_title: Optional[str]
  forward_from_chat_username: Optional[str]
  forward_from_message_id: Optional[int]

  # For voice/video-note transcripts: the original media message id.
  # In-memory /recap ignores this field.
  # In DB the row is stored under this id so links point at the media message.
  media_source_message_id: Optional[int] = field(default=None)


# ==========================
# Helpers
# ==========================


def _cm_to_db_row(cm: "ChatMessage") -> dict:
  """
  Convert a ChatMessage to the dict shape expected by recap_db.upsert_message.
  For transcripts, the DB row is stored under media_source_message_id so that
  Telegram links point at the original media message.
  """
  db_message_id = cm.media_source_message_id if cm.media_source_message_id else cm.message_id
  return {
      "chat_id": cm.chat_id,
      "message_id": db_message_id,
      "chat_username": cm.chat_username,
      "user_id": cm.user_id,
      "user_name": cm.user_name,
      "text": cm.text,
      "content_kind": cm.content_kind,
      "date": cm.date,
      "is_bot": cm.is_bot,
      "reply_to_message_id": cm.reply_to_message_id,
      "reply_to_user_name": cm.reply_to_user_name,
      "reply_to_text": cm.reply_to_text,
      "is_forwarded": cm.is_forwarded,
      "forward_from_name": cm.forward_from_name,
      "forward_from_chat_title": cm.forward_from_chat_title,
      "forward_from_chat_username": cm.forward_from_chat_username,
      "forward_from_message_id": cm.forward_from_message_id,
      "media_source_message_id": cm.media_source_message_id,
  }


def _get_openai_client() -> OpenAI:
  """Create the API client lazily so utilities and tests can import the module."""
  global client
  if client is None:
    if not OPENAI_TOKEN:
      raise RuntimeError("OPENAI_TOKEN is not set in environment.")
    client = OpenAI(api_key=OPENAI_TOKEN, base_url=RECAP_LLM_URL)
  return client


def _safe_trim(s: str, limit: int) -> str:
  s = (s or "").strip()
  if limit <= 0:
    return ""
  if len(s) <= limit:
    return s
  if limit == 1:
    return "…"
  return s[: limit - 1] + "…"


def _display_name_from_message(msg: Message) -> str:
  u = msg.from_user
  if not u:
    return "Unknown"
  if u.full_name:
    return u.full_name
  if u.username:
    return u.username
  return str(u.id)


def _link_prefix(chat_id: int, chat_username: Optional[str]) -> Optional[str]:
  """
  Returns message link prefix ending with '/' if possible:
    - Public chats: https://t.me/<username>/
    - Private supergroups/channels: https://t.me/c/<internal_id>/
  Otherwise: None (no stable permalinks for basic groups)
  """
  if chat_username:
    return f"https://t.me/{chat_username}/"

  s = str(chat_id)
  if s.startswith("-100"):
    internal_id = s[4:]
    return f"https://t.me/c/{internal_id}/"

  return None


def _extract_forward_info(msg: Message) -> Tuple[bool, Optional[str], Optional[str], Optional[str], Optional[int]]:
  """
  Returns:
    (is_forwarded, forward_from_name, forward_from_chat_title, forward_from_chat_username, forward_from_message_id)
  """
  fwd_from_user = getattr(msg, "forward_from", None)
  fwd_from_chat = getattr(msg, "forward_from_chat", None)
  fwd_sender_name = getattr(msg, "forward_sender_name", None)
  fwd_msg_id = getattr(msg, "forward_from_message_id", None)
  fwd_origin = getattr(msg, "forward_origin", None)

  is_forwarded = bool(
      getattr(msg, "forward_date", None)
      or fwd_from_user
      or fwd_from_chat
      or fwd_sender_name
      or fwd_origin
  )

  forward_from_name: Optional[str] = None
  forward_from_chat_title: Optional[str] = None
  forward_from_chat_username: Optional[str] = None

  if fwd_from_user:
    forward_from_name = (
        getattr(fwd_from_user, "full_name", None)
        or getattr(fwd_from_user, "username", None)
        or str(getattr(fwd_from_user, "id", ""))
    )
  elif fwd_sender_name:
    forward_from_name = fwd_sender_name
  elif fwd_origin:
    sender_user = getattr(fwd_origin, "sender_user", None)
    if sender_user:
      forward_from_name = (
          getattr(sender_user, "full_name", None)
          or getattr(sender_user, "username", None)
          or str(getattr(sender_user, "id", ""))
      )
    else:
      forward_from_name = (
          getattr(fwd_origin, "sender_user_name", None)
          or getattr(fwd_origin, "sender_name", None)
      )

    origin_chat = getattr(fwd_origin, "chat", None) or getattr(fwd_origin, "sender_chat", None)
    if origin_chat:
      forward_from_chat_title = getattr(origin_chat, "title", None)
      forward_from_chat_username = getattr(origin_chat, "username", None)

  if fwd_from_chat:
    forward_from_chat_title = getattr(fwd_from_chat, "title", None)
    forward_from_chat_username = getattr(fwd_from_chat, "username", None)
    if not forward_from_name:
      forward_from_name = (
          forward_from_chat_title
          or forward_from_chat_username
          or str(getattr(fwd_from_chat, "id", ""))
      )

  return is_forwarded, forward_from_name, forward_from_chat_title, forward_from_chat_username, fwd_msg_id


def _media_kind_label(content_kind: str) -> Optional[str]:
  if content_kind == "voice_transcript":
    return "voice transcript"
  if content_kind == "video_note_transcript":
    return "video note transcript"
  if content_kind == "image_description":
    return "image description"
  return None


async def _add_to_history(
    msg: Message,
    text_override: Optional[str] = None,
    content_kind: str = "text",
    author_msg: Optional[Message] = None,
    media_source_message_id: Optional[int] = None,
) -> None:
  """Store one message in global in-memory history (and enqueue to DB if enabled)."""
  if not msg or not msg.chat:
    logger.info("Skip message without chat: %s", msg)
    return

  speaker_msg = author_msg or msg
  message_text = text_override if text_override is not None else (msg.text or msg.caption or "")
  message_text = (message_text or "").strip()

  if not message_text:
    logger.info("Skip message without text/caption or chat: %s", msg)
    return

  chat_id = msg.chat_id
  chat_username = getattr(msg.chat, "username", None)

  now = datetime.now(timezone.utc)
  msg_date = msg.date or now
  if msg_date.tzinfo is None:
    msg_date = msg_date.replace(tzinfo=timezone.utc)

  reply = msg.reply_to_message
  if reply is not None:
    reply_name = _display_name_from_message(reply)
    reply_text = reply.text or reply.caption or ""
    reply_id = reply.message_id
  else:
    reply_name = None
    reply_text = None
    reply_id = None

  is_fwd, fwd_from_name, fwd_chat_title, fwd_chat_username, fwd_msg_id = _extract_forward_info(speaker_msg)

  cm = ChatMessage(
      chat_id=chat_id,
      chat_username=chat_username,
      message_id=msg.message_id,
      user_id=speaker_msg.from_user.id if speaker_msg.from_user else None,
      user_name=_display_name_from_message(speaker_msg),
      text=message_text,
      content_kind=content_kind,
      date=msg_date,
      is_bot=bool(speaker_msg.from_user and speaker_msg.from_user.is_bot),
      reply_to_message_id=reply_id,
      reply_to_user_name=reply_name,
      reply_to_text=reply_text,
      is_forwarded=is_fwd,
      forward_from_name=fwd_from_name,
      forward_from_chat_title=fwd_chat_title,
      forward_from_chat_username=fwd_chat_username,
      forward_from_message_id=fwd_msg_id,
      media_source_message_id=media_source_message_id,
  )

  async with history_lock:
    messages = chat_history.setdefault(chat_id, [])
    messages.append(cm)

    cutoff = now - HISTORY_RETENTION
    before_len = len(messages)
    messages = [m for m in messages if m.date >= cutoff]
    messages.sort(key=lambda m: (m.date, m.message_id))
    chat_history[chat_id] = messages

  logger.info(
      "Stored message chat_id=%s msg_id=%s by=%s; history size: %s (removed %s old)",
      chat_id,
      cm.message_id,
      cm.user_name,
      len(chat_history[chat_id]),
      before_len - len(chat_history[chat_id]),
      )

  # Enqueue for persistent storage (non-blocking; no effect if PG not configured)
  if recap_db.is_enabled():
    q = recap_db.get_write_queue()
    if q is not None:
      try:
        q.put_nowait(_cm_to_db_row(cm))
      except asyncio.QueueFull:
        logger.warning("DB write queue is full; dropping message chat_id=%s msg_id=%s", chat_id, cm.message_id)
      except Exception as exc:
        logger.exception("Failed to enqueue message for DB: %s", exc)


def _telegram_media_suffix(msg: Message) -> str:
  media = msg.voice or msg.video_note
  mime_type = getattr(media, "mime_type", None)
  if mime_type == "audio/ogg":
    return ".ogg"
  if mime_type == "audio/mpeg":
    return ".mp3"
  if mime_type == "audio/wav":
    return ".wav"
  if mime_type == "video/mp4" or msg.video_note:
    return ".mp4"
  if msg.voice:
    return ".ogg"
  return ".bin"


async def _download_telegram_media_to_temp(msg: Message) -> str:
  media = msg.voice or msg.video_note
  if not media:
    raise ValueError("Message does not contain voice or video_note media.")

  tg_file = await media.get_file()

  fd, temp_path = tempfile.mkstemp(suffix=_telegram_media_suffix(msg), prefix="recap-audio-")
  os.close(fd)

  try:
    await tg_file.download_to_drive(custom_path=temp_path)
    return temp_path
  except Exception:
    try:
      os.remove(temp_path)
    except OSError:
      pass
    raise


def _transcribe_audio_file(file_path: str) -> str:
  kwargs = {"model": TRANSCRIPTION_MODEL}
  if TRANSCRIPTION_LANGUAGE:
    kwargs["language"] = TRANSCRIPTION_LANGUAGE

  with open(file_path, "rb") as audio_file:
    resp = _get_openai_client().audio.transcriptions.create(file=audio_file, **kwargs)

  if isinstance(resp, str):
    return resp.strip()
  if isinstance(resp, dict):
    return str(resp.get("text", "")).strip()
  return str(getattr(resp, "text", "") or "").strip()


async def _transcribe_telegram_media(msg: Message) -> str:
  temp_path = await _download_telegram_media_to_temp(msg)
  try:
    logger.info(
        "Transcribing media chat_id=%s msg_id=%s model=%s",
        msg.chat_id,
        msg.message_id,
        TRANSCRIPTION_MODEL,
    )
    return await asyncio.to_thread(_transcribe_audio_file, temp_path)
  finally:
    try:
      os.remove(temp_path)
    except OSError:
      logger.warning("Could not remove temp media file: %s", temp_path)


def _transcript_reply_text(msg: Message, transcript: str) -> str:
  return _safe_trim(transcript, MAX_TRANSCRIPT_REPLY_LENGTH)


async def _download_photo_data_uri(msg: Message) -> str:
  """Download the largest photo size from a message and return a base64 data URI."""
  photo = msg.photo[-1]
  tg_file = await photo.get_file()
  data = await tg_file.download_as_bytearray()
  encoded = base64.b64encode(bytes(data)).decode("ascii")
  return f"data:image/jpeg;base64,{encoded}"


def _chat_vision(
    model: str,
    system: str,
    prompt_text: str,
    image_data_uris: List[str],
    temperature: float = 0.3,
    max_tokens: int = 200,
):
  content: List[dict] = [{"type": "text", "text": prompt_text}]
  for uri in image_data_uris:
    content.append({"type": "image_url", "image_url": {"url": uri}})
  return _get_openai_client().chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": content},
      ],
      temperature=temperature,
      max_tokens=max_tokens,
  )


def _describe_images(data_uris: List[str], caption: Optional[str]) -> str:
  model = IMAGE_MODEL_BIG if len(data_uris) == 1 else IMAGE_MODEL_SIMPLE
  user_text = ""
  if caption:
    user_text = f"Подпись автора: {caption}\n\n"
  user_text += "Опиши изображение." if len(data_uris) == 1 else "Опиши набор изображений."

  logger.info("Describing %s image(s) with model=%s", len(data_uris), model)
  resp = _chat_vision(model, IMAGE_PROMPT, user_text, data_uris)
  return _safe_trim((resp.choices[0].message.content or "").strip(), 500)


def _select_slice_for_recap(chat_id: int, from_message_id: Optional[int]) -> List[ChatMessage]:
  """Return messages for recap: last 24h, optionally starting from given message_id."""
  now = datetime.now(timezone.utc)
  cutoff = now - HISTORY_RETENTION

  messages = chat_history.get(chat_id, [])
  messages = [m for m in messages if m.date >= cutoff]

  if from_message_id is not None:
    messages = [m for m in messages if m.message_id >= from_message_id]

  if len(messages) > MAX_MESSAGES_FOR_SUMMARY:
    messages = messages[-MAX_MESSAGES_FOR_SUMMARY:]

  logger.info(
      "Selected %s messages for recap in chat_id=%s starting_from_msg_id=%s",
      len(messages),
      chat_id,
      from_message_id,
  )
  return messages


def _build_conversation_text(messages: List[ChatMessage]) -> str:
  """Build a compact text representation to feed into LLM."""
  lines: List[str] = []
  for m in messages:
    base_text = _safe_trim(m.text, 240)
    user = m.user_name or f"User {m.user_id}"
    media_label = _media_kind_label(m.content_kind)

    if media_label:
      parts = [f"[id={m.message_id}] {user} [{media_label}]: {base_text}"]
    else:
      parts = [f"[id={m.message_id}] {user}: {base_text}"]

    if m.is_forwarded:
      origin = (
          m.forward_from_name
          or m.forward_from_chat_title
          or m.forward_from_chat_username
          or "Unknown"
      )
      parts.append(f"(FORWARDED from {origin})")

    if m.reply_to_message_id:
      parts.append(f"(reply_to_id={m.reply_to_message_id})")

    lines.append(" ".join(parts))
  return "\n".join(lines)


def _chat_text(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.4,
    max_tokens: int = 500,
):
  return _get_openai_client().chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
      ],
      temperature=temperature,
      max_tokens=max_tokens,
  )


def _sanitize_links_in_html(text: str, link_prefix: Optional[str], allowed_message_ids: set[int]) -> str:
  """
  Validate <a href="...">..</a>:
    - href must start with link_prefix
    - last path segment must be int and present in allowed_message_ids
  If invalid -> replace the whole <a ...>inner</a> with inner (no link).
  """
  if not text:
    return ""

  if not link_prefix:
    # no stable permalinks -> strip all anchors
    return re.sub(r"<a\s+[^>]*>(.*?)</a>", r"\1", text, flags=re.IGNORECASE | re.DOTALL)

  # Match anchors with href="..."
  anchor_re = re.compile(
      r'<a\s+[^>]*href\s*=\s*"([^"]+)"[^>]*>(.*?)</a>',
      flags=re.IGNORECASE | re.DOTALL,
  )

  def repl(m: re.Match) -> str:
    href = (m.group(1) or "").strip()
    inner = m.group(2) or ""
    if not href.startswith(link_prefix):
      return inner

    last = href.rstrip("/").split("/")[-1]
    try:
      mid = int(last)
    except Exception:
      return inner

    if mid not in allowed_message_ids:
      return inner

    # keep as-is (but escape href to be safe)
    safe_href = html.escape(href, quote=True)
    return f'<a href="{safe_href}">{inner}</a>'

  return anchor_re.sub(repl, text)


async def _summarize_conversation_narrative(
    messages: List[ChatMessage],
    link_prefix: Optional[str],
) -> str:
  """LLM generates a lively story recap with inline links (HTML)."""
  if not messages:
    return "За выбранный период новых сообщений не найдено."

  conv_text = _build_conversation_text(messages)
  allowed_ids = {m.message_id for m in messages}

  system = (
    "Ты Telegram-бот, который делает живой рекап чата по-русски, просто и без воды. "
    "Верни ТОЛЬКО HTML-текст для Telegram (без markdown, без заголовков). "
    "Важно: если в чате обсуждали что-то потенциально незаконное/опасное — перескажи без инструкций и деталей, "
    "только общими словами."
  )

  if link_prefix:
    link_rules = (
      f"У тебя есть префикс ссылки на сообщения: {link_prefix}\n"
      "Когда в рассказе упоминаешь важный момент/тему — вставляй ссылку прямо в тексте так:\n"
      f'<a href="{link_prefix}123">короткая фраза</a>\n'
      "Где 123 — это message id из строк вида [id=...].\n"
      f'Например, Стас рассказал как <a href="{link_prefix}123">запилил новую фичу с нейронками</a> на своём проекте. \n'
      "Не делай списков/пунктов. Вставь 4–8 ссылок максимум, естественно по ходу рассказа, и ставь ссылки на семантические фразы, не на имена авторов.\n"
      "Ссылки должны быть именно на сообщения этого чата.\n"
    )
  else:
    link_rules = (
      "В этом чате нет стабильных публичных ссылок на сообщения (не супергруппа/нет username). "
      "Пиши рекап без ссылок.\n"
    )

  user_prompt = (
      "Ниже идут сообщения чата в хронологическом порядке.\n"
      "Сделай по ним короткий пересказ для человека, который пропустил разговор.\n"
      "Стиль: как в болтовне с друзьями, живо, с эмоциями, но без кринжа.\n"
      "Формат: 2–4 абзаца, цельный рассказ, без списков, добавляя ссылку на сообщение для каждого семантически нового топика.\n"
      "Не цитируй дословно.\n"
      "Строки [voice transcript] и [video note transcript] — это расшифровки голосовых сообщений и кружочков; считай, что это сказал автор строки, а не бот.\n"
      "Строки [image description] — это автоматическое описание картинок, которые отправил автор; включай их в рассказ как «автор поделился изображением: ...».\n"
      "Отмечай форварды как 'кто-то форварднул ...' если это важно (они помечены как FORWARDED).\n"
      "Укажи, остались ли открытые вопросы/что дальше.\n"
      "Длина: чтобы влезло в одно сообщение Telegram (ориентир до ~1500–1800 символов).\n\n"
      + link_rules
      + "\nСообщения:\n"
      + conv_text
  )

  try:
    logger.info("Sending narrative request to model=%s link_prefix=%s", RECAP_MODEL, link_prefix)
    resp = await asyncio.to_thread(
        _chat_text,
        RECAP_MODEL,
        system,
        user_prompt,
        0.4,
        700,
    )
    text = (resp.choices[0].message.content or "").strip()
    text = _safe_trim(text, 4000)

    # Validate/strip unsafe or hallucinated links
    text = _sanitize_links_in_html(text, link_prefix, allowed_ids)
    text = _safe_trim(text, 3800)

    return text or "Не удалось сделать осмысленный рекап."
  except Exception as e:
    logger.exception("OpenAI error: %s", e)
    return "Не получилось обратиться к LLM и сделать рекап."


async def _handle_image_group(msgs: List) -> None:
  """Parse a batch of photo messages (1 or more) and store the result in history."""
  msgs = sorted(msgs, key=lambda m: m.message_id)
  capped = msgs[:MAX_IMAGES_PER_MESSAGE]
  if len(msgs) > MAX_IMAGES_PER_MESSAGE:
    logger.info(
        "Image group capped from %s to %s images for msg_id=%s",
        len(msgs),
        MAX_IMAGES_PER_MESSAGE,
        msgs[0].message_id,
    )

  # Pick caption from whichever message has one
  caption: Optional[str] = None
  for m in capped:
    if m.caption:
      caption = m.caption.strip()
      break

  anchor_msg = capped[0]
  logger.info(
      "Handling image group: %s image(s), chat_id=%s anchor_msg_id=%s",
      len(capped),
      anchor_msg.chat_id,
      anchor_msg.message_id,
  )

  try:
    data_uris = []
    for m in capped:
      uri = await _download_photo_data_uri(m)
      data_uris.append(uri)

    description = await asyncio.to_thread(_describe_images, data_uris, caption)
  except Exception as e:
    logger.exception("Image description error: %s", e)
    return

  if not description:
    return

  stored_text = f"{caption}\n{description}" if caption else description
  await _add_to_history(anchor_msg, text_override=stored_text, content_kind="image_description")


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  """Handle incoming photos: buffer album images with debounce, then describe them."""
  msg = update.effective_message
  if not msg or not msg.photo:
    return

  group_id = msg.media_group_id

  if not group_id:
    # Single photo — process immediately
    await _handle_image_group([msg])
    return

  # Album: buffer and debounce
  async with media_group_lock:
    media_group_buffer.setdefault(group_id, []).append(msg)

    existing_task = media_group_tasks.get(group_id)
    if existing_task and not existing_task.done():
      existing_task.cancel()

    async def _flush(gid: str) -> None:
      await asyncio.sleep(MEDIA_GROUP_DEBOUNCE_SECONDS)
      async with media_group_lock:
        buffered = media_group_buffer.pop(gid, [])
        media_group_tasks.pop(gid, None)
      if buffered:
        await _handle_image_group(buffered)

    task = asyncio.ensure_future(_flush(group_id))
    media_group_tasks[group_id] = task


# ==========================
# Handlers
# ==========================


async def debug_raw(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  """Log every update as dict to see what exactly we receive."""
  try:
    data = update.to_dict()
  except Exception:
    data = str(update)
  logger.info("RAW UPDATE: %s", data)


async def on_regular_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  """Store every non-command message in history if it has text/caption."""
  msg = update.effective_message
  if not msg:
    return
  if not msg.text and not msg.caption:
    return

  logger.info(
      "Incoming message chat_id=%s msg_id=%s from=%s",
      msg.chat_id,
      msg.message_id,
      _display_name_from_message(msg),
  )
  await _add_to_history(msg)


async def on_voice_or_video_note(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  """Transcribe Telegram voice messages and video notes, then store them in history."""
  msg = update.effective_message
  if not msg or not (msg.voice or msg.video_note):
    return

  logger.info(
      "Incoming media for transcription chat_id=%s msg_id=%s from=%s kind=%s",
      msg.chat_id,
      msg.message_id,
      _display_name_from_message(msg),
      "video_note" if msg.video_note else "voice",
  )

  try:
    transcript = await _transcribe_telegram_media(msg)
  except Exception as e:
    logger.exception("Transcription error: %s", e)
    await msg.reply_text("Не получилось расшифровать это сообщение.")
    return

  if not transcript:
    await msg.reply_text("Не получилось разобрать речь в этом сообщении.")
    return

  content_kind = "video_note_transcript" if msg.video_note else "voice_transcript"
  reply_msg = await msg.reply_text(_transcript_reply_text(msg, transcript))
  await _add_to_history(
      reply_msg,
      text_override=transcript,
      content_kind=content_kind,
      author_msg=msg,
      media_source_message_id=msg.message_id,
  )


async def cmd_recap(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  """Handle /recap command."""
  msg = update.effective_message
  chat = update.effective_chat

  if not chat or not msg:
    return

  chat_id = chat.id
  chat_username = getattr(chat, "username", None)
  prefix = _link_prefix(chat_id, chat_username)

  logger.info(
      "Received /recap in chat_id=%s msg_id=%s is_reply=%s",
      chat_id,
      msg.message_id,
      bool(msg.reply_to_message),
  )

  if not msg.reply_to_message:
    async with history_lock:
      messages = _select_slice_for_recap(chat_id, from_message_id=None)

    if not messages:
      await msg.reply_text("В памяти за последние сутки нет сообщений для рекапа.")
      return

    recap_html = await _summarize_conversation_narrative(messages, prefix)
    await msg.reply_text(
        recap_html,
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )
    return

  start_id = msg.reply_to_message.message_id
  logger.info("Recap starting from message_id=%s", start_id)

  async with history_lock:
    messages = _select_slice_for_recap(chat_id, from_message_id=start_id)

  if not messages:
    await msg.reply_text("Не нашёл сообщений начиная с этого реплая за последние сутки.")
    return

  recap_html = await _summarize_conversation_narrative(messages, prefix)
  await msg.reply_text(
      recap_html,
      parse_mode=ParseMode.HTML,
      disable_web_page_preview=True,
  )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  text = (
    "Я делаю короткий рекап переписки.\n\n"
    "1) Пролистай чат до первого непрочитанного сообщения.\n"
    "2) Ответь на него реплаем с командой /recap.\n"
    "3) Я возьму сообщения с него за последние сутки и сделаю живой пересказ.\n\n"
    "Поиск по истории: /search запрос, /s запрос, /п запрос или ? запрос.\n\n"
    "Важно: для работы в группах у бота должен быть выключен privacy mode, "
    "и бот должен быть добавлен в группу после изменения настройки."
  )
  await update.effective_message.reply_text(text)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
  logger.exception("Unhandled error: %s", context.error)


# ==========================
# Lifecycle hooks
# ==========================


async def _post_init(application) -> None:
  """Initialise PostgreSQL pool and start background tasks (if PG is configured)."""
  await recap_db.init_pool()
  if recap_db.is_enabled():
    q = recap_db.get_write_queue()
    if q is not None:
      application.bot_data["writer_task"] = asyncio.create_task(
          recap_db.run_writer(q)
      )
      application.bot_data["indexer_task"] = asyncio.create_task(
          recap_index.run_indexer()
      )
      logger.info("DB writer and indexer tasks started")
  else:
    logger.info("PostgreSQL not configured — running without persistent storage")


async def _post_shutdown(application) -> None:
  """Stop background tasks and close the DB pool cleanly."""
  indexer = application.bot_data.get("indexer_task")
  if indexer:
    indexer.cancel()
    try:
      await asyncio.wait_for(asyncio.shield(indexer), timeout=5.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
      pass

  writer = application.bot_data.get("writer_task")
  if writer:
    q = recap_db.get_write_queue()
    if q is not None:
      await q.put(None)  # sentinel: tell writer to stop
    try:
      await asyncio.wait_for(writer, timeout=10.0)
    except asyncio.TimeoutError:
      writer.cancel()

  await recap_db.close_pool()
  logger.info("DB pool closed")


# ==========================
# Entry point
# ==========================


def main() -> None:
  if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment.")
  if not OPENAI_TOKEN:
    raise RuntimeError("OPENAI_TOKEN is not set in environment.")

  logger.info("Starting recap bot")
  application = (
      ApplicationBuilder()
      .token(TELEGRAM_BOT_TOKEN)
      .post_init(_post_init)
      .post_shutdown(_post_shutdown)
      .build()
  )

  application.add_handler(
      MessageHandler(filters.ALL, debug_raw),
      group=-100,
  )

  application.add_handler(CommandHandler("recap", cmd_recap))
  application.add_handler(CommandHandler("help", cmd_help))
  application.add_handler(CommandHandler(["search", "s"], recap_search.cmd_search))
  application.add_handler(CommandHandler("init", recap_import.cmd_init))
  application.add_handler(CommandHandler("init_status", recap_import.cmd_init_status))
  application.add_handler(CommandHandler("init_cancel", recap_import.cmd_init_cancel))

  application.add_handler(
      MessageHandler(
          filters.TEXT & filters.Regex(recap_search.SEARCH_TEXT_ALIAS_PATTERN),
          recap_search.cmd_search,
      )
  )

  application.add_handler(MessageHandler(filters.VOICE | filters.VIDEO_NOTE, on_voice_or_video_note))
  application.add_handler(MessageHandler(filters.PHOTO, on_photo))
  application.add_handler(MessageHandler(filters.Document.ALL, recap_import.on_import_document))

  # Store all non-command messages (we'll keep only those with text/caption inside handler)
  application.add_handler(MessageHandler(~filters.COMMAND, on_regular_message))

  application.add_error_handler(error_handler)

  application.run_polling(allowed_updates=Update.ALL_TYPES)
  logger.info("Bot stopped")


if __name__ == "__main__":
  main()
