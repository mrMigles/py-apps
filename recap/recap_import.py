# recap_import.py
# -*- coding: utf-8 -*-
"""
/init, /init_status, /init_cancel — Telegram Desktop export import.

Flow:
  /init      — authorise (chat admin/owner), enter «awaiting upload» mode.
  upload     — receive JSON / JSON.GZ / ZIP document, stream-parse with ijson,
               normalise each message to the DB schema, upsert idempotently,
               report progress by editing a status message.
  /init_status  — show message/chunk/unindexed counts from the DB.
  /init_cancel  — cancel an ongoing import; already-upserted rows are kept.

Pure helpers (normalize_export_message, _parse_export_text, _parse_export_date,
_parse_from_id, _is_admin_or_owner) are free of I/O so they can be unit-tested.

Telegram Desktop export format notes:
  - messages[].type       == "message" for chat messages (skip "service" etc.)
  - messages[].id         integer
  - messages[].date       ISO-8601 string  ("2024-01-15T12:00:00")
  - messages[].date_unixtime  unix timestamp as string (fallback)
  - messages[].from       display name  (string)
  - messages[].from_id    "user123456" / "channel123" etc.
  - messages[].text       string OR list of strings/entity-dicts
  - messages[].reply_to_message_id  integer (optional)
  - messages[].forwarded_from       string or None
  - messages[].media_type  "voice_message" | "video_message" | "sticker" | …
  - messages[].photo      non-null when it is a photo message
"""

import asyncio
import contextlib
import gzip
import json
import logging
import os
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import recap_db

logger = logging.getLogger("recap-bot.import")

# ---------------------------------------------------------------------------
# Per-chat mutable import state
# ---------------------------------------------------------------------------

# chat_id → True when the bot is waiting for the export document
_awaiting_upload: Dict[int, bool] = {}
# chat_id → running asyncio.Task
_import_tasks: Dict[int, asyncio.Task] = {}
# chat_id → asyncio.Event set to signal cancellation
_cancel_events: Dict[int, asyncio.Event] = {}


# ---------------------------------------------------------------------------
# Pure helper functions (unit-testable)
# ---------------------------------------------------------------------------

def _is_admin_or_owner(member_status: str) -> bool:
    """Return True if Telegram ChatMember status is administrator or creator."""
    return member_status in ("administrator", "creator")


def _parse_export_date(date_field) -> Optional[datetime]:
    """
    Parse a date from a Telegram Desktop export message.
    Accepts:
      - ISO-8601 string: "2024-01-15T12:00:00"
      - Unix timestamp string: "1705320000"
      - Plain int unix timestamp
    Returns aware UTC datetime or None.
    """
    if date_field is None:
        return None
    s = str(date_field).strip()
    # ISO-8601
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    # Unix timestamp
    try:
        ts = int(s)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        pass
    return None


def _parse_export_text(text_field) -> str:
    """
    Flatten the Telegram Desktop 'text' field.
    It can be:
      - a plain string
      - a list of strings and/or entity-dicts {"type":…, "text":…}
    Returns plain text (no HTML).
    """
    if isinstance(text_field, str):
        return text_field
    if isinstance(text_field, list):
        parts: List[str] = []
        for item in text_field:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _parse_from_id(from_id) -> Optional[int]:
    """
    Parse the from_id field which Telegram Desktop formats as "user123456",
    "channel123", "chat123", or a bare integer.
    Returns the numeric part as int, or None.
    """
    if from_id is None:
        return None
    s = str(from_id).strip()
    for prefix in ("user", "channel", "chat"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    try:
        return int(s)
    except ValueError:
        return None


def normalize_export_message(
    raw: dict,
    chat_id: int,
    chat_username: Optional[str],
) -> Optional[dict]:
    """
    Normalise one raw Telegram Desktop export message dict into a DB row dict.

    Returns None when the message should be skipped:
      - type != "message"
      - no text content after flattening
      - missing required fields (id, date)
    """
    if raw.get("type") != "message":
        return None

    text = _parse_export_text(raw.get("text", "")).strip()
    if not text:
        return None

    msg_id = raw.get("id")
    if not isinstance(msg_id, int):
        return None

    date = _parse_export_date(raw.get("date") or raw.get("date_unixtime"))
    if date is None:
        return None

    user_id = _parse_from_id(raw.get("from_id"))
    user_name = str(
        raw.get("from") or raw.get("actor") or f"User {user_id or msg_id}"
    )[:200]

    forwarded_from = raw.get("forwarded_from")
    is_forwarded = bool(forwarded_from)

    # Map export media_type to our content_kind vocabulary
    media_type = raw.get("media_type", "")
    if media_type == "voice_message":
        content_kind = "voice_transcript"
    elif media_type in ("video_message", "video"):
        content_kind = "video_note_transcript"
    elif raw.get("photo"):
        content_kind = "image_description"
    else:
        content_kind = "text"

    return {
        "chat_id": chat_id,
        "message_id": msg_id,
        "chat_username": chat_username,
        "user_id": user_id,
        "user_name": user_name,
        "text": text[:4000],
        "content_kind": content_kind,
        "date": date,
        "is_bot": False,
        "reply_to_message_id": raw.get("reply_to_message_id"),
        "reply_to_user_name": None,
        "reply_to_text": None,
        "is_forwarded": is_forwarded,
        "forward_from_name": str(forwarded_from)[:200] if forwarded_from else None,
        "forward_from_chat_title": None,
        "forward_from_chat_username": None,
        "forward_from_message_id": None,
        "media_source_message_id": None,
    }


# ---------------------------------------------------------------------------
# Export file streaming
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _open_export_stream(file_path: str, file_name: str):
    """
    Open the export file as a binary stream, handling .json, .json.gz, and .zip.
    For ZIP files, looks for result.json or the first .json entry inside.
    """
    name = file_name.lower()
    if name.endswith(".json.gz") or name.endswith(".gz"):
        with gzip.open(file_path, "rb") as f:
            yield f
    elif name.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zf:
            names = zf.namelist()
            entry = (
                next((n for n in names if n.lower().endswith("result.json")), None)
                or next((n for n in names if n.lower().endswith(".json")), None)
            )
            if not entry:
                raise ValueError(f"No JSON file found inside ZIP ({file_name})")
            with zf.open(entry) as f:
                yield f
    else:
        with open(file_path, "rb") as f:
            yield f


def _collect_rows_sync(
    file_path: str,
    file_name: str,
    chat_id: int,
    chat_username: Optional[str],
    cancel_event: asyncio.Event,
) -> List[dict]:
    """
    Synchronous: stream-parse the export file with ijson and return normalised
    message rows.  Respects cancel_event by checking it periodically (every
    500 messages).  Runs in a thread via asyncio.to_thread.
    """
    import ijson  # imported here so the module is importable without ijson installed

    rows: List[dict] = []
    with _open_export_stream(file_path, file_name) as f:
        for i, raw in enumerate(ijson.items(f, "messages.item")):
            if i % 500 == 0 and cancel_event.is_set():
                break
            row = normalize_export_message(raw, chat_id, chat_username)
            if row:
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Import task
# ---------------------------------------------------------------------------

async def _run_import(
    chat_id: int,
    chat_username: Optional[str],
    file_path: str,
    file_name: str,
    status_msg,            # telegram.Message — edited for progress
    cancel_event: asyncio.Event,
) -> None:
    """
    Full import pipeline:
      1. Stream-parse the export file (in a thread).
      2. Upsert rows in async batches of 100.
      3. Periodically edit status_msg with progress.
      4. Clean up temp file.
    """
    count = 0
    try:
        if not await recap_db.is_search_enabled(chat_id):
            await status_msg.edit_text(
                "Импорт остановлен: поиск и индексация в чате выключены."
            )
            return
        await status_msg.edit_text("Читаю файл экспорта…")
        try:
            rows = await asyncio.to_thread(
                _collect_rows_sync,
                file_path, file_name, chat_id, chat_username, cancel_event,
            )
        except Exception as exc:
            logger.exception("Export parse error for chat_id=%s: %s", chat_id, exc)
            await status_msg.edit_text(f"Ошибка при разборе файла: {exc}")
            return

        total = len(rows)
        if total == 0:
            await status_msg.edit_text("Не найдено текстовых сообщений для импорта.")
            return

        if cancel_event.is_set():
            await status_msg.edit_text("Импорт отменён до начала загрузки.")
            return

        # Wipe any previously stored history for this chat so the import
        # produces a clean, fully reindexed history instead of merging with
        # stale/partial data from an earlier import or from live traffic.
        await status_msg.edit_text(
            "Файл прочитан. Удаляю старую историю чата перед повторной индексацией…"
        )
        try:
            await recap_db.wipe_chat(chat_id)
        except Exception as exc:
            logger.exception("Failed to wipe chat history for chat_id=%s: %s", chat_id, exc)
            await status_msg.edit_text(f"Не удалось очистить старую историю: {exc}")
            return

        await status_msg.edit_text(f"Импортирую {total:,} сообщений…")

        for i, row in enumerate(rows):
            if cancel_event.is_set():
                await status_msg.edit_text(
                    f"Импорт отменён. Загружено сообщений: {count:,}."
                )
                return

            if not await recap_db.is_search_enabled(chat_id):
                await status_msg.edit_text(
                    f"Импорт остановлен: индексация выключена. "
                    f"Загружено сообщений: {count:,}."
                )
                return

            if await recap_db.upsert_message(row):
                count += 1

            # Edit progress every 500 messages
            if count % 500 == 0:
                try:
                    await status_msg.edit_text(
                        f"Импортирую… {count:,} / {total:,} сообщений."
                    )
                except Exception:
                    pass
                # Yield to event loop to stay responsive
                await asyncio.sleep(0)

        db_total = await recap_db.count_messages(chat_id)
        db_unindexed = await recap_db.count_unindexed(chat_id)
        await status_msg.edit_text(
            f"Импорт завершён: загружено {count:,} из {total:,} сообщений.\n"
            f"В базе сообщений: {db_total:,}, ожидают индексации: {db_unindexed:,}.\n"
            "Индексирование начнётся автоматически в течение нескольких минут."
        )

    except asyncio.CancelledError:
        try:
            await status_msg.edit_text(
                f"Импорт прерван. Загружено сообщений: {count:,}."
            )
        except Exception:
            pass
    except Exception as exc:
        logger.exception("Import task error for chat_id=%s: %s", chat_id, exc)
        try:
            await status_msg.edit_text(f"Ошибка импорта: {exc}")
        except Exception:
            pass
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass
        _import_tasks.pop(chat_id, None)
        _cancel_events.pop(chat_id, None)


# ---------------------------------------------------------------------------
# Telegram command handlers
# ---------------------------------------------------------------------------

async def cmd_init(update, context) -> None:
    """
    /init — request a Telegram Desktop export to import into this chat's history.
    Only chat administrators / owners may run this command.
    """
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

    chat_id = chat.id
    user = update.effective_user

    if not await recap_db.is_search_enabled(chat_id):
        await msg.reply_text(
            "Поиск и индексация в этом чате выключены. "
            "Сначала включите их командой /search_on."
        )
        return

    # Authorisation: admins/owner only (skip for private chats)
    if chat.type != "private":
        try:
            member = await context.bot.get_chat_member(chat_id, user.id)
            if not _is_admin_or_owner(member.status):
                await msg.reply_text(
                    "Эта команда доступна только администраторам чата."
                )
                return
        except Exception as exc:
            logger.warning("get_chat_member failed: %s", exc)
            await msg.reply_text("Не удалось проверить права. Попробуй позже.")
            return

    # If an import is already running, report it
    task = _import_tasks.get(chat_id)
    if task and not task.done():
        await msg.reply_text(
            "Импорт уже запущен. Используй /init_cancel чтобы отменить его."
        )
        return

    _awaiting_upload[chat_id] = True
    await msg.reply_text(
        "Пришли файл экспорта Telegram Desktop: result.json, result.json.gz или ZIP-архив. "
        "Экспортировать историю можно через меню чата в приложении Telegram Desktop.\n\n"
        "⚠️ После загрузки файла вся текущая сохранённая история этого чата будет "
        "удалена и переиндексирована с нуля.\n\n"
        "Для отмены — /init_cancel."
    )


async def cmd_init_status(update, context) -> None:
    """/init_status — show import and indexing progress for the current chat."""
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return

    if not recap_db.is_enabled():
        await msg.reply_text("PostgreSQL не подключён — постоянная история недоступна.")
        return

    chat_id = chat.id
    if not await recap_db.is_search_enabled(chat_id):
        await msg.reply_text(
            "Поиск и индексация в этом чате выключены. "
            "Администратор может включить их командой /search_on."
        )
        return
    total = await recap_db.count_messages(chat_id)
    chunks = await recap_db.count_chunks(chat_id)
    unindexed = await recap_db.count_unindexed(chat_id)

    task = _import_tasks.get(chat_id)
    running = task is not None and not task.done()

    status_line = "⏳ Импорт выполняется." if running else ""
    await msg.reply_text(
        f"История чата:\n"
        f"  Сообщений в базе: {total:,}\n"
        f"  Проиндексировано фрагментов: {chunks:,}\n"
        f"  Ожидают индексации: {unindexed:,}\n"
        f"{status_line}"
    )


async def cmd_init_cancel(update, context) -> None:
    """/init_cancel — cancel an ongoing import or clear the awaiting-upload state."""
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return

    chat_id = chat.id
    _awaiting_upload.pop(chat_id, None)

    cancel_ev = _cancel_events.get(chat_id)
    if cancel_ev:
        cancel_ev.set()

    task = _import_tasks.get(chat_id)
    if task and not task.done():
        await msg.reply_text("Отправлен сигнал отмены — импорт остановится после текущей порции.")
    else:
        await msg.reply_text("Нет активного импорта. Ожидание загрузки файла сброшено.")


async def on_import_document(update, context) -> None:
    """
    Handle an incoming document while in «awaiting upload» state.
    Silently ignores documents when no import has been initiated.
    """
    msg = update.effective_message
    chat = update.effective_chat
    if not msg or not chat:
        return

    chat_id = chat.id
    if not _awaiting_upload.get(chat_id):
        return  # not in import mode for this chat

    if not await recap_db.is_search_enabled(chat_id):
        _awaiting_upload.pop(chat_id, None)
        await msg.reply_text(
            "Файл не принят: поиск и индексация в чате выключены."
        )
        return

    doc = msg.document
    if not doc:
        return

    file_name = doc.file_name or "export.json"
    name_lower = file_name.lower()
    if not (
        name_lower.endswith(".json")
        or name_lower.endswith(".json.gz")
        or name_lower.endswith(".gz")
        or name_lower.endswith(".zip")
    ):
        await msg.reply_text(
            "Неподдерживаемый формат. Пришли result.json, result.json.gz или ZIP."
        )
        return

    # Clear awaiting state immediately so duplicate uploads are ignored
    _awaiting_upload.pop(chat_id, None)

    chat_username = getattr(chat, "username", None)

    status_msg = await msg.reply_text("Загружаю файл…")

    # Download to a temp file
    try:
        tg_file = await doc.get_file()
        suffix = (
            ".json.gz" if name_lower.endswith(".gz") else
            ".zip" if name_lower.endswith(".zip") else
            ".json"
        )
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="recap-import-")
        os.close(fd)
        await tg_file.download_to_drive(custom_path=temp_path)
    except Exception as exc:
        logger.exception("Failed to download export file: %s", exc)
        await status_msg.edit_text(f"Не удалось скачать файл: {exc}")
        return

    cancel_ev = asyncio.Event()
    _cancel_events[chat_id] = cancel_ev

    task = asyncio.create_task(
        _run_import(
            chat_id=chat_id,
            chat_username=chat_username,
            file_path=temp_path,
            file_name=file_name,
            status_msg=status_msg,
            cancel_event=cancel_ev,
        )
    )
    _import_tasks[chat_id] = task
