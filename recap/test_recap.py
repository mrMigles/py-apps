import json
import pathlib
import sys
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import recap  # noqa: E402
import recap_db  # noqa: E402
import recap_index  # noqa: E402
import recap_search  # noqa: E402
import recap_import  # noqa: E402

from datetime import datetime, timedelta, timezone

import pytest


def message(message_id: int, text: str = "Текст", *, age_hours: int = 0) -> recap.ChatMessage:
    return recap.ChatMessage(
        chat_id=-100123,
        chat_username=None,
        message_id=message_id,
        user_id=7,
        user_name="Иван",
        text=text,
        content_kind="text",
        date=datetime.now(timezone.utc) - timedelta(hours=age_hours),
        is_bot=False,
        reply_to_message_id=None,
        reply_to_user_name=None,
        reply_to_text=None,
        is_forwarded=False,
        forward_from_name=None,
        forward_from_chat_title=None,
        forward_from_chat_username=None,
        forward_from_message_id=None,
    )


@pytest.fixture(autouse=True)
def clear_history():
    recap.chat_history.clear()
    yield
    recap.chat_history.clear()


@pytest.mark.parametrize(
    ("value", "limit", "expected"),
    [
        ("  привет  ", 20, "привет"),
        ("abcdef", 4, "abc…"),
        ("abc", 1, "…"),
        ("abc", 0, ""),
    ],
)
def test_safe_trim(value, limit, expected):
    assert recap._safe_trim(value, limit) == expected


def test_link_prefix_for_public_and_private_supergroups():
    assert recap._link_prefix(-100123, "my_chat") == "https://t.me/my_chat/"
    assert recap._link_prefix(-100123, None) == "https://t.me/c/123/"
    assert recap._link_prefix(-123, None) is None


def test_sanitize_links_keeps_only_known_messages():
    prefix = "https://t.me/c/123/"
    html = (
        f'<a href="{prefix}10">известное</a> '
        f'<a href="{prefix}999">выдуманное</a> '
        '<a href="https://example.com/10">чужое</a>'
    )

    result = recap._sanitize_links_in_html(html, prefix, {10})

    assert f'<a href="{prefix}10">известное</a>' in result
    assert "выдуманное" in result and f'href="{prefix}999"' not in result
    assert "чужое" in result and "example.com" not in result


def test_build_conversation_marks_forwards_and_replies():
    item = message(10, "Обсудили релиз")
    item.is_forwarded = True
    item.forward_from_name = "Пётр"
    item.reply_to_message_id = 8

    result = recap._build_conversation_text([item])

    assert "[id=10] Иван: Обсудили релиз" in result
    assert "(FORWARDED from Пётр)" in result
    assert "(reply_to_id=8)" in result


def test_select_slice_filters_old_messages_and_start_id():
    recap.chat_history[-100123] = [
        message(1, age_hours=25),
        message(2),
        message(3),
    ]

    selected = recap._select_slice_for_recap(-100123, from_message_id=3)

    assert [item.message_id for item in selected] == [3]


def test_openai_client_requires_token_only_when_used(monkeypatch):
    monkeypatch.setattr(recap, "client", None)
    monkeypatch.setattr(recap, "OPENAI_TOKEN", None)

    with pytest.raises(RuntimeError, match="OPENAI_TOKEN"):
        recap._get_openai_client()


# ----- Image parsing tests -----

def test_media_kind_label_image_description():
    assert recap._media_kind_label("image_description") == "image description"


def test_media_kind_label_other_kinds_unchanged():
    assert recap._media_kind_label("voice_transcript") == "voice transcript"
    assert recap._media_kind_label("video_note_transcript") == "video note transcript"
    assert recap._media_kind_label("text") is None


def test_image_model_selection_single(monkeypatch):
    """Single image should use IMAGE_MODEL_BIG."""
    monkeypatch.setattr(recap, "IMAGE_MODEL_BIG", "big-model")
    monkeypatch.setattr(recap, "IMAGE_MODEL_SIMPLE", "simple-model")

    calls = []

    def fake_chat_vision(model, system, prompt_text, image_data_uris, **kwargs):
        calls.append(model)
        class _Resp:
            choices = [type("C", (), {"message": type("M", (), {"content": "описание"})()})()]
        return _Resp()

    monkeypatch.setattr(recap, "_chat_vision", fake_chat_vision)

    result = recap._describe_images(["data:image/jpeg;base64,AA=="], caption=None)
    assert calls == ["big-model"]
    assert result == "описание"


def test_image_model_selection_multiple(monkeypatch):
    """Multiple images should use IMAGE_MODEL_SIMPLE."""
    monkeypatch.setattr(recap, "IMAGE_MODEL_BIG", "big-model")
    monkeypatch.setattr(recap, "IMAGE_MODEL_SIMPLE", "simple-model")

    calls = []

    def fake_chat_vision(model, system, prompt_text, image_data_uris, **kwargs):
        calls.append(model)
        class _Resp:
            choices = [type("C", (), {"message": type("M", (), {"content": "описание"})()})()]
        return _Resp()

    monkeypatch.setattr(recap, "_chat_vision", fake_chat_vision)

    result = recap._describe_images(
        ["data:image/jpeg;base64,AA==", "data:image/jpeg;base64,BB=="],
        caption=None,
    )
    assert calls == ["simple-model"]
    assert result == "описание"


def test_image_caption_included_in_prompt(monkeypatch):
    """Caption text should appear in the user prompt sent to the vision model."""
    monkeypatch.setattr(recap, "IMAGE_MODEL_BIG", "big-model")
    monkeypatch.setattr(recap, "IMAGE_MODEL_SIMPLE", "simple-model")

    prompts = []

    def fake_chat_vision(model, system, prompt_text, image_data_uris, **kwargs):
        prompts.append(prompt_text)
        class _Resp:
            choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]
        return _Resp()

    monkeypatch.setattr(recap, "_chat_vision", fake_chat_vision)

    recap._describe_images(["data:image/jpeg;base64,AA=="], caption="тестовая подпись")
    assert "тестовая подпись" in prompts[0]


def test_image_max_cap_in_conversation_text():
    """image_description messages should appear tagged in conversation output."""
    img_msg = message(20, "описание котика")
    img_msg.content_kind = "image_description"

    result = recap._build_conversation_text([img_msg])
    assert "[image description]" in result
    assert "описание котика" in result


def test_image_stored_text_combines_caption_and_description():
    """stored_text should be caption + newline + description when caption is present."""
    caption = "смотри какой кот"
    description = "рыжий кот лежит на диване"
    stored = f"{caption}\n{description}"

    assert stored.startswith(caption)
    assert description in stored


# =============================================================================
# recap_db — graceful-degradation guard
# =============================================================================


def test_is_enabled_false_when_no_pg_database(monkeypatch):
    """is_enabled() must return False when PG_DATABASE is not set."""
    monkeypatch.setattr(recap_db, "PG_DATABASE", None)
    assert recap_db.is_enabled() is False


def test_is_enabled_true_when_pg_database_set(monkeypatch):
    monkeypatch.setattr(recap_db, "PG_DATABASE", "mydb")
    assert recap_db.is_enabled() is True


def test_get_write_queue_none_before_init():
    """Before init_pool(), get_write_queue() returns None."""
    assert recap_db._pool is None  # not initialised in unit tests
    assert recap_db.get_write_queue() is None


# =============================================================================
# recap_import — pure helper functions
# =============================================================================


def test_parse_export_date_iso():
    dt = recap_import._parse_export_date("2024-01-15T12:00:00")
    assert isinstance(dt, datetime)
    assert dt.year == 2024
    assert dt.month == 1
    assert dt.day == 15
    assert dt.tzinfo is not None


def test_parse_export_date_unix_string():
    dt = recap_import._parse_export_date("1705320000")
    assert isinstance(dt, datetime)
    assert dt.tzinfo is not None


def test_parse_export_date_none():
    assert recap_import._parse_export_date(None) is None
    assert recap_import._parse_export_date("") is None
    assert recap_import._parse_export_date("garbage") is None


def test_parse_export_text_plain_string():
    assert recap_import._parse_export_text("hello world") == "hello world"


def test_parse_export_text_entity_array():
    entities = [
        "Hello ",
        {"type": "bold", "text": "world"},
        "!",
    ]
    assert recap_import._parse_export_text(entities) == "Hello world!"


def test_parse_export_text_empty():
    assert recap_import._parse_export_text([]) == ""
    assert recap_import._parse_export_text(None) == ""


def test_parse_from_id_user_prefix():
    assert recap_import._parse_from_id("user123456") == 123456


def test_parse_from_id_channel_prefix():
    assert recap_import._parse_from_id("channel999") == 999


def test_parse_from_id_bare_int():
    assert recap_import._parse_from_id(42) == 42


def test_parse_from_id_none():
    assert recap_import._parse_from_id(None) is None


def test_is_admin_or_owner():
    assert recap_import._is_admin_or_owner("administrator") is True
    assert recap_import._is_admin_or_owner("creator") is True
    assert recap_import._is_admin_or_owner("member") is False
    assert recap_import._is_admin_or_owner("left") is False


def test_normalize_export_message_basic():
    raw = {
        "type": "message",
        "id": 42,
        "date": "2024-03-01T10:00:00",
        "from": "Иван",
        "from_id": "user100",
        "text": "Привет!",
    }
    row = recap_import.normalize_export_message(raw, chat_id=-100123, chat_username="mychat")

    assert row is not None
    assert row["message_id"] == 42
    assert row["user_name"] == "Иван"
    assert row["user_id"] == 100
    assert row["text"] == "Привет!"
    assert row["content_kind"] == "text"
    assert row["chat_id"] == -100123
    assert row["chat_username"] == "mychat"
    assert row["is_forwarded"] is False


def test_normalize_export_message_skips_service_type():
    raw = {"type": "service", "id": 1, "date": "2024-01-01T00:00:00", "text": "pin"}
    assert recap_import.normalize_export_message(raw, -100, None) is None


def test_normalize_export_message_skips_empty_text():
    raw = {
        "type": "message",
        "id": 5,
        "date": "2024-01-01T00:00:00",
        "from": "Bob",
        "from_id": "user5",
        "text": "",
        "media_type": "voice_message",  # audio only, no transcript
    }
    assert recap_import.normalize_export_message(raw, -100, None) is None


def test_normalize_export_message_forwarded():
    raw = {
        "type": "message",
        "id": 10,
        "date": "2024-01-01T00:00:00",
        "from": "Alice",
        "from_id": "user7",
        "text": "пересланное",
        "forwarded_from": "Пётр",
    }
    row = recap_import.normalize_export_message(raw, -100, None)
    assert row is not None
    assert row["is_forwarded"] is True
    assert row["forward_from_name"] == "Пётр"


def test_normalize_export_message_entity_text():
    raw = {
        "type": "message",
        "id": 20,
        "date": "2024-01-01T00:00:00",
        "from": "Alice",
        "from_id": "user7",
        "text": [
            "Смотри: ",
            {"type": "bold", "text": "важно"},
        ],
    }
    row = recap_import.normalize_export_message(raw, -100, None)
    assert row is not None
    assert row["text"] == "Смотри: важно"


def test_normalize_export_message_voice_content_kind():
    raw = {
        "type": "message",
        "id": 30,
        "date": "2024-01-01T00:00:00",
        "from": "Ivan",
        "from_id": "user8",
        "text": "расшифровка",
        "media_type": "voice_message",
    }
    row = recap_import.normalize_export_message(raw, -100, None)
    assert row is not None
    assert row["content_kind"] == "voice_transcript"


# =============================================================================
# recap_index — pure helper functions
# =============================================================================


def test_build_index_text_basic():
    rows = [
        {
            "message_id": 1,
            "user_name": "Иван",
            "user_id": 10,
            "text": "Привет",
            "content_kind": "text",
            "is_forwarded": False,
            "reply_to_message_id": None,
        },
        {
            "message_id": 2,
            "user_name": "Мария",
            "user_id": 20,
            "text": "Привет!",
            "content_kind": "text",
            "is_forwarded": False,
            "reply_to_message_id": 1,
        },
    ]
    result = recap_index.build_index_text(rows)
    assert "[id=1] Иван: Привет" in result
    assert "[id=2] Мария: Привет!" in result
    assert "(reply_to_id=1)" in result


def test_build_index_text_voice_transcript():
    rows = [{"message_id": 5, "user_name": "B", "user_id": 2, "text": "ok",
             "content_kind": "voice_transcript", "is_forwarded": False, "reply_to_message_id": None}]
    result = recap_index.build_index_text(rows)
    assert "[voice transcript]" in result


def test_parse_chunk_json_validates_ids():
    """Unknown message IDs returned by LLM must be dropped."""
    valid_ids = {10, 11, 12}
    raw = json.dumps([
        {
            "message_ids": [10, 11, 999],  # 999 is unknown
            "summary": "Обсудили релиз",
            "keywords": ["релиз"],
            "important_message_ids": [10, 999],
            "is_complete": True,
        }
    ])
    chunks = recap_index.parse_chunk_json(raw, valid_ids)
    assert len(chunks) == 1
    assert 999 not in chunks[0]["message_ids"]
    assert 999 not in chunks[0]["important_message_ids"]
    assert 10 in chunks[0]["message_ids"]
    assert 11 in chunks[0]["message_ids"]


def test_parse_chunk_json_drops_chunk_with_no_valid_ids():
    raw = json.dumps([
        {"message_ids": [999, 888], "summary": "x", "keywords": [], "important_message_ids": [], "is_complete": True}
    ])
    assert recap_index.parse_chunk_json(raw, {1, 2, 3}) == []


def test_parse_chunk_json_incomplete_flag():
    valid_ids = {1, 2, 3}
    raw = json.dumps([
        {"message_ids": [1, 2], "summary": "a", "keywords": [], "important_message_ids": [], "is_complete": True},
        {"message_ids": [3], "summary": "b", "keywords": [], "important_message_ids": [], "is_complete": False},
    ])
    chunks = recap_index.parse_chunk_json(raw, valid_ids)
    assert len(chunks) == 2
    assert chunks[0]["is_complete"] is True
    assert chunks[1]["is_complete"] is False


def test_parse_chunk_json_strips_markdown_fence():
    valid_ids = {5}
    raw = "```json\n" + json.dumps([
        {"message_ids": [5], "summary": "s", "keywords": [], "important_message_ids": [], "is_complete": True}
    ]) + "\n```"
    chunks = recap_index.parse_chunk_json(raw, valid_ids)
    assert len(chunks) == 1


def test_parse_chunk_json_invalid_json():
    assert recap_index.parse_chunk_json("not json", {1}) == []


def test_parse_chunk_json_not_array():
    raw = json.dumps({"message_ids": [1], "summary": "x", "is_complete": True})
    assert recap_index.parse_chunk_json(raw, {1}) == []


# =============================================================================
# recap_search — pure helper functions
# =============================================================================


def test_parse_search_filters_basic():
    raw = json.dumps({
        "query": "что обсуждали про деплой",
        "date_from": "2024-01-01",
        "date_to": "2024-01-31",
        "participants": ["Иван"],
        "exact_terms": ["k8s"],
    })
    f = recap_search.parse_search_filters(raw)
    assert f["query"] == "что обсуждали про деплой"
    assert isinstance(f["date_from"], datetime)
    assert isinstance(f["date_to"], datetime)
    assert f["participants"] == ["Иван"]
    assert f["exact_terms"] == ["k8s"]


def test_parse_search_filters_fallback_on_bad_json():
    result = recap_search.parse_search_filters("вот такой запрос")
    assert result["query"] == "вот такой запрос"


def test_parse_search_filters_markdown_fence():
    raw = "```json\n" + json.dumps({"query": "тест"}) + "\n```"
    f = recap_search.parse_search_filters(raw)
    assert f["query"] == "тест"


def test_extract_cited_ids_order_and_dedup():
    text = "Смотри [id=5] и [id=2], а еще раз [id=5] и [id=999]."
    result = recap_search.extract_cited_ids(text, {2, 5})
    assert result == [5, 2]


def test_extract_cited_ids_no_markers():
    assert recap_search.extract_cited_ids("Ничего не найдено.", {1, 2}) == []


def test_convert_id_markers_to_links_known_id():
    prefix = "https://t.me/c/123/"
    result = recap_search.convert_id_markers_to_links(
        "смотри [id=42] вот это", prefix, {42}
    )
    assert f'href="{prefix}42"' in result
    assert "[id=42]" not in result


def test_convert_id_markers_drops_unknown_id():
    prefix = "https://t.me/c/123/"
    result = recap_search.convert_id_markers_to_links(
        "смотри [id=999] вот это", prefix, {1, 2}
    )
    assert "[id=999]" not in result
    assert "href" not in result


def test_convert_id_markers_no_prefix():
    """Without a link prefix, markers become plain-text "(#N)" references."""
    result = recap_search.convert_id_markers_to_links(
        "смотри [id=5] вот", None, {5}
    )
    assert "[id=5]" not in result
    assert "href" not in result
    assert "(#5)" in result


def test_convert_id_markers_multiple():
    prefix = "https://t.me/mygroup/"
    result = recap_search.convert_id_markers_to_links(
        "[id=1] и [id=2] и [id=3]", prefix, {1, 3}
    )
    assert f'href="{prefix}1"' in result
    assert f'href="{prefix}3"' in result
    # id=2 is not in valid_ids
    assert f'href="{prefix}2"' not in result


def test_ensure_citation_adds_marker_when_missing():
    result = recap_search.ensure_citation("Сергей поделился фото клубники.", [42, 99])
    assert "[id=42]" in result
    assert result.startswith("Сергей поделился фото клубники.")


def test_ensure_citation_noop_when_marker_present():
    original = "Сергей поделился фото клубники [id=42]."
    result = recap_search.ensure_citation(original, [99])
    assert result == original
    assert "[id=99]" not in result


def test_ensure_citation_noop_when_no_fallback_ids():
    original = "Ничего не найдено."
    result = recap_search.ensure_citation(original, [])
    assert result == original


def test_ensure_citation_caps_at_three_markers():
    result = recap_search.ensure_citation("Текст.", [1, 2, 3, 4, 5])
    assert result.count("[id=") == 3


def test_convert_id_markers_strips_raw_id_list():
    prefix = "https://t.me/c/123/"
    # Some models dump a raw list of message IDs instead of [id=N] markers.
    result = recap_search.convert_id_markers_to_links(
        "Ничего не найдено [315950, 315957, 315958].", prefix, {315950}
    )
    assert "315950" not in result
    assert "315957" not in result
    assert "[" not in result
    assert result == "Ничего не найдено."


class _FakeMessage:
    def __init__(self):
        self.edit_text = AsyncMock()
        self.delete = AsyncMock()


class _FakeChat:
    def __init__(self, chat_id, username=None):
        self.id = chat_id
        self.username = username


@pytest.mark.asyncio
async def test_search_never_trades_off_reply_against_citation(monkeypatch):
    """
    /search must never trade the reply off against the citation link: even
    when the first reply-target candidate fails (e.g. an imported message
    that no longer exists live), it must retry the next candidate AND the
    sent text must keep carrying the citation/link — reply and citation are
    independent mechanisms, never alternatives to each other.
    """
    chat_id = -100999
    monkeypatch.setattr(recap_db, "is_enabled", lambda: True)
    monkeypatch.setattr(
        recap_search, "_normalise_query_sync", lambda q: json.dumps({"query": q})
    )
    monkeypatch.setattr(recap_search, "_embed_sync", lambda text: [0.1, 0.2])

    chunk = {
        "id": 1,
        "message_ids": [10, 11],
        "important_message_ids": [10],
        "first_message_id": 10,
        "summary": "тест",
    }
    monkeypatch.setattr(recap_db, "hybrid_search", AsyncMock(return_value=[chunk]))

    messages = [
        {"message_id": 10, "media_source_message_id": None, "user_name": "Иван", "text": "первое"},
        {"message_id": 11, "media_source_message_id": None, "user_name": "Пётр", "text": "второе"},
    ]
    monkeypatch.setattr(recap_db, "get_chunk_messages", AsyncMock(return_value=messages))

    # The LLM "forgets" the [id=N] marker convention entirely — ensure_citation
    # must inject a fallback citation from the chunk's important_message_ids.
    monkeypatch.setattr(
        recap_search, "_generate_answer_sync",
        lambda messages, query: "Нашёл ответ без маркера.",
    )

    processing_msg = _FakeMessage()
    update = MagicMock()
    update.effective_message.reply_text = AsyncMock(return_value=processing_msg)
    update.effective_chat = _FakeChat(chat_id, username=None)

    context = MagicMock()
    context.args = ["тест"]
    # First candidate (message_id=10) fails exactly like Telegram's real
    # "Message to be replied not found"; the second candidate (message_id=11)
    # must then be tried and succeed.
    context.bot.send_message = AsyncMock(
        side_effect=[Exception("Message to be replied not found"), None]
    )

    await recap_search.cmd_search(update, context)

    assert context.bot.send_message.call_count == 2
    first_call, second_call = context.bot.send_message.call_args_list
    assert first_call.kwargs["reply_to_message_id"] == 10
    assert second_call.kwargs["reply_to_message_id"] == 11
    # Both attempts carry the exact same citation-bearing text — the citation
    # never depends on whether the reply attempt itself succeeds.
    assert first_call.kwargs["text"] == second_call.kwargs["text"]
    assert "#10" in second_call.kwargs["text"]
    assert 'href="https://t.me/c/999/10"' in second_call.kwargs["text"]
