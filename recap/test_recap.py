import json
import pathlib
import sys
from types import SimpleNamespace
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


def test_reaction_update_is_not_treated_as_new_media():
    update = SimpleNamespace(
        message_reaction=SimpleNamespace(),
        message_reaction_count=None,
    )

    assert recap._is_reaction_update(update)


@pytest.mark.asyncio
@pytest.mark.parametrize("media_kind", ["voice", "video_note"])
@pytest.mark.parametrize("update_kind", ["message", "channel_post", "business_message"])
async def test_media_edits_do_not_repeat_transcription(monkeypatch, media_kind, update_kind):
    payload = {
        "message_id": 42,
        "date": 1788880000,
        "chat": {"id": -100123, "type": "supergroup"},
        "from": {"id": 7, "is_bot": False, "first_name": "Иван"},
        media_kind: {"file_id": "media", "file_unique_id": "unique", "duration": 45,
                     **({"length": 240} if media_kind == "video_note" else {})},
    }
    original = recap.Update.de_json({"update_id": 1, update_kind: payload}, None)
    edited = recap.Update.de_json({
        "update_id": 2, "edited_" + update_kind: {**payload, "edit_date": 1788908000},
    }, None)
    transcribe = AsyncMock(return_value="Расшифровка")
    reply = AsyncMock(return_value=MagicMock())
    add_history = AsyncMock()
    monkeypatch.setattr(recap, "_transcribe_telegram_media", transcribe)
    monkeypatch.setattr(recap.Message, "reply_text", reply)
    monkeypatch.setattr(recap, "_add_to_history", add_history)

    await recap.on_voice_or_video_note(original, MagicMock())
    await recap.on_voice_or_video_note(edited, MagicMock())
    # Even with empty history (e.g. after restart), edits must stay silent.
    recap.chat_history.clear()
    await recap.on_voice_or_video_note(edited, MagicMock())

    transcribe.assert_awaited_once_with(original.effective_message)
    reply.assert_awaited_once_with("Расшифровка")
    add_history.assert_awaited_once()


@pytest.mark.asyncio
async def test_media_handler_ignores_reaction_update(monkeypatch):
    update = SimpleNamespace(
        message_reaction=SimpleNamespace(),
        message_reaction_count=None,
        effective_message=SimpleNamespace(voice=SimpleNamespace(), video_note=None),
    )
    transcribe = AsyncMock(side_effect=AssertionError("must not transcribe reactions"))
    monkeypatch.setattr(recap, "_transcribe_telegram_media", transcribe)

    await recap.on_voice_or_video_note(update, MagicMock())

    transcribe.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_missing_chat_setting_means_search_disabled(monkeypatch):
    cursor = MagicMock()
    cursor.fetchone = AsyncMock(return_value=None)
    connection = MagicMock()
    connection.execute = AsyncMock(return_value=cursor)
    pool = MagicMock()
    pool.connection.return_value.__aenter__ = AsyncMock(return_value=connection)
    pool.connection.return_value.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(recap_db, "_pool", pool)
    monkeypatch.setattr(recap_db, "_search_enabled_cache", {})

    assert await recap_db.is_search_enabled(-100404) is False
    connection.execute.assert_awaited_once_with(
        "SELECT search_enabled FROM chat_settings WHERE chat_id = %s",
        (-100404,),
    )


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/search клубника", "клубника"),
        ("/search@recap_bot клубника", "клубника"),
        ("/s  клубника вчера", "клубника вчера"),
        ("/п клубника", "клубника"),
        ("? клубника", "клубника"),
        ("?\nклубника", "клубника"),
        ("/s", ""),
        ("обычное сообщение", None),
    ],
)
def test_extract_search_query_aliases(text, expected):
    assert recap_search.extract_search_query(text) == expected


@pytest.mark.parametrize("text", ["/п запрос", "/П запрос", "? запрос", "?"])
def test_search_text_alias_pattern(text):
    assert recap_search.SEARCH_TEXT_ALIAS_PATTERN.search(text)


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


def test_parse_search_filters_keywords():
    f = recap_search.parse_search_filters(
        json.dumps({"query": "поездка", "keywords": ["поездка", "Грузия", ""]})
    )
    assert f["keywords"] == ["поездка", "Грузия"]


def test_extract_keywords_drops_question_and_stop_words():
    words = recap_search.extract_keywords("Когда мы обсуждали поездку в Грузию?")
    assert words == ["поездку", "грузию"]


def test_build_term_queries_sanitises_tsquery_syntax():
    frags = recap_search.build_term_queries(
        ["поездка!", "Грузия & (Тбилиси) | :*'", "поезд в Москву", "где", "", "Поездка"]
    )
    assert frags == ["поездка:*", "грузия:* & тбилиси:*", "поезд:* & москву:*"]
    for frag in frags:
        assert "|" not in frag and "(" not in frag and "'" not in frag and "!" not in frag


def test_build_term_queries_short_tokens_are_exact():
    assert recap_search.build_term_queries(["ПК"]) == ["пк"]


def test_build_or_tsquery():
    assert recap_search.build_or_tsquery(["a:*", "b:* & c:*"]) == "(a:*) | (b:* & c:*)"
    assert recap_search.build_or_tsquery([]) == ""


@pytest.mark.parametrize(
    "text, expected",
    [
        ("/search поездка", True),
        ("  ? поездка", True),
        ("поездка?", False),
        ("", False),
    ],
)
def test_is_search_command_text(text, expected):
    assert recap_search.is_search_command_text(text) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Кто знает, где купить билеты", True),
        ("Билеты уже купили?", True),
        ("Билеты купили на сайте РЖД", False),
    ],
)
def test_looks_like_question(text, expected):
    assert recap_search.looks_like_question(text) is expected


def _hit(mid, minutes, user, text, rank=0.1, terms=(0,)):
    base = datetime(2025, 3, 12, 10, 0, tzinfo=timezone.utc)
    return {
        "message_id": mid,
        "media_source_message_id": None,
        "date": base + timedelta(minutes=minutes),
        "user_id": hash(user) % 1000,
        "user_name": user,
        "text": text,
        "rank": rank,
        "matched_terms": list(terms),
    }


def test_cluster_hits_splits_on_time_gap():
    hits = [
        _hit(1, 0, "Иван", "билеты"),
        _hit(2, 10, "Пётр", "билеты"),
        _hit(3, 200, "Иван", "билеты"),
    ]
    clusters = recap_search.cluster_hits(hits, n_terms=1)
    assert sorted(len(c.hits) for c in clusters) == [1, 2]


def test_cluster_hits_discussion_outranks_lone_question():
    hits = [
        # A lone question much later whose rank alone beats the whole
        # discussion (0.9 > 0.4 × 1.5): only the question penalty sinks it.
        _hit(50, 600, "Оля", "Кто знает, где купить билеты в Тбилиси?", rank=0.9, terms=(0, 1)),
        # The actual discussion.
        _hit(10, 0, "Иван", "Билеты в Тбилиси купили на сайте", rank=0.2, terms=(0, 1)),
        _hit(11, 5, "Пётр", "Да, билеты по 200 лари", rank=0.2, terms=(0,)),
    ]
    clusters = recap_search.cluster_hits(hits, n_terms=2)
    assert [h["message_id"] for h in clusters[0].hits] == [10, 11]
    assert clusters[0].authors == ["Иван", "Пётр"]
    assert clusters[1].hit_ids == [50]


def test_cluster_hits_participant_boost():
    hits = [
        _hit(1, 0, "Иван", "билеты", rank=0.1),
        _hit(2, 300, "Пётр", "билеты", rank=0.12),
    ]
    assert recap_search.cluster_hits(hits, 1)[0].hits[0]["message_id"] == 2
    boosted = recap_search.cluster_hits(hits, 1, participants=["иван"])
    assert boosted[0].hits[0]["message_id"] == 1


def test_candidate_hit_ids_prefer_discussion_over_question():
    c = recap_search.Candidate(
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        hits=[
            _hit(1, 0, "Оля", "Где билеты?", rank=0.9),
            _hit(2, 1, "Иван", "Билеты у меня", rank=0.1),
        ],
    )
    assert c.hit_ids == [2, 1]


def _window():
    return [
        {"message_id": 1, "media_source_message_id": None, "user_name": "Оля", "text": "Где билеты?"},
        {"message_id": 2, "media_source_message_id": None, "user_name": "Иван <b>", "text": "У меня & в почте"},
        {"message_id": 3, "media_source_message_id": None, "user_name": "Пётр", "text": "ок"},
        {"message_id": 4, "media_source_message_id": None, "user_name": "Иван", "text": "x" * 500},
    ]


def test_build_excerpt_orders_escapes_and_links():
    excerpt = recap_search.build_excerpt(
        _window(), hit_ids=[2, 1], cited_ids=[4], link_prefix="https://t.me/c/1/", n=3
    )
    lines = excerpt.split("\n")
    # Chronological order of the three picked messages (cited 4, hits 2 and 1).
    assert [l.rsplit("#", 1)[1] for l in lines] == ["1</a>", "2</a>", "4</a>"]
    assert "Иван &lt;b&gt;" in lines[1]
    assert "У меня &amp; в почте" in lines[1]
    assert '<a href="https://t.me/c/1/4">#4</a>' in lines[2]
    assert "…" in lines[2]


def test_build_excerpt_without_link_prefix_and_fills_with_followups():
    excerpt = recap_search.build_excerpt(
        _window(), hit_ids=[1], cited_ids=[], link_prefix=None, n=2
    )
    assert excerpt.split("\n")[0].endswith("(#1)")
    assert excerpt.split("\n")[1].endswith("(#2)")
    assert "<a " not in excerpt


def test_normalize_export_message_skips_search_commands_and_bot():
    base = {
        "type": "message",
        "id": 7,
        "date": "2024-01-01T00:00:00",
        "from": "Иван",
        "from_id": "user100",
    }
    assert recap_import.normalize_export_message({**base, "text": "/search билеты"}, -100, None) is None
    assert recap_import.normalize_export_message({**base, "text": "? билеты"}, -100, None) is None
    assert recap_import.normalize_export_message(
        {**base, "text": "Нашёл ответ", "from_id": "user555"}, -100, None, bot_user_id=555
    ) is None
    assert recap_import.normalize_export_message(
        {**base, "text": "Нашёл ответ"}, -100, None, bot_user_id=555
    ) is not None


@pytest.mark.asyncio
async def test_find_discussions_builds_or_query_and_adds_vector_fallback(monkeypatch):
    t0 = datetime(2025, 3, 12, 10, 0, tzinfo=timezone.utc)
    search = AsyncMock(return_value=[_hit(10, 0, "Иван", "билеты", terms=(0,))])
    monkeypatch.setattr(recap_db, "search_messages", search)
    monkeypatch.setattr(recap_search, "_embed_sync", lambda text: [0.1])
    monkeypatch.setattr(recap_db, "vector_search_chunks", AsyncMock(return_value=[
        # Overlaps the full-text discussion → skipped.
        {"start_date": t0 - timedelta(minutes=5), "end_date": t0 + timedelta(minutes=5)},
        # Separate discussion → appended.
        {"start_date": t0 + timedelta(days=3), "end_date": t0 + timedelta(days=3, minutes=30)},
    ]))

    cands = await recap_search.find_discussions(
        -100, {"query": "билеты", "keywords": ["билеты", "Грузия"]}, "билеты", bot_id=77
    )

    kwargs = search.await_args.kwargs
    assert kwargs["tsquery"] == "(билеты:*) | (грузия:*)"
    assert kwargs["term_queries"] == ["билеты:*", "грузия:*"]
    assert kwargs["exclude_user_id"] == 77
    assert [c.source for c in cands] == ["fts", "vector"]


def _two_candidates():
    t0 = datetime(2025, 3, 12, 10, 0, tzinfo=timezone.utc)
    return [
        recap_search.Candidate(start=t0, end=t0, hits=[_hit(10, 0, "Иван", "билеты")], authors=["Иван"]),
        recap_search.Candidate(start=t0 + timedelta(days=1), end=t0 + timedelta(days=1), source="vector"),
    ]


def _window_for(chat_id, start, end, **kw):
    if start.day == 12:
        return [
            {"message_id": 9, "media_source_message_id": None, "user_name": "Оля", "text": "Где билеты?"},
            {"message_id": 10, "media_source_message_id": None, "user_name": "Иван", "text": "Билеты у меня"},
        ]
    return [{"message_id": 20, "media_source_message_id": None, "user_name": "Пётр", "text": "Второе обсуждение"}]


async def _run_search(monkeypatch, user_id=42, answer="Билеты у Ивана."):
    chat_id = -100999
    recap_search._SESSIONS.clear()
    monkeypatch.setattr(recap_db, "is_enabled", lambda: True)
    monkeypatch.setattr(recap_db, "is_search_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(
        recap_search, "_normalise_query_sync", lambda q: json.dumps({"query": q})
    )
    monkeypatch.setattr(recap_search, "find_discussions", AsyncMock(return_value=_two_candidates()))
    monkeypatch.setattr(recap_db, "get_context_window", AsyncMock(side_effect=_window_for))
    monkeypatch.setattr(recap_search, "_generate_answer_sync", lambda messages, query: answer)

    processing_msg = _FakeMessage()
    update = MagicMock()
    update.effective_message.text = "/search билеты"
    update.effective_message.reply_text = AsyncMock(return_value=processing_msg)
    update.effective_chat = _FakeChat(chat_id)
    update.effective_user.id = user_id
    context = MagicMock()

    await recap_search.cmd_search(update, context)
    return processing_msg


def _buttons(markup):
    return [b.callback_data.rsplit(":", 1)[1] for b in markup.inline_keyboard[0]]


@pytest.mark.asyncio
async def test_search_edits_status_message_into_result_with_buttons(monkeypatch):
    processing_msg = await _run_search(monkeypatch, answer="Билеты у Ивана.")

    processing_msg.edit_text.assert_awaited_once()
    call = processing_msg.edit_text.await_args
    text = call.args[0]
    assert "Результат 1/2" in text
    # The LLM forgot the [id=N] marker: a citation to the discussion (not the
    # question) is injected and rendered as a link.
    assert 'Билеты у Ивана. <a href="https://t.me/c/999/10">#10</a>' in text
    assert _buttons(call.kwargs["reply_markup"]) == ["ok", "more"]

    (session,) = recap_search._SESSIONS.values()
    assert session.user_id == 42
    assert session.index == 0


def _callback(session, action, user_id, chat_id=-100999):
    query = MagicMock()
    query.data = f"srch:{session.token}:{action}" if session else f"srch:nope:{action}"
    query.from_user.id = user_id
    query.message.chat_id = chat_id
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query
    return update, query


@pytest.mark.asyncio
async def test_more_by_author_edits_same_message_with_next_result(monkeypatch):
    await _run_search(monkeypatch)
    (session,) = recap_search._SESSIONS.values()

    update, query = _callback(session, "more", 42)
    await recap_search.on_search_callback(update, MagicMock())

    query.edit_message_text.assert_awaited_once()
    call = query.edit_message_text.await_args
    assert "Результат 2/2" in call.args[0]
    assert "Второе обсуждение" in call.args[0]
    # Last result: «Ещё» disappears, «OK» stays.
    assert _buttons(call.kwargs["reply_markup"]) == ["ok"]
    assert session.index == 1

    # Another «Ещё» on the last page does not edit anything.
    update, query = _callback(session, "more", 42)
    await recap_search.on_search_callback(update, MagicMock())
    query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_buttons_are_only_for_search_author(monkeypatch):
    await _run_search(monkeypatch)
    (session,) = recap_search._SESSIONS.values()

    for action in ("more", "ok"):
        update, query = _callback(session, action, user_id=7)
        await recap_search.on_search_callback(update, MagicMock())
        assert query.answer.await_args.kwargs.get("show_alert") is True
        assert "автору" in query.answer.await_args.args[0]
        query.edit_message_text.assert_not_awaited()
        query.edit_message_reply_markup.assert_not_awaited()
    assert session.index == 0


@pytest.mark.asyncio
async def test_ok_removes_buttons_and_session(monkeypatch):
    await _run_search(monkeypatch)
    (session,) = recap_search._SESSIONS.values()

    update, query = _callback(session, "ok", 42)
    await recap_search.on_search_callback(update, MagicMock())

    query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)
    query.edit_message_text.assert_not_awaited()
    assert session.token not in recap_search._SESSIONS


@pytest.mark.asyncio
async def test_expired_search_session_alerts(monkeypatch):
    recap_search._SESSIONS.clear()
    update, query = _callback(None, "more", 42)
    await recap_search.on_search_callback(update, MagicMock())
    assert "устарел" in query.answer.await_args.args[0]
    query.edit_message_text.assert_not_awaited()

@pytest.mark.asyncio
async def test_search_is_disabled_by_default_for_chat(monkeypatch):
    monkeypatch.setattr(recap_db, "is_enabled", lambda: True)
    monkeypatch.setattr(recap_db, "is_search_enabled", AsyncMock(return_value=False))
    monkeypatch.setattr(
        recap_search,
        "_normalise_query_sync",
        MagicMock(side_effect=AssertionError("LLM must not be called")),
    )

    update = MagicMock()
    update.effective_message.reply_text = AsyncMock()
    update.effective_chat = _FakeChat(-100123)
    context = MagicMock()

    await recap_search.cmd_search(update, context)

    reply = update.effective_message.reply_text.await_args.args[0]
    assert "выключены" in reply
    assert "/search_on" in reply


@pytest.mark.asyncio
async def test_search_on_persists_setting_for_admin(monkeypatch):
    monkeypatch.setattr(recap_db, "is_enabled", lambda: True)
    set_enabled = AsyncMock()
    monkeypatch.setattr(recap_db, "set_search_enabled", set_enabled)

    update = MagicMock()
    update.effective_message.reply_text = AsyncMock()
    update.effective_chat = _FakeChat(-100123)
    update.effective_chat.type = "supergroup"
    update.effective_user.id = 42
    context = MagicMock()
    context.bot.get_chat_member = AsyncMock(
        return_value=MagicMock(status="administrator")
    )

    await recap_search.cmd_search_on(update, context)

    set_enabled.assert_awaited_once_with(-100123, True)
    assert "включены" in update.effective_message.reply_text.await_args.args[0]


@pytest.mark.asyncio
async def test_search_off_rejected_for_non_admin(monkeypatch):
    monkeypatch.setattr(recap_db, "is_enabled", lambda: True)
    set_enabled = AsyncMock()
    monkeypatch.setattr(recap_db, "set_search_enabled", set_enabled)

    update = MagicMock()
    update.effective_message.reply_text = AsyncMock()
    update.effective_chat = _FakeChat(-100123)
    update.effective_chat.type = "supergroup"
    update.effective_user.id = 42
    context = MagicMock()
    context.bot.get_chat_member = AsyncMock(return_value=MagicMock(status="member"))

    await recap_search.cmd_search_off(update, context)

    set_enabled.assert_not_awaited()
    assert "только администраторам" in update.effective_message.reply_text.await_args.args[0]


# ===========================================================================
# End-to-end tests
#
# The real Application (handler routing, filters, Bot API (de)serialisation)
# processes real Update JSON. Telegram is an in-process fake BaseRequest that
# records every Bot API call, and the LLM is a fake client. Search tests run
# against a throw-away embedded PostgreSQL + pgvector (pgserver) — the actual
# SQL and the schema migration are executed; they are skipped when pgserver
# is not installed. Nothing leaves the machine.
# ===========================================================================

import asyncio  # noqa: E402
import itertools  # noqa: E402
import time  # noqa: E402
from urllib.parse import urlparse  # noqa: E402

import pytest_asyncio  # noqa: E402
from telegram import Update  # noqa: E402
from telegram.request import BaseRequest  # noqa: E402

if sys.platform == "win32":
    # psycopg's async driver does not support the Proactor event loop.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

BOT_ID = 5000
AUTHOR_ID = 42
OTHER_ID = 43
GROUP_ID = -100777
NAMES = {AUTHOR_ID: "Оля", OTHER_ID: "Пётр", 44: "Иван"}
_update_ids = itertools.count(1)


def _chat_json(chat_id: int) -> dict:
    if chat_id > 0:
        return {"id": chat_id, "type": "private", "first_name": NAMES.get(chat_id, "U")}
    return {"id": chat_id, "type": "supergroup", "title": "Тестовый чат"}


def _user_json(user_id: int) -> dict:
    if user_id == BOT_ID:
        return {"id": BOT_ID, "is_bot": True, "first_name": "Recap", "username": "recap_test_bot"}
    return {"id": user_id, "is_bot": False, "first_name": NAMES.get(user_id, "U")}


class FakeTelegram(BaseRequest):
    """Minimal in-process Bot API: records calls, returns plausible results."""

    def __init__(self):
        self.calls = []
        self.forbidden_chats = set()
        self.undeletable = False
        self._message_ids = itertools.count(10_000)

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    @property
    def read_timeout(self):
        return None

    def _message(self, chat_id, message_id, text=""):
        return {
            "message_id": message_id,
            "date": int(time.time()),
            "chat": _chat_json(chat_id),
            "from": _user_json(BOT_ID),
            "text": text,
        }

    async def do_request(self, url, method, request_data=None, **kwargs):
        name = url.rsplit("/", 1)[-1]
        params = dict(request_data.parameters) if request_data else {}
        self.calls.append((name, params))

        def error(code, description):
            return code, json.dumps(
                {"ok": False, "error_code": code, "description": description}
            ).encode()

        if name == "getMe":
            result = _user_json(BOT_ID)
        elif name == "sendMessage":
            chat_id = int(params["chat_id"])
            if chat_id in self.forbidden_chats:
                return error(403, "Forbidden: bot can't initiate conversation with a user")
            result = self._message(chat_id, next(self._message_ids), params.get("text", ""))
        elif name in ("editMessageText", "editMessageReplyMarkup"):
            result = self._message(
                int(params["chat_id"]), int(params["message_id"]), params.get("text", "")
            )
        elif name == "deleteMessage":
            if self.undeletable:
                return error(400, "Bad Request: message can't be deleted")
            result = True
        elif name == "answerCallbackQuery":
            result = True
        else:
            raise AssertionError(f"Unexpected Bot API call: {name}")
        return 200, json.dumps({"ok": True, "result": result}).encode()

    def sent(self, name):
        return [p for n, p in self.calls if n == name]

    def names(self):
        return [n for n, _ in self.calls if n != "getMe"]


def _message_json(text, message_id, user_id, chat_id, date=None, reply_to=None):
    date = date or datetime.now(timezone.utc)
    data = {
        "message_id": message_id,
        "date": int(date.timestamp()),
        "chat": _chat_json(chat_id),
        "from": _user_json(user_id),
        "text": text,
    }
    if text.startswith("/"):
        command = text.split()[0]
        data["entities"] = [{"type": "bot_command", "offset": 0, "length": len(command)}]
    if reply_to is not None:
        data["reply_to_message"] = reply_to
    return data


async def send_text(app, text, message_id, user_id=AUTHOR_ID, chat_id=GROUP_ID, **kw):
    """Deliver a message update through the real Application."""
    msg = _message_json(text, message_id, user_id, chat_id, **kw)
    await app.process_update(
        Update.de_json({"update_id": next(_update_ids), "message": msg}, app.bot)
    )
    return msg


async def press_button(app, callback_data, user_id, message_id, chat_id=GROUP_ID):
    """Deliver an inline-button press through the real Application."""
    data = {
        "update_id": next(_update_ids),
        "callback_query": {
            "id": str(next(_update_ids)),
            "from": _user_json(user_id),
            "chat_instance": "test",
            "data": callback_data,
            "message": _message_json("result", message_id, BOT_ID, chat_id),
        },
    }
    await app.process_update(Update.de_json(data, app.bot))


class FakeLLM:
    """OpenAI-compatible client stub: *reply(system, user)* produces the text."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.embeddings = SimpleNamespace(create=self._embed)

    def _create(self, model, messages, **kwargs):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        self.prompts.append((system, user))
        content = self.reply(system, user)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def _embed(self, model, input):
        return SimpleNamespace(data=[SimpleNamespace(embedding=_unit_vector())])


def _unit_vector():
    return [1.0] + [0.0] * (recap_db.EMBEDDING_DIM - 1)


@pytest_asyncio.fixture
async def tg(monkeypatch):
    """Real Application wired to the fake Telegram; returns (app, fake)."""
    monkeypatch.setattr(recap_db, "PG_DATABASE", None)
    fake = FakeTelegram()
    app = recap.build_application(
        "123:TEST", request=fake, get_updates_request=FakeTelegram()
    )
    await app.initialize()
    try:
        yield app, fake
    finally:
        await app.shutdown()


def _recap_llm(system, user):
    # Links to a message of this chat survive link sanitising.
    return 'Иван <a href="https://t.me/c/777/2">рассказал про поход</a>, все согласились.'


async def _seed_group_history(app):
    await send_text(app, "Всем привет, какие планы на выходные?", 1, user_id=OTHER_ID)
    await send_text(app, "Идём в поход на Эльбрус", 2, user_id=44)
    await send_text(app, "Я с вами, беру палатку", 3, user_id=AUTHOR_ID)


# --- /privaterecap and /recap ----------------------------------------------

@pytest.mark.asyncio
async def test_e2e_private_recap_goes_to_dm_and_request_is_deleted(tg, monkeypatch):
    app, fake = tg
    llm = FakeLLM(_recap_llm)
    monkeypatch.setattr(recap, "client", llm)
    await _seed_group_history(app)

    await send_text(app, "/privaterecap", 10)

    assert fake.names() == ["deleteMessage", "sendMessage"]
    (deleted,) = fake.sent("deleteMessage")
    assert int(deleted["chat_id"]) == GROUP_ID and int(deleted["message_id"]) == 10

    (dm,) = fake.sent("sendMessage")
    assert int(dm["chat_id"]) == AUTHOR_ID
    assert dm["parse_mode"] == "HTML"
    assert dm["text"].startswith("Рекап чата «Тестовый чат»:")
    assert '<a href="https://t.me/c/777/2">рассказал про поход</a>' in dm["text"]
    # The recap covers the group's history.
    _, prompt = llm.prompts[-1]
    assert "Эльбрус" in prompt and "палатку" in prompt and "/privaterecap" not in prompt


@pytest.mark.asyncio
async def test_e2e_private_recap_from_replied_message(tg, monkeypatch):
    app, fake = tg
    llm = FakeLLM(_recap_llm)
    monkeypatch.setattr(recap, "client", llm)
    await _seed_group_history(app)
    replied = _message_json("Идём в поход на Эльбрус", 2, 44, GROUP_ID)

    await send_text(app, "/privaterecap", 11, reply_to=replied)

    _, prompt = llm.prompts[-1]
    assert "Эльбрус" in prompt
    assert "какие планы" not in prompt  # message 1 precedes the replied one
    (dm,) = fake.sent("sendMessage")
    assert int(dm["chat_id"]) == AUTHOR_ID


@pytest.mark.asyncio
async def test_e2e_private_recap_when_dm_is_closed(tg, monkeypatch):
    app, fake = tg
    monkeypatch.setattr(recap, "client", FakeLLM(_recap_llm))
    await _seed_group_history(app)
    fake.forbidden_chats.add(AUTHOR_ID)

    await send_text(app, "/privaterecap", 12)

    assert fake.sent("deleteMessage")
    dm_attempt, notice = fake.sent("sendMessage")
    assert int(dm_attempt["chat_id"]) == AUTHOR_ID
    assert int(notice["chat_id"]) == GROUP_ID
    assert f'<a href="tg://user?id={AUTHOR_ID}">' in notice["text"]
    assert "/start" in notice["text"]
    # The recap itself is never posted to the group.
    assert "поход" not in notice["text"]


@pytest.mark.asyncio
async def test_e2e_private_recap_still_sent_without_delete_right(tg, monkeypatch):
    app, fake = tg
    monkeypatch.setattr(recap, "client", FakeLLM(_recap_llm))
    await _seed_group_history(app)
    fake.undeletable = True

    await send_text(app, "/privaterecap", 13)

    (dm,) = fake.sent("sendMessage")
    assert int(dm["chat_id"]) == AUTHOR_ID
    assert "поход" in dm["text"]


@pytest.mark.asyncio
async def test_e2e_private_recap_with_empty_history(tg, monkeypatch):
    app, fake = tg
    llm = FakeLLM(_recap_llm)
    monkeypatch.setattr(recap, "client", llm)

    await send_text(app, "/privaterecap", 14)

    (dm,) = fake.sent("sendMessage")
    assert int(dm["chat_id"]) == AUTHOR_ID
    assert "нет сообщений" in dm["text"]
    assert not llm.prompts


@pytest.mark.asyncio
async def test_e2e_recap_still_replies_in_group(tg, monkeypatch):
    app, fake = tg
    monkeypatch.setattr(recap, "client", FakeLLM(_recap_llm))
    await _seed_group_history(app)

    await send_text(app, "/recap", 15)

    assert fake.names() == ["sendMessage"]
    (reply,) = fake.sent("sendMessage")
    assert int(reply["chat_id"]) == GROUP_ID
    assert reply["reply_parameters"]["message_id"] == 15
    assert "поход" in reply["text"]


# --- search against a real PostgreSQL ---------------------------------------

@pytest.fixture(scope="session")
def pg_uri(tmp_path_factory):
    pgserver = pytest.importorskip("pgserver", reason="embedded PostgreSQL not installed")
    server = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="stop")
    return server.get_uri()


def _drop_tables(uri):
    import psycopg

    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS messages, chunks, chat_settings")


@pytest_asyncio.fixture
async def pg(pg_uri, monkeypatch):
    """Point recap_db at an empty embedded database; returns its URI."""
    parsed = urlparse(pg_uri)
    monkeypatch.setattr(recap_db, "PG_HOST", parsed.hostname)
    monkeypatch.setattr(recap_db, "PG_PORT", parsed.port)
    monkeypatch.setattr(recap_db, "PG_DATABASE", parsed.path.lstrip("/"))
    monkeypatch.setattr(recap_db, "PG_USERNAME", parsed.username)
    monkeypatch.setattr(recap_db, "PG_PASSWORD", "unused")
    recap_db._search_enabled_cache.clear()
    _drop_tables(pg_uri)
    try:
        yield pg_uri
    finally:
        await recap_db.close_pool()
        _drop_tables(pg_uri)


def _db_row(message_id, user_id, text, date, user_name=None):
    return {
        "chat_id": GROUP_ID,
        "message_id": message_id,
        "chat_username": None,
        "user_id": user_id,
        "user_name": user_name or NAMES.get(user_id, "U"),
        "text": text,
        "content_kind": "text",
        "date": date,
        "is_bot": False,
        "reply_to_message_id": None,
        "reply_to_user_name": None,
        "reply_to_text": None,
        "is_forwarded": False,
        "forward_from_name": None,
        "forward_from_chat_title": None,
        "forward_from_chat_username": None,
        "forward_from_message_id": None,
        "media_source_message_id": None,
    }


@pytest.mark.asyncio
async def test_e2e_migration_adds_fts_to_existing_prod_schema(pg):
    import psycopg

    # The schema as deployed before message-level FTS, with data in it.
    legacy_schema = recap_db._SCHEMA_SQL.split("-- Message-level full-text search")[0]
    with psycopg.connect(pg, autocommit=True) as conn:
        conn.execute(legacy_schema)
        conn.execute(
            "INSERT INTO messages (chat_id, message_id, user_id, user_name, text, date) "
            "VALUES (%s, 1, 44, 'Иван', 'Поездка в Грузию была отличная', NOW())",
            (GROUP_ID,),
        )

    # Bot start-up runs the migration — twice, as on every restart.
    await recap_db.init_pool()
    await recap_db.close_pool()
    await recap_db.init_pool()

    with psycopg.connect(pg, autocommit=True) as conn:
        assert conn.execute(
            "SELECT indexname FROM pg_indexes WHERE indexname = 'messages_tsv'"
        ).fetchone()
        (tsv,) = conn.execute("SELECT tsv::text FROM messages WHERE message_id = 1").fetchone()
    assert "поездк" in tsv  # existing row backfilled, stemmed and lower-cased

    fragments = recap_search.build_term_queries(["поездку", "Грузии"])
    hits = await recap_db.search_messages(
        GROUP_ID, recap_search.build_or_tsquery(fragments), fragments
    )
    assert [h["message_id"] for h in hits] == [1]
    assert sorted(hits[0]["matched_terms"]) == [0, 1]


async def _seed_search_history():
    """A discussion, a later lone question, search noise and another topic."""
    await recap_db.set_search_enabled(GROUP_ID, True)
    day1 = datetime(2025, 5, 1, 12, 0, tzinfo=timezone.utc)
    day3 = day1 + timedelta(days=2)
    day5 = day1 + timedelta(days=4)
    rows = [
        _db_row(199, 44, "Смотрите, я забронировал отель", day1 - timedelta(minutes=3)),
        _db_row(200, 44, "Билеты в Тбилиси брали на сайте авиакомпании, вышло 200 евро", day1),
        _db_row(201, OTHER_ID, "Да, билеты лучше покупать заранее, в Тбилиси летом дорого", day1 + timedelta(minutes=4)),
        _db_row(202, 44, "И поезд из Батуми до Тбилиси удобный", day1 + timedelta(minutes=9)),
        _db_row(250, 44, "Кто пойдёт на футбол в субботу?", day3),
        _db_row(251, OTHER_ID, "Я пойду, возьму шарф", day3 + timedelta(minutes=2)),
        _db_row(300, AUTHOR_ID, "А кто помнит, где мы брали билеты в Тбилиси?", day5),
        _db_row(301, OTHER_ID, "не помню", day5 + timedelta(minutes=1)),
        # Noise from an old /init import: a search request and the bot's answer.
        _db_row(400, AUTHOR_ID, "? билеты в Тбилиси", day5 + timedelta(minutes=5)),
        _db_row(401, BOT_ID, "Нашёл: билеты в Тбилиси брали на сайте", day5 + timedelta(minutes=6), "Recap"),
    ]
    for row in rows:
        assert await recap_db.upsert_message(row)
    # An indexed chunk for the semantic fallback.
    await recap_db.insert_chunk(
        chat_id=GROUP_ID, summary="Футбол в субботу", keywords=["футбол"],
        message_ids=[250, 251], first_message_id=250,
        start_date=day3, end_date=day3 + timedelta(minutes=2),
        embedding=_unit_vector(),
    )


def _search_llm(system, user):
    if "нормализуешь" in system:
        return json.dumps({
            "query": "где покупали билеты в Тбилиси",
            "keywords": ["билеты", "Тбилиси"],
        })
    if "[id=200]" in user:
        return "Билеты брали на сайте авиакомпании [id=200], вышло 200 евро."
    if "[id=300]" in user:
        return "Здесь только вопрос, ответа нет."
    return "Про билеты тут ничего нет."


def _keyboard_actions(markup):
    return [b["callback_data"].rsplit(":", 1)[1] for b in markup["inline_keyboard"][0]]


@pytest.mark.asyncio
async def test_e2e_search_finds_discussion_and_paginates(pg, tg, monkeypatch):
    app, fake = tg
    monkeypatch.setattr(recap_db, "PG_DATABASE", urlparse(pg).path.lstrip("/"))
    await recap_db.init_pool()
    await _seed_search_history()
    monkeypatch.setattr(recap_search, "_client", FakeLLM(_search_llm))
    recap_search._SESSIONS.clear()

    await send_text(app, "? где мы покупали билеты в Тбилиси", 500)

    # "Ищу…" is a reply to the request and is then edited into result 1.
    (searching,) = fake.sent("sendMessage")
    assert searching["text"] == "Ищу…"
    assert searching["reply_parameters"]["message_id"] == 500
    result_id = fake.calls[-1][1]["message_id"]
    first = fake.sent("editMessageText")[-1]
    assert "Результат 1/3" in first["text"]
    # The discussion, not the later question or the old search/bot noise.
    assert 'Билеты брали на сайте авиакомпании <a href="https://t.me/c/777/200">#200</a>' in first["text"]
    assert "t.me/c/777/201" in first["text"]
    for noise in ("/300", "/400", "/401"):
        assert noise not in first["text"]
    assert _keyboard_actions(first["reply_markup"]) == ["ok", "more"]
    token = first["reply_markup"]["inline_keyboard"][0][1]["callback_data"]

    # Somebody else cannot page through the author's search.
    edits_before = len(fake.sent("editMessageText"))
    await press_button(app, token, OTHER_ID, result_id)
    alert = fake.sent("answerCallbackQuery")[-1]
    assert alert["show_alert"] is True and "автору" in alert["text"]
    assert len(fake.sent("editMessageText")) == edits_before

    # «Ещё» edits the same message: the lone question comes next…
    await press_button(app, token, AUTHOR_ID, result_id)
    second = fake.sent("editMessageText")[-1]
    assert int(second["message_id"]) == int(result_id)
    assert "Результат 2/3" in second["text"]
    assert "t.me/c/777/300" in second["text"]
    assert "t.me/c/777/301" in second["text"]  # the follow-up is shown too

    # …then the semantic (embedding) match, with «Ещё» gone on the last page.
    await press_button(app, token, AUTHOR_ID, result_id)
    third = fake.sent("editMessageText")[-1]
    assert "Результат 3/3" in third["text"]
    assert "футбол" in third["text"]
    assert _keyboard_actions(third["reply_markup"]) == ["ok"]

    # Old search requests and the bot's own answers never show up anywhere.
    for page in (first, second, third):
        for noise in ("/400", "/401", "? билеты", "Нашёл:"):
            assert noise not in page["text"]

    # «OK» removes the buttons and ends the session.
    await press_button(app, token.replace(":more", ":ok"), AUTHOR_ID, result_id)
    (removed,) = fake.sent("editMessageReplyMarkup")
    assert int(removed["message_id"]) == int(result_id)
    assert not removed.get("reply_markup")
    assert not recap_search._SESSIONS

    # Later presses are answered as expired.
    await press_button(app, token, AUTHOR_ID, result_id)
    assert "устарел" in fake.sent("answerCallbackQuery")[-1]["text"]


@pytest.mark.asyncio
async def test_e2e_search_nothing_found(pg, tg, monkeypatch):
    app, fake = tg
    monkeypatch.setattr(recap_db, "PG_DATABASE", urlparse(pg).path.lstrip("/"))
    await recap_db.init_pool()
    await recap_db.set_search_enabled(GROUP_ID, True)
    monkeypatch.setattr(recap_search, "_client", FakeLLM(_search_llm))

    await send_text(app, "/search билеты", 600)

    assert fake.sent("editMessageText")[-1]["text"] == "По этому запросу ничего не найдено."
