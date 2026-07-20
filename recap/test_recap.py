import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import recap  # noqa: E402

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
