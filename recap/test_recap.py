from datetime import datetime, timedelta, timezone

import pytest

from recap import recap


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
