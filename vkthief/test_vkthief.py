import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import vkthief  # noqa: E402


@pytest.mark.parametrize(
    "url",
    [
        "https://vk.com/clip-1_2",
        "https://vkvideo.ru/video-1_2",
        "https://www.youtube.com/shorts/abc",
        "https://www.tiktok.com/@user/video/123",
        "https://vm.tiktok.com/abc/",
        "https://rutube.ru/shorts/abc/",
    ],
)
def test_extract_video_url_supports_known_services(url):
    assert vkthief.extract_video_url(f"Посмотри {url}!") == url


@pytest.mark.parametrize(
    "text",
    [
        "https://youtube.com/watch?v=abc",
        "https://tiktok.com/@user",
        "https://example.com/shorts/abc",
        "без ссылки",
    ],
)
def test_extract_video_url_rejects_unsupported_links(text):
    assert vkthief.extract_video_url(text) is None


def test_format_video_caption_trims_and_limits_text():
    assert vkthief.format_video_caption(None) is None
    assert vkthief.format_video_caption("   ") is None
    assert vkthief.format_video_caption(" описание ") == "описание"

    result = vkthief.format_video_caption("x" * 2000)
    assert len(result) == vkthief.MAX_CAPTION_LENGTH
    assert result.endswith("...")


def test_get_chat_partition_key_prefers_username():
    message = SimpleNamespace(
        chat=SimpleNamespace(username=" MyChat ", title="Title", full_name=None),
        chat_id=123,
    )
    assert vkthief.get_chat_partition_key(message) == "mychat"


def test_download_video_uses_downloaded_file(monkeypatch, tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    captured_options = {}

    class FakeYoutubeDL:
        def __init__(self, options):
            captured_options.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def extract_info(self, url, download):
            assert url == "https://example.test/video"
            assert download is True
            return {
                "description": "Описание",
                "requested_downloads": [{"filepath": str(video_path)}],
            }

    monkeypatch.setattr(vkthief, "YoutubeDL", FakeYoutubeDL)

    path, description = vkthief.download_video("https://example.test/video", tmp_path)

    assert path == video_path
    assert description == "Описание"
    assert captured_options["noplaylist"] is True
    assert captured_options["max_filesize"] == vkthief.MAX_FILE_MB * 1024 * 1024


COOKIE_LINE = ".youtube.com\tTRUE\t/\tTRUE\t1893456000\tSID\tsecret"


def make_fake_youtube_dl(video_path, captured):
    class FakeYoutubeDL:
        def __init__(self, options):
            captured["options"] = dict(options)
            cookie_file = options.get("cookiefile")
            captured["cookies"] = (
                vkthief.Path(cookie_file).read_text(encoding="utf-8") if cookie_file else None
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def extract_info(self, url, download):
            return {"requested_downloads": [{"filepath": str(video_path)}]}

    return FakeYoutubeDL


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://youtube.com/shorts/abc", True),
        ("https://www.youtube.com/shorts/abc", True),
        ("https://m.youtube.com/shorts/abc", True),
        ("https://music.youtube.com/watch?v=abc", True),
        ("https://youtu.be/abc", True),
        ("https://www.youtube-nocookie.com/embed/abc", True),
        ("https://notyoutube.com/shorts/abc", False),
        ("https://youtube.com.evil.test/shorts/abc", False),
        ("https://vk.com/clip-1_2", False),
    ],
)
def test_is_youtube_url(url, expected):
    assert vkthief.is_youtube_url(url) is expected


def test_load_youtube_cookies_empty():
    assert vkthief.load_youtube_cookies("") is None
    assert vkthief.load_youtube_cookies("   ") is None


def test_load_youtube_cookies_from_file(tmp_path):
    cookie_path = tmp_path / "cookies.txt"
    cookie_path.write_text(
        "# Netscape HTTP Cookie File\n# comment\n\n" + COOKIE_LINE + "\n", encoding="utf-8"
    )

    assert vkthief.load_youtube_cookies(str(cookie_path)) == (
        "# Netscape HTTP Cookie File\n" + COOKIE_LINE + "\n"
    )


def test_load_youtube_cookies_from_content_adds_header_and_keeps_httponly():
    httponly = "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t1893456000\tHSID\tx"
    result = vkthief.load_youtube_cookies(COOKIE_LINE + "\n" + httponly)

    assert result == "# Netscape HTTP Cookie File\n" + COOKIE_LINE + "\n" + httponly + "\n"


def test_load_youtube_cookies_repairs_escaped_newlines_and_lost_tabs():
    spaced = COOKIE_LINE.replace("\t", " ")
    empty_value = ".youtube.com TRUE / FALSE 0 PREF"
    result = vkthief.load_youtube_cookies(spaced + "\\n" + empty_value)

    assert result == (
        "# Netscape HTTP Cookie File\n"
        + COOKIE_LINE
        + "\n.youtube.com\tTRUE\t/\tFALSE\t0\tPREF\t\n"
    )


def test_load_youtube_cookies_rejects_garbage_and_missing_file(tmp_path):
    assert vkthief.load_youtube_cookies("not cookies at all") is None
    assert vkthief.load_youtube_cookies(str(tmp_path / "missing.txt")) is None


def test_load_youtube_cookies_strips_bom(tmp_path):
    cookie_path = tmp_path / "cookies.txt"
    cookie_path.write_text("﻿# Netscape HTTP Cookie File\n" + COOKIE_LINE, encoding="utf-8")

    assert COOKIE_LINE in vkthief.load_youtube_cookies(str(cookie_path))


def test_load_youtube_cookies_logs_missing_file(tmp_path, caplog):
    missing = tmp_path / "missing.txt"

    assert vkthief.load_youtube_cookies(str(missing)) is None
    assert "YOUTUBE_COOKIES file not found" in caplog.text


def test_summarize_youtube_cookies_detects_login_and_expired():
    expired = ".youtube.com\tTRUE\t/\tTRUE\t1000\tLOGIN_INFO\told"
    session = ".youtube.com\tTRUE\t/\tFALSE\t0\tPREF\tx"
    text = vkthief.normalize_netscape_cookies("\n".join([COOKIE_LINE, expired, session]))

    assert vkthief.summarize_youtube_cookies(text) == {
        "count": 3,
        "expired": 1,
        "logged_in": True,  # SID is still valid
        "login_cookies": ["LOGIN_INFO", "SID"],
    }

    only_expired = vkthief.normalize_netscape_cookies(expired + "\n" + session)
    assert vkthief.summarize_youtube_cookies(only_expired)["logged_in"] is False


def test_log_youtube_cookies_status(monkeypatch, caplog):
    caplog.set_level("INFO")

    monkeypatch.setattr(vkthief, "YOUTUBE_COOKIES", "")
    assert vkthief.log_youtube_cookies_status() is False
    assert "YOUTUBE_COOKIES is not set" in caplog.text

    caplog.clear()
    monkeypatch.setattr(vkthief, "YOUTUBE_COOKIES", COOKIE_LINE)
    assert vkthief.log_youtube_cookies_status() is True
    assert "YouTube cookies loaded successfully: 1 cookies, 0 expired, login cookies: SID" in caplog.text

    caplog.clear()
    monkeypatch.setattr(vkthief, "YOUTUBE_COOKIES", ".youtube.com\tTRUE\t/\tFALSE\t0\tPREF\tx")
    assert vkthief.log_youtube_cookies_status() is True
    assert "no valid login cookies" in caplog.text

    caplog.clear()
    monkeypatch.setattr(vkthief, "YOUTUBE_COOKIES", "garbage that is not cookies at all ok")
    assert vkthief.log_youtube_cookies_status() is False
    assert "neither an existing file nor valid Netscape cookies" in caplog.text


def test_download_video_routes_yt_dlp_warnings_to_log(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        vkthief, "YoutubeDL", make_fake_youtube_dl(tmp_path / "video.mp4", captured)
    )

    vkthief.download_video("https://vk.com/clip-1_2", tmp_path)

    assert captured["options"]["logger"] is vkthief.YT_DLP_LOGGER


def test_download_video_passes_private_cookie_copy_for_youtube(monkeypatch, tmp_path):
    source = tmp_path / "source_cookies.txt"
    source.write_text(COOKIE_LINE + "\n", encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    captured = {}
    monkeypatch.setattr(vkthief, "YOUTUBE_COOKIES", str(source))
    monkeypatch.setattr(
        vkthief, "YoutubeDL", make_fake_youtube_dl(work_dir / "video.mp4", captured)
    )

    vkthief.download_video("https://m.youtube.com/shorts/abc", work_dir)

    cookie_file = vkthief.Path(captured["options"]["cookiefile"])
    assert cookie_file.parent == work_dir
    assert cookie_file != source
    assert captured["cookies"] == "# Netscape HTTP Cookie File\n" + COOKIE_LINE + "\n"


def test_download_video_skips_cookies_for_other_sites(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(vkthief, "YOUTUBE_COOKIES", COOKIE_LINE)
    monkeypatch.setattr(
        vkthief, "YoutubeDL", make_fake_youtube_dl(tmp_path / "video.mp4", captured)
    )

    vkthief.download_video("https://vk.com/clip-1_2", tmp_path)

    assert "cookiefile" not in captured["options"]


def test_download_video_without_cookies_env(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(vkthief, "YOUTUBE_COOKIES", "")
    monkeypatch.setattr(
        vkthief, "YoutubeDL", make_fake_youtube_dl(tmp_path / "video.mp4", captured)
    )

    vkthief.download_video("https://youtube.com/shorts/abc", tmp_path)

    assert "cookiefile" not in captured["options"]


FRESH_SIDCC = ".youtube.com\tTRUE\t/\tFALSE\t1893456000\tSIDCC\tfresh"


@pytest.fixture(autouse=True)
def reset_cookie_state(monkeypatch):
    monkeypatch.setattr(vkthief, "_youtube_cookies_in_memory", None)
    monkeypatch.setattr(vkthief, "_youtube_cookies_save_failed", False)


def make_refresh_youtube_dl(page, new_cookies, captured):
    """Fake YoutubeDL that 'gets' new cookies from YouTube and saves the jar on exit like yt-dlp."""

    class FakeResponse:
        def read(self):
            return page.encode()

    class FakeYoutubeDL:
        def __init__(self, options):
            captured["options"] = dict(options)
            self.cookie_file = vkthief.Path(options["cookiefile"])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.cookie_file.write_text(
                "# Netscape HTTP Cookie File\n" + new_cookies + "\n", encoding="utf-8"
            )

        def urlopen(self, url):
            captured["url"] = url
            return FakeResponse()

    return FakeYoutubeDL


def test_save_youtube_cookies_writes_back_to_file(monkeypatch, tmp_path):
    source = tmp_path / "cookies.txt"
    source.write_text(COOKIE_LINE + "\n", encoding="utf-8")
    updated = tmp_path / "updated.txt"
    updated.write_text(COOKIE_LINE + "\n" + FRESH_SIDCC + "\n", encoding="utf-8")
    monkeypatch.setattr(vkthief, "YOUTUBE_COOKIES", str(source))

    assert vkthief.save_youtube_cookies(updated) is True
    assert FRESH_SIDCC in source.read_text(encoding="utf-8")
    # Nothing new the second time: the file isn't rewritten.
    assert vkthief.save_youtube_cookies(updated) is False


def test_save_youtube_cookies_keeps_login_when_youtube_logs_out(monkeypatch, tmp_path):
    source = tmp_path / "cookies.txt"
    source.write_text(COOKIE_LINE + "\n", encoding="utf-8")
    updated = tmp_path / "updated.txt"
    updated.write_text(FRESH_SIDCC + "\n", encoding="utf-8")  # SID is gone
    monkeypatch.setattr(vkthief, "YOUTUBE_COOKIES", str(source))

    assert vkthief.save_youtube_cookies(updated) is False
    assert source.read_text(encoding="utf-8") == COOKIE_LINE + "\n"


def test_save_youtube_cookies_keeps_contents_in_memory(monkeypatch, tmp_path):
    updated = tmp_path / "updated.txt"
    updated.write_text(COOKIE_LINE + "\n" + FRESH_SIDCC + "\n", encoding="utf-8")
    monkeypatch.setattr(vkthief, "YOUTUBE_COOKIES", COOKIE_LINE)

    assert vkthief.save_youtube_cookies(updated) is True
    assert FRESH_SIDCC in vkthief.load_youtube_cookies()


@pytest.mark.parametrize(
    ("page", "expected", "saved"),
    [
        ('{"LOGGED_IN":true,"x":1}', True, True),
        ('{"LOGGED_IN": false}', False, False),
        ("<html>consent page</html>", None, True),
    ],
)
def test_refresh_youtube_cookies(monkeypatch, tmp_path, page, expected, saved):
    source = tmp_path / "cookies.txt"
    source.write_text(COOKIE_LINE + "\n", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(vkthief, "YOUTUBE_COOKIES", str(source))
    monkeypatch.setattr(
        vkthief, "YoutubeDL", make_refresh_youtube_dl(page, COOKIE_LINE + "\n" + FRESH_SIDCC, captured)
    )

    assert vkthief.refresh_youtube_cookies() is expected
    assert captured["url"] == vkthief.YOUTUBE_HOME_URL
    assert (FRESH_SIDCC in source.read_text(encoding="utf-8")) is saved


def test_refresh_youtube_cookies_without_cookies(monkeypatch):
    monkeypatch.setattr(vkthief, "YOUTUBE_COOKIES", "")
    assert vkthief.refresh_youtube_cookies() is None


def test_download_video_saves_refreshed_cookies(monkeypatch, tmp_path):
    source = tmp_path / "cookies.txt"
    source.write_text(COOKIE_LINE + "\n", encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    video_path = work_dir / "video.mp4"

    class FakeYoutubeDL:
        def __init__(self, options):
            self.cookie_file = vkthief.Path(options["cookiefile"])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.cookie_file.write_text(COOKIE_LINE + "\n" + FRESH_SIDCC + "\n", encoding="utf-8")

        def extract_info(self, url, download):
            return {"requested_downloads": [{"filepath": str(video_path)}]}

    monkeypatch.setattr(vkthief, "YOUTUBE_COOKIES", str(source))
    monkeypatch.setattr(vkthief, "YoutubeDL", FakeYoutubeDL)

    path, _ = vkthief.download_video("https://youtube.com/shorts/abc", work_dir)

    assert path == video_path
    assert FRESH_SIDCC in source.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_handle_message_enqueues_supported_url():
    queue = __import__("asyncio").Queue()
    message = SimpleNamespace(
        text="https://youtube.com/shorts/abc",
        caption=None,
        chat=SimpleNamespace(username="chat"),
        chat_id=42,
    )
    context = SimpleNamespace(
        application=SimpleNamespace(bot_data={vkthief.VIDEO_TASK_QUEUE_KEY: queue})
    )
    update = SimpleNamespace(message=message)

    await vkthief.handle_message(update, context)

    task = queue.get_nowait()
    assert task.video_url == "https://youtube.com/shorts/abc"
    assert task.partition_key == "chat"
