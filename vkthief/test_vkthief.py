from types import SimpleNamespace

import pytest

from vkthief import vkthief


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
