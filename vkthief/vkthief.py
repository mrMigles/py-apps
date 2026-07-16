import asyncio
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from telegram import Message, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from yt_dlp import YoutubeDL


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

BOT_TOKEN = os.environ.get("BOT_TOKEN")
MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "49"))
MAX_CAPTION_LENGTH = 1024
VIDEO_WORKER_COUNT = max(1, int(os.environ.get("VIDEO_WORKER_COUNT", "10")))
VIDEO_TASK_QUEUE_KEY = "video_task_queue"
VIDEO_CHAT_LOCKS_KEY = "video_chat_locks"
VIDEO_WORKER_TASKS_KEY = "video_worker_tasks"

START_TEXT = (
    "Кинь ссылку на VK clip/video, Instagram Reels, YouTube Shorts, TikTok "
    "или Rutube Shorts, я скачаю и пришлю mp4 файлом."
)
DOWNLOAD_STATUS_TEXT = "Скачиваю видео..."
UPLOAD_STATUS_TEXT = "Загружаю в Telegram..."
SIZE_TOO_LARGE_TEXT = (
    "Файл получился слишком большой: {size_mb:.1f} MB. "
    "Текущий лимит: {max_file_mb} MB."
)
FAILED_STATUS_TEXT = (
    "Не получилось скачать/отправить видео: {error} или файл слишком большой."
)

# Supported video examples:
# https://vk.ru/clip-213485029_456239556?c=1
# https://vkvideo.ru/clip-32012866_456245183
# https://www.instagram.com/reel/SHORTCODE/
# https://www.youtube.com/shorts/VIDEO_ID
# https://www.tiktok.com/@user/video/123456789
# https://rutube.ru/shorts/VIDEO_ID/
SUPPORTED_VIDEO_URL_RE = re.compile(
    r"https?://(?:(?:www\.|m\.)?(?:(?:vk\.ru|vk\.com|vkvideo\.ru)/(?:clip|video)\S+|"
    # r"instagram\.com/reels?/\S+|"
    r"youtube\.com/shorts/\S+|"
    r"tiktok\.com/@[^/\s]+/video/\S+|"
    r"rutube\.ru/shorts/\S+)|"
    r"(?:vm|vt)\.tiktok\.com/\S+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VideoTask:
    message: Message
    context: ContextTypes.DEFAULT_TYPE
    video_url: str
    partition_key: str


def extract_video_url(text: str) -> str | None:
    match = SUPPORTED_VIDEO_URL_RE.search(text or "")
    if not match:
        return None

    url = match.group(0).rstrip(".,!?)]}")
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    normalized_hostname = hostname.removeprefix("www.").removeprefix("m.")

    allowed_path_by_host = {
        "vk.ru": ("/clip", "/video"),
        "vk.com": ("/clip", "/video"),
        "vkvideo.ru": ("/clip", "/video"),
        "instagram.com": ("/reel/", "/reels/"),
        "youtube.com": ("/shorts/",),
        "tiktok.com": ("/@",),
        "vm.tiktok.com": ("/",),
        "vt.tiktok.com": ("/",),
        "rutube.ru": ("/shorts/",),
    }
    allowed_paths = allowed_path_by_host.get(normalized_hostname)
    if not allowed_paths or not parsed.path.startswith(allowed_paths):
        return None

    if normalized_hostname == "tiktok.com" and "/video/" not in parsed.path:
        return None

    return url


def get_chat_partition_key(message: Message) -> str:
    chat = message.chat
    for attr_name in ("username", "title", "full_name"):
        attr_value = getattr(chat, attr_name, None)
        if isinstance(attr_value, str) and attr_value.strip():
            return attr_value.strip().lower()

    return str(message.chat_id)


def format_video_caption(description: str | None) -> str | None:
    if not description:
        return None

    caption = description.strip()
    if not caption:
        return None

    if len(caption) <= MAX_CAPTION_LENGTH:
        return caption

    return caption[: MAX_CAPTION_LENGTH - 3].rstrip() + "..."


def download_video(url: str, work_dir: Path) -> tuple[Path, str | None]:
    output_template = str(work_dir / "%(extractor)s_%(id)s.%(ext)s")

    ydl_opts = {
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        "max_filesize": MAX_FILE_MB * 1024 * 1024,

        # Prefer a single MP4 file small enough for the default Telegram Bot API.
        # If size is unknown, yt-dlp may still download a larger file, so we check size after download.
        "format": (
            "best[ext=mp4][filesize<49M]/"
            "best[ext=mp4][filesize_approx<49M]/"
            "best[ext=mp4][height<=720]/"
            "best[ext=mp4]/"
            "best"
        ),

        # Avoid weird filenames.
        "restrictfilenames": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        description = info.get("description")
        if not isinstance(description, str):
            description = None

        requested = info.get("requested_downloads") or []
        if requested and requested[0].get("filepath"):
            return Path(requested[0]["filepath"]), description

        return Path(ydl.prepare_filename(info)), description


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(START_TEXT)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return

    message_text = message.text or message.caption
    if not message_text:
        return

    video_url = extract_video_url(message_text)
    if not video_url:
        return

    partition_key = get_chat_partition_key(message)
    task = VideoTask(
        message=message,
        context=context,
        video_url=video_url,
        partition_key=partition_key,
    )

    queue = context.application.bot_data.get(VIDEO_TASK_QUEUE_KEY)
    if queue is None:
        logging.warning("Video worker queue is not initialized, processing inline")
        await process_video_task(task)
        return

    await queue.put(task)
    logging.info(
        "Queued video task chat_partition=%s queue_size=%s",
        partition_key,
        queue.qsize(),
    )


async def process_video_task(task: VideoTask) -> None:
    message = task.message
    context = task.context
    video_url = task.video_url

    status = await message.reply_text(DOWNLOAD_STATUS_TEXT)
    temp_dir = Path(tempfile.mkdtemp(prefix="vk_tg_bot_"))
    downloaded_file: Path | None = None

    try:
        await context.bot.send_chat_action(
            chat_id=message.chat_id,
            action=ChatAction.UPLOAD_VIDEO,
        )

        downloaded_file, description = await asyncio.to_thread(download_video, video_url, temp_dir)

        if not downloaded_file.exists():
            raise RuntimeError("Downloaded file was not found")

        size_mb = downloaded_file.stat().st_size / 1024 / 1024
        if size_mb > MAX_FILE_MB:
            await status.edit_text(
                SIZE_TOO_LARGE_TEXT.format(size_mb=size_mb, max_file_mb=MAX_FILE_MB)
            )
            return

        await status.edit_text(UPLOAD_STATUS_TEXT)

        caption = format_video_caption(description)

        with downloaded_file.open("rb") as video_file:
            if downloaded_file.suffix.lower() == ".mp4":
                await message.reply_video(
                    video=video_file,
                    supports_streaming=True,
                    read_timeout=300,
                    write_timeout=300,
                    connect_timeout=30,
                    caption=caption,
                )
            else:
                await message.reply_document(
                    document=video_file,
                    read_timeout=300,
                    write_timeout=300,
                    connect_timeout=30,
                    caption=caption,
                )

        try:
            await status.delete()
        except Exception:
            pass

    except Exception as exc:
        logging.exception("Failed to process video")
        await status.edit_text(FAILED_STATUS_TEXT.format(error=exc))

    finally:
        # Each queued task owns its temp directory, so parallel workers never delete each other's files.
        shutil.rmtree(temp_dir, ignore_errors=True)


async def video_worker(
    worker_index: int,
    queue: asyncio.Queue[VideoTask],
    chat_locks: dict[str, asyncio.Lock],
) -> None:
    while True:
        task = await queue.get()
        try:
            lock = chat_locks.setdefault(task.partition_key, asyncio.Lock())
            async with lock:
                logging.info(
                    "Worker %s started video task chat_partition=%s",
                    worker_index,
                    task.partition_key,
                )
                await process_video_task(task)
        except Exception:
            logging.exception("Worker %s failed with an unhandled exception", worker_index)
        finally:
            queue.task_done()


async def start_video_workers(application: Application) -> None:
    queue: asyncio.Queue[VideoTask] = asyncio.Queue()
    chat_locks: dict[str, asyncio.Lock] = {}
    tasks = [
        asyncio.create_task(
            video_worker(worker_index, queue, chat_locks),
            name=f"video-worker-{worker_index}",
        )
        for worker_index in range(VIDEO_WORKER_COUNT)
    ]
    application.bot_data[VIDEO_TASK_QUEUE_KEY] = queue
    application.bot_data[VIDEO_CHAT_LOCKS_KEY] = chat_locks
    application.bot_data[VIDEO_WORKER_TASKS_KEY] = tasks
    logging.info("Started %s video workers", VIDEO_WORKER_COUNT)


async def stop_video_workers(application: Application) -> None:
    tasks = application.bot_data.get(VIDEO_WORKER_TASKS_KEY, [])
    for task in tasks:
        task.cancel()

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    logging.info("Stopped video workers")


def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("Set BOT_TOKEN environment variable")

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(start_video_workers)
        .post_shutdown(stop_video_workers)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    video_filter = ~filters.COMMAND & (
        filters.Regex(SUPPORTED_VIDEO_URL_RE) | filters.CaptionRegex(SUPPORTED_VIDEO_URL_RE)
    )
    app.add_handler(MessageHandler(video_filter, handle_message))

    app.run_polling()


if __name__ == "__main__":
    main()
