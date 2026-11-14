import asyncio
import logging
import os
import re
import shutil
import tempfile
import threading
import time
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
# httpx logs every request URL at INFO, and Telegram URLs contain the bot token.
logging.getLogger("httpx").setLevel(logging.WARNING)

BOT_TOKEN = os.environ.get("BOT_TOKEN")
MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "49"))
MAX_CAPTION_LENGTH = 1024
VIDEO_WORKER_COUNT = max(1, int(os.environ.get("VIDEO_WORKER_COUNT", "10")))
# Netscape-format cookies.txt for YouTube: either a path to the file or the file contents itself.
YOUTUBE_COOKIES = os.environ.get("YOUTUBE_COOKIES", "")
NETSCAPE_COOKIE_HEADER = "# Netscape HTTP Cookie File"
NETSCAPE_COOKIE_FIELDS = 7
YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")
# Present only when the Google account is signed in.
YOUTUBE_LOGIN_COOKIES = frozenset({"LOGIN_INFO", "SID", "__Secure-1PSID", "__Secure-3PSID"})
# yt-dlp sends its progress to logger.debug, warnings to .warning, errors to .error.
YT_DLP_LOGGER = logging.getLogger("yt_dlp")
# How often the bot visits youtube.com with the cookies to keep them fresh; 0 disables it.
YOUTUBE_COOKIES_REFRESH_HOURS = float(os.environ.get("YOUTUBE_COOKIES_REFRESH_HOURS", "24"))
YOUTUBE_HOME_URL = "https://www.youtube.com/"
YOUTUBE_LOGGED_IN_RE = re.compile(r'"LOGGED_IN"\s*:\s*(true|false)')
VIDEO_TASK_QUEUE_KEY = "video_task_queue"
VIDEO_CHAT_LOCKS_KEY = "video_chat_locks"
VIDEO_WORKER_TASKS_KEY = "video_worker_tasks"
YOUTUBE_COOKIES_REFRESH_TASK_KEY = "youtube_cookies_refresh_task"

# Downloads run in worker threads, so reading/writing the cookies file is guarded by a lock.
_youtube_cookies_lock = threading.Lock()
# Refreshed cookies when YOUTUBE_COOKIES holds the contents (there is no file to write to).
_youtube_cookies_in_memory: str | None = None
_youtube_cookies_save_failed = False

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


def is_youtube_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    # Covers youtube.com, www./m./music.youtube.com, youtu.be, www.youtube-nocookie.com.
    return any(hostname == host or hostname.endswith("." + host) for host in YOUTUBE_HOSTS)


def youtube_cookies_path(value: str | None = None) -> Path | None:
    """The cookies file if YOUTUBE_COOKIES (or `value`) is a path to an existing file."""
    value = (YOUTUBE_COOKIES if value is None else value).strip()
    if not value or "\n" in value or "\t" in value:
        return None
    try:
        cookie_path = Path(value).expanduser()
        return cookie_path if cookie_path.is_file() else None
    except (OSError, ValueError):
        return None  # too long / invalid for a path: it's the cookie contents


def load_youtube_cookies(value: str | None = None) -> str | None:
    """Return Netscape cookies.txt text from YOUTUBE_COOKIES (a file path or the contents)."""
    from_env = value is None
    value = (YOUTUBE_COOKIES if value is None else value).strip()
    if not value:
        return None

    cookie_path = youtube_cookies_path(value)
    if cookie_path:
        try:
            # Re-read on every download so the file can be refreshed without a restart.
            with _youtube_cookies_lock:
                value = cookie_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            logging.warning("Cannot read YOUTUBE_COOKIES file %s: %s", cookie_path, exc)
            return None
    elif "\n" not in value and "\t" not in value and len(value.split()) < NETSCAPE_COOKIE_FIELDS - 1:
        # Too few fields to be even one cookie line, so it must be a path.
        logging.warning("YOUTUBE_COOKIES file not found: %s", value)
        return None
    elif from_env and _youtube_cookies_in_memory:
        # Cookies were given as contents (nothing to write to), so use the refreshed copy.
        return _youtube_cookies_in_memory

    return normalize_netscape_cookies(value)


def save_youtube_cookies(cookie_file: Path) -> bool:
    """Persist cookies that yt-dlp updated in `cookie_file` back to YOUTUBE_COOKIES."""
    global _youtube_cookies_in_memory, _youtube_cookies_save_failed

    try:
        updated = normalize_netscape_cookies(cookie_file.read_text(encoding="utf-8-sig"))
    except OSError:
        return False
    if not updated:
        return False

    current = load_youtube_cookies()
    if current == updated:
        return False
    if current and summarize_youtube_cookies(current)["logged_in"] and not (
        summarize_youtube_cookies(updated)["logged_in"]
    ):
        # Never overwrite a signed-in session with a jar YouTube has logged out.
        logging.warning("YouTube dropped the login cookies, keeping the old YOUTUBE_COOKIES")
        return False

    with _youtube_cookies_lock:
        cookie_path = youtube_cookies_path()
        if cookie_path is None:
            _youtube_cookies_in_memory = updated
            return True
        try:
            # Write in place: os.replace() fails on a file bind-mounted into Docker.
            cookie_path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            if not _youtube_cookies_save_failed:
                logging.warning(
                    "Cannot save refreshed YouTube cookies to %s (%s), keeping them in memory",
                    cookie_path,
                    exc,
                )
            _youtube_cookies_save_failed = True
            return False
    return True


def refresh_youtube_cookies() -> bool | None:
    """Open youtube.com with the cookies like a browser would and save what YouTube refreshed.

    Returns True if the account is signed in, False if not, None if unknown or no cookies.
    Only cookies YouTube itself re-sets (SIDCC, __Secure-*PSIDCC, YSC, VISITOR_*) get fresher;
    the main Google session (SID, __Secure-*PSIDTS) can only be extended by a real browser.
    """
    cookies_text = load_youtube_cookies()
    if not cookies_text:
        return None

    with tempfile.TemporaryDirectory(prefix="yt_cookies_") as temp_dir:
        cookie_file = Path(temp_dir) / "youtube_cookies.txt"
        cookie_file.write_text(cookies_text, encoding="utf-8")
        ydl_opts = {
            "cookiefile": str(cookie_file),
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 30,
            "logger": YT_DLP_LOGGER,
        }
        with YoutubeDL(ydl_opts) as ydl:  # saves the updated cookie jar to cookie_file on exit
            page = ydl.urlopen(YOUTUBE_HOME_URL).read().decode("utf-8", "replace")

        match = YOUTUBE_LOGGED_IN_RE.search(page)
        logged_in = None if match is None else match.group(1) == "true"
        if logged_in is False:
            logging.warning(
                "YouTube cookies refresh: the account is NOT signed in, re-export the cookies"
            )
            return False

        saved = save_youtube_cookies(cookie_file)
        logging.info(
            "YouTube cookies refresh: signed in=%s, cookies %s",
            "yes" if logged_in else "unknown",
            "updated" if saved else "unchanged",
        )
        return logged_in


def summarize_youtube_cookies(cookies_text: str) -> dict:
    now = time.time()
    names, expired = set(), set()
    for line in cookies_text.splitlines():
        fields = line.split("\t")
        if len(fields) != NETSCAPE_COOKIE_FIELDS:
            continue
        name = fields[5]
        names.add(name)
        try:
            expires = int(fields[4])
        except ValueError:
            expires = 0
        if 0 < expires < now:  # 0 means a session cookie
            expired.add(name)

    login_cookies = names & YOUTUBE_LOGIN_COOKIES
    return {
        "count": len(names),
        "expired": len(expired),
        "logged_in": bool(login_cookies - expired),
        "login_cookies": sorted(login_cookies),
    }


def log_youtube_cookies_status() -> bool:
    """Log at startup whether YOUTUBE_COOKIES is usable. Returns True if cookies were loaded."""
    if not YOUTUBE_COOKIES.strip():
        logging.info("YOUTUBE_COOKIES is not set, YouTube is downloaded without cookies")
        return False

    cookies_text = load_youtube_cookies()
    if not cookies_text:
        return False  # load_youtube_cookies already logged why

    summary = summarize_youtube_cookies(cookies_text)
    if summary["logged_in"]:
        logging.info(
            "YouTube cookies loaded successfully: %s cookies, %s expired, login cookies: %s",
            summary["count"],
            summary["expired"],
            ", ".join(summary["login_cookies"]),
        )
    else:
        logging.warning(
            "YouTube cookies loaded (%s cookies, %s expired), but there are no valid login "
            "cookies (%s): the account is not signed in or the cookies have expired",
            summary["count"],
            summary["expired"],
            ", ".join(sorted(YOUTUBE_LOGIN_COOKIES)),
        )
    return True


def normalize_netscape_cookies(text: str) -> str | None:
    # Env values often lose real newlines/tabs: accept literal "\n"/"\t" and whitespace-separated fields.
    if "\n" not in text and "\\n" in text:
        text = text.replace("\\n", "\n")
    text = text.replace("\\t", "\t")

    cookie_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip("\r\n")
        if not line.strip():
            continue
        # "#HttpOnly_" lines are real cookies, other "#" lines are comments.
        if line.startswith("#") and not line.startswith("#HttpOnly_"):
            continue
        fields = line.split("\t")
        if len(fields) != NETSCAPE_COOKIE_FIELDS:
            fields = line.split()
            if len(fields) == NETSCAPE_COOKIE_FIELDS - 1:
                fields.append("")  # cookie with an empty value
            if len(fields) != NETSCAPE_COOKIE_FIELDS:
                logging.warning("Skipping malformed YouTube cookie line")
                continue
        cookie_lines.append("\t".join(fields))

    if not cookie_lines:
        logging.warning(
            "YOUTUBE_COOKIES is neither an existing file nor valid Netscape cookies, "
            "downloading without cookies"
        )
        return None

    # yt-dlp (via http.cookiejar.MozillaCookieJar) refuses files without this header.
    return NETSCAPE_COOKIE_HEADER + "\n" + "\n".join(cookie_lines) + "\n"


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

        # Route yt-dlp warnings/errors into our log (they're silenced by no_warnings otherwise),
        # e.g. "The provided YouTube account cookies are no longer valid".
        "logger": YT_DLP_LOGGER,
    }

    if is_youtube_url(url):
        cookies_text = load_youtube_cookies()
        if cookies_text:
            # yt-dlp writes the cookie jar back to `cookiefile` on exit, so each task gets
            # its own copy: parallel workers don't clobber one file and the source may be read-only.
            cookie_file = work_dir / "youtube_cookies.txt"
            cookie_file.write_text(cookies_text, encoding="utf-8")
            ydl_opts["cookiefile"] = str(cookie_file)
            logging.info(
                "Using YouTube cookies (%s cookies) for %s",
                summarize_youtube_cookies(cookies_text)["count"],
                url,
            )
        elif YOUTUBE_COOKIES.strip():
            logging.warning("YOUTUBE_COOKIES is set but unusable, downloading %s without cookies", url)

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        description = info.get("description")
        if not isinstance(description, str):
            description = None

        requested = info.get("requested_downloads") or []
        if requested and requested[0].get("filepath"):
            video_path = Path(requested[0]["filepath"])
        else:
            video_path = Path(ydl.prepare_filename(info))

    # yt-dlp has saved the cookies YouTube refreshed during the download into our copy on exit.
    if "cookiefile" in ydl_opts:
        save_youtube_cookies(Path(ydl_opts["cookiefile"]))

    return video_path, description


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
    if YOUTUBE_COOKIES.strip() and YOUTUBE_COOKIES_REFRESH_HOURS > 0:
        # Lives in the same list, so stop_video_workers() cancels it too.
        tasks.append(
            asyncio.create_task(youtube_cookies_refresher(), name="youtube-cookies-refresher")
        )
    application.bot_data[VIDEO_WORKER_TASKS_KEY] = tasks
    logging.info("Started %s video workers", VIDEO_WORKER_COUNT)


async def youtube_cookies_refresher() -> None:
    # The first run happens right at startup and doubles as an online "are we signed in" check.
    while True:
        try:
            await asyncio.to_thread(refresh_youtube_cookies)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logging.warning("YouTube cookies refresh failed: %s", exc)
        await asyncio.sleep(YOUTUBE_COOKIES_REFRESH_HOURS * 3600)


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

    log_youtube_cookies_status()

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
