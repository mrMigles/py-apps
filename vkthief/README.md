# VKThief Bot

Русскоязычный Telegram-бот скачивает короткие видео по ссылке и отправляет их обратно файлом. Поддерживаются VK Clips/Video, YouTube Shorts, TikTok и Rutube Shorts. Скачивание выполняет `yt-dlp`, задачи обрабатываются параллельными воркерами, но сообщения одного чата сохраняют порядок.

Instagram Reels пока не поддерживается фактически, хотя сервис упоминается в приветствии.

## Установка и запуск

Требуется Python 3.10 или новее. Для некоторых роликов `yt-dlp` также нужен установленный в системе `ffmpeg`.

```bash
cd vkthief
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Переменные окружения:

- `BOT_TOKEN` — токен Telegram-бота, обязателен.
- `MAX_FILE_MB` — максимальный размер отправляемого файла, по умолчанию `49`.
- `VIDEO_WORKER_COUNT` — количество параллельных воркеров, по умолчанию `10`.
- `YOUTUBE_COOKIES` — необязательные cookies YouTube, если YouTube требует входа («Sign in to confirm you're not a bot»). Можно указать путь к файлу `cookies.txt` или само содержимое файла. Подробности ниже.

### Cookies для YouTube

`yt-dlp` принимает cookies только в формате Netscape (`cookies.txt`), JSON не подходит. Как их получить:

1. Откройте приватное окно браузера, войдите в YouTube (лучше с отдельного, не основного аккаунта).
2. В этой же вкладке откройте `https://www.youtube.com/robots.txt` и экспортируйте cookies для `youtube.com` расширением вроде «Get cookies.txt LOCALLY» (или `yt-dlp --cookies-from-browser chrome --cookies cookies.txt`).
3. Сразу закройте приватное окно: иначе YouTube обновит cookies в браузере, и экспортированные перестанут работать.

Значение `YOUTUBE_COOKIES`:

- **путь к файлу**, например `YOUTUBE_COOKIES=/secrets/youtube_cookies.txt`. Файл перечитывается при каждом скачивании, поэтому его можно обновить без перезапуска бота;
- **содержимое файла**, например из секрета Docker/CI. Заголовок `# Netscape HTTP Cookie File` бот добавит сам, если его нет. Переносы строк в виде `\n` и табуляции, превратившиеся в пробелы, тоже допускаются.

Cookies передаются в `yt-dlp` только для ссылок YouTube: `youtube.com` и его поддомены `www.`, `m.`, `music.`, а также `youtu.be` и `youtube-nocookie.com`. Cookies в экспорте стоят на домене `.youtube.com`, поэтому одного файла хватает для всех этих адресов. Сайты кроме YouTube скачиваются без cookies.

Как понять, что cookies подхватились, — смотрите лог:

- при старте `YouTube cookies loaded successfully: N cookies, ...` — всё хорошо;
- `... but there are no valid login cookies` — файл прочитан, но вход в аккаунт не выполнен или cookies истекли;
- `YOUTUBE_COOKIES file not found` или `... neither an existing file nor valid Netscape cookies` — неверный путь или формат;
- при скачивании с YouTube — `Using YouTube cookies (N cookies) for <ссылка>`;
- `The provided YouTube account cookies are no longer valid` — предупреждение самого `yt-dlp`: YouTube отверг cookies, их нужно выгрузить заново.

`yt-dlp` при завершении перезаписывает файл cookies, поэтому бот для каждой задачи делает отдельную копию во временной папке. Так параллельные воркеры не портят один файл, а исходный файл может быть доступен только для чтения.

Из корня репозитория:

```bash
python vkthief/vkthief.py
```

Отправьте боту `/start`, затем ссылку на поддерживаемое видео.

## Тесты

```bash
pytest vkthief
```

Тесты используют заглушки и не скачивают видео из интернета.
