# Python Bots

A small collection of independent, Russian-language Telegram bots. Each application lives in its own directory and includes its own documentation, dependencies, and test suite.

## Applications

- [`vkthief`](vkthief/) — downloads short videos from VK, YouTube Shorts, TikTok, and Rutube and sends them to a Telegram chat.
- [`recap`](recap/) — keeps a rolling in-memory chat history, transcribes voice messages, and creates Russian-language chat recaps through an OpenAI-compatible API. Optionally stores full message history in PostgreSQL/pgvector, indexes it into semantically coherent chunks, and provides `/search` for hybrid vector + lexical retrieval. Includes `/init` to import Telegram Desktop JSON exports.

## Quick start

Use Python 3.10 or newer. Install and run each bot independently:

```bash
pip install -r vkthief/requirements.txt
python vkthief/vkthief.py
```

```bash
pip install -r recap/requirements.txt
python recap/recap.py
```

The required environment variables and operational notes are documented in each application's Russian README.

## Tests

After installing both requirement files, run all unit tests from the repository root:

```bash
pytest
```

The test suites mock external integrations and do not require Telegram or OpenAI credentials.