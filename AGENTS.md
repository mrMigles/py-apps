# AGENTS.md

This repository is a collection of independent Telegram bots written in Python, each living in its own folder (`vkthief/`, `recap/`, ...). Bots don't share code or dependencies with each other.

## Bot folder structure

Flat, no packages:

```
<bot_name>/
  <bot_name>.py       # the whole bot in a single file
  test_<bot_name>.py
  requirements.txt
  README.md           # in Russian, the bots are Russian-speaking
```

## Rules

- No `__init__.py`, `__main__.py`, or any other `__`-prefixed files — a bot folder is just a directory, not a package. Run with: `python <bot_name>/<bot_name>.py`.
- Fewer files is better. Don't split a bot into modules/helpers/utils unless truly necessary — a single `<bot_name>.py` file is fine.
- Splitting into multiple files is only acceptable for a genuinely large feature, when a single file becomes unmanageable. Default to not splitting.
- One `test_<bot_name>.py` per bot, no `tests/` folder.
- Each bot has its own `requirements.txt` and its own `README.md` in Russian (what the bot does, environment variables, how to run it, how to run its tests).
- The root `README.md` is in English — a general overview of the repository and the list of bots.
- Tests must not hit the network/Telegram/LLM APIs — mocks/stubs only.
