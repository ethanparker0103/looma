# Looma — Project Standards

## Purpose
Looma converts a YouTube link or an uploaded video into a refined title, 3–5
line summary, 5–10 key insights, chapter markers, and an audio-friendly
narration. It is a single-operator, single-machine tool — no auth, no
multi-tenancy, no cloud deployment in v1.

## Tech Stack
- **Backend:** Python 3.11+, FastAPI, Uvicorn.
- **Speech-to-text:** `openai-whisper` (default model: `small`).
- **Audio ingest:** `yt-dlp` for YouTube, `ffmpeg` for upload conversion.
- **LLM:** Anthropic Claude (default) or OpenAI GPT (fallback). Strict JSON
  output validated with Pydantic.
- **TTS:** Edge TTS (default) or OpenAI TTS.
- **Storage:** SQLite at `data/jobs.db`; MP3s at `data/audio/` and
  `data/outputs/`.
- **Frontend:** static HTML/CSS/JS served by FastAPI from `/` — no build
  step, no Node toolchain.

## Repository Layout
```
looma/
├── backend/
│   ├── app/
│   │   ├── main.py, config.py, models.py
│   │   ├── pipeline/{ingest,transcribe,extract,narrate,orchestrator}.py
│   │   ├── storage/{jobs,files}.py
│   │   └── prompts/{extract_system,extract_user}.txt
│   ├── tests/  (conftest, test_ingest, test_transcribe, test_extract,
│   │             test_narrate, test_orchestrator, test_api, fixtures/)
│   ├── requirements.txt, pyproject.toml, .env.example
├── frontend/{index.html, styles.css, app.js}
├── data/{jobs.db, audio/, outputs/}    # runtime
├── docs/references/
├── .env.example, .gitignore, README.md, run.sh, claude.md
```

## Coding Standards
- **Type hints everywhere.** Use `from __future__ import annotations` only if it
  speeds up module import without breaking runtime introspection.
- **Pydantic v2** for all input/output schemas (`BaseModel`, `Field`).
- **Async-first**: I/O-bound work in `async def`; CPU-bound Whisper and TTS
  calls wrapped in `asyncio.to_thread`.
- **Strict JSON contract** with the LLM — system prompt demands no prose, no
  markdown fences. Pydantic re-validates; retry once on schema failure.
- **Errors** use shape `{"error": "<msg>", "code": "<MACHINE_CODE>"}` with
  proper HTTP codes (200 / 400 / 404 / 413 / 415 / 500). See `models.py`.
- **Logging:** prefer `logging.getLogger(__name__)` over `print`.
- **Tests:** `pytest` with `pytest-asyncio`. Aim for unit + integration
  coverage; at least 10 tests passing before declaring done.

## Workflow
1. Read `.humanize/rlcr/<session>/goal-tracker.md` at the start of each round
   to know the current target AC and what's still pending.
2. Implement only the active AC. Keep diffs small and focused.
3. Run the narrowest meaningful verification (imports, smoke CLI, a single
   pytest marker). Don't run the full suite unless the change crosses module
   boundaries.
4. Update the goal tracker: move the AC to "done" once verified, or document
   blockers in Open Issues.
5. Update `README.md` whenever setup, env vars, or operator workflow changes.

## Environment & Secrets
- `.env` is gitignored. `.env.example` is the source of truth for variables.
- At least one of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` MUST be set;
  `app.main` exits with a single-line error otherwise (AC-14).
- `ffmpeg` MUST be on `$PATH`; same fail-fast behavior.

## Forbidden Patterns
- No `print()` for diagnostics in production code paths (use `logging`).
- No `pkill` / `killall` against broad process groups (use targeted PIDs).
- No global mutable state across requests.
- No shell calls with `shell=True`; build argv lists instead.
- No large model checkpoints or generated audio committed to git.
