# Looma

Convert a YouTube link or an uploaded video into structured, reusable
knowledge — refined title, 3–5 line summary, 5–10 key insights, chapter
markers, and an audio-friendly narration. Looma is **not** a blind
transcriber: the LLM extraction step filters filler and reorganizes content
for both reading and listening.

> **Demo:** drop your own demo screenshot at `docs/demo.png` and the path
> above (and in the badge below) will resolve to a real image in the
> repo. The placeholder file currently checked in is a 1×1 transparent
> PNG so the link doesn't 404 on a fresh clone.
>
> ![Looma demo placeholder](docs/demo.png)

## Features
- **Two ingest paths:** paste a YouTube URL, or upload `.mp4 / .mov / .mkv /
  .webm` (≤200 MB, ≤90 minutes).
- **Whisper transcription** with configurable model size (`tiny|base|small|
  medium|large`, default `small`).
- **LLM extraction** (Anthropic Claude by default, OpenAI as fallback) that
  returns a strict JSON object with title, summary, insights, chapters, and
  a filler-free narrative.
- **Edge TTS narration** (default, free) or OpenAI TTS (opt-in) producing a
  playable MP3 bound to `GET /audio/{job_id}.mp3`.
- **Job history** persisted in SQLite (`data/jobs.db`); list, fetch, and
  delete via the API.
- **Single-page UI** with tabbed input, live stage progress, audio player
  with clickable chapter timestamps, and a "Copy as Markdown" button.
- **Canonical error responses** — every non-2xx body uses
  `{"error": "<msg>", "code": "<machine_code>"}` with a status from the
  allow-list `200 / 400 / 404 / 413 / 415 / 500`. JS clients can rely on
  this shape without sniffing the status code (see [Error Shape](#error-shape)).

## Prerequisites
- **Python 3.11+** (tested with 3.11 and 3.12). The app's startup
  guard verifies that `ffmpeg` and at least one LLM API key are
  available; if either is missing the process exits with a
  one-line actionable error.
- **`ffmpeg`** on `$PATH` (install with `sudo apt-get install -y ffmpeg`
  on Debian/Ubuntu). `ffprobe` is also expected; the install
  command above provides both.
- At least one LLM API key — **Anthropic** (`ANTHROPIC_API_KEY`) or
  **OpenAI** (`OPENAI_API_KEY`). The startup guard refuses to launch
  without one.
- A modern browser to use the UI at `http://127.0.0.1:8000/`.

## Quick Start
```bash
# 1. Clone and enter the project
cd looma

# 2. Copy the env template and fill in at least one API key
cp .env.example .env
$EDITOR .env

# 3. Create a virtualenv and install dependencies
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Verify your environment
python -c "import whisper, yt_dlp, fastapi, edge_tts, anthropic, openai; print('ok')"
ffmpeg -version | head -1

# 5. Run the app (from the repo root)
cd ..
bash run.sh
# or, explicitly:
# uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open <http://127.0.0.1:8000/> in a browser.

## Install in detail
The first time you set up Looma on a fresh machine:

1. **System packages (Debian/Ubuntu):**
   ```bash
   sudo apt-get update
   sudo apt-get install -y ffmpeg python3.11 python3.11-venv
   ```
2. **Python venv + deps:**
   ```bash
   cd backend
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Optional**: bake a specific Whisper model into the venv by
   pre-downloading it (default `small` ~ 460 MB):
   ```python
   import whisper; whisper.load_model("small")
   ```
   Without this, the first request pays a one-time model load.

The first run on a clean clone will:
- Create `data/audio/`, `data/outputs/`, and `data/jobs.db`.
- Download the chosen Whisper model on first use.
- Print `INFO: Uvicorn running on http://127.0.0.1:8000` when ready.

## Environment Variables
See `.env.example` for the full list. The defaults are sensible on
Linux/macOS; the `MAX_*` and `DATA_DIR` settings are the ones you'll
most often tweak.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `ANTHROPIC_API_KEY` | one of | — | Enables the Anthropic LLM provider. |
| `OPENAI_API_KEY` | one of | — | Enables the OpenAI LLM (or TTS) provider. |
| `LLM_PROVIDER` | no | `anthropic` | `anthropic` or `openai`. |
| `WHISPER_MODEL` | no | `small` | `tiny\|base\|small\|medium\|large`. |
| `TTS_PROVIDER` | no | `edge` | `edge` (free) or `openai`. |
| `TTS_VOICE` | no | `en-US-AriaNeural` | Edge voice name. |
| `MAX_VIDEO_SECONDS` | no | `5400` | Reject videos longer than 90 min. |
| `MAX_UPLOAD_MB` | no | `200` | Reject uploads larger than 200 MB. |
| `MAX_PIPELINE_SECONDS` | no | `300` | Hard cap on wall-clock per job; the orchestrator logs a WARNING when a run exceeds this. |
| `DATA_DIR` | no | `./data` | Where SQLite + MP3s live. |
| `FRONTEND_DIR` | no | `<repo>/frontend` | Path the static-files mount serves at `/`. |
| `HOST` | no | `127.0.0.1` | Bind address (consumed by `run.sh`). |
| `PORT` | no | `8000` | Bind port (consumed by `run.sh`). |

At least one of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` must be set;
the process refuses to start otherwise.

## Run
After installing, two equivalent ways to launch the dev server:

```bash
# Option A: helper script (loads .env, activates venv)
bash run.sh                # http://127.0.0.1:8000
bash run.sh --reload       # with uvicorn --reload

# Option B: invoke uvicorn directly (so you can set extra flags)
cd backend && source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Once running, the following are live:

| URL | What it serves |
| --- | --- |
| `http://127.0.0.1:8000/` | Single-page UI. |
| `http://127.0.0.1:8000/healthz` | Liveness probe (always 200). |
| `http://127.0.0.1:8000/api/jobs?limit=20` | Most recent 20 jobs. |
| `http://127.0.0.1:8000/docs` | Auto-generated OpenAPI / Swagger UI. |
| `http://127.0.0.1:8000/redoc` | Auto-generated ReDoc UI. |

## API Reference
All non-2xx responses use the canonical error shape (see
[Error Shape](#error-shape) below); the table below is the success-path
contract.

| Method | Path | Body | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/extract` | `{"youtube_url": "..."}` *or* `multipart/form-data` with `file` | Run the four-stage pipeline; returns a `LoomaResult`. |
| `GET` | `/api/jobs?limit=20` | — | Most recent `limit` jobs (1–200, default 20). |
| `GET` | `/api/jobs/{job_id}` | — | A single job row, or 404 NOT_FOUND. |
| `DELETE` | `/api/jobs/{job_id}` | — | Delete the job row and its on-disk MP3s. |
| `GET` | `/audio/{job_id}.mp3` | — | Stream the TTS narration MP3 (supports HTTP Range). |
| `GET` | `/healthz` | — | Liveness probe. |

### Error Shape
Every non-2xx response from the API is shape-stable:
```json
{"error": "human-readable message", "code": "MACHINE_CODE"}
```
The HTTP status is one of `200 / 400 / 404 / 413 / 415 / 500`. The
`Content-Type` is always `application/json` so JS clients can `resp.json()`
without sniffing. Common machine codes:

| Code | Status | When |
| --- | --- | --- |
| `INVALID_URL` | 400 | YouTube URL is malformed or has a non-YouTube host. |
| `UNSUPPORTED_SOURCE` | 400 | URL parses but its host isn't `youtube.com` / `youtu.be`. |
| `UNSUPPORTED_MEDIA` | 415 | Upload extension isn't `.mp4`/`.mov`/`.mkv`/`.webm`. |
| `PAYLOAD_TOO_LARGE` | 413 | Upload > `MAX_UPLOAD_MB` *or* resulting MP3 > 50 MB. |
| `NOT_FOUND` | 404 | Unknown route, missing static asset, or unknown `job_id`. |
| `LLM_SCHEMA_ERROR` | 500 | LLM returned a payload that failed Pydantic validation twice. |
| `TTS_FAILED` | 500 | TTS provider raised. |
| `TRANSCRIPTION_FAILED` | 500 | Whisper raised. |
| `DOWNLOAD_FAILED` | 500 | `yt-dlp` could not fetch the video. |
| `INTERNAL_ERROR` | 500 | Any other uncaught failure (defense-in-depth 500 handler). |

## Smoke Test
```bash
# After running bash run.sh in another terminal:
curl -X POST http://127.0.0.1:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{"youtube_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

A successful response carries a `job_id` and an `audio_url`; the
narration MP3 is then fetchable at `http://127.0.0.1:8000<audio_url>`.

To verify the canonical error shape:
```bash
# 400 — non-YouTube host
curl -i -X POST http://127.0.0.1:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{"youtube_url":"https://vimeo.com/123"}'
# -> {"error":"Host 'vimeo.com' is not a YouTube domain. ...","code":"UNSUPPORTED_SOURCE"}
```

## Performance
The orchestrator instruments every stage with `time.perf_counter` and
surfaces the per-stage durations on `LoomaResult.timings`. The hard
budget is `MAX_PIPELINE_SECONDS` (default 300 s) — a 20-minute YouTube
source should finish inside 5 minutes on a modern laptop. The
expected per-stage split is:

| Stage | Soft budget | What it does |
| --- | --- | --- |
| `ingest` | 60 s | `yt-dlp` audio extraction (YouTube) **or** `ffmpeg` normalize (upload). |
| `transcribe` | 240 s | Whisper `small` on a 20-min source (~3-9× realtime). |
| `extract` | 30 s | LLM call + 1 retry on schema failure. |
| `narrate` | 60 s | Edge TTS synthesis of a 150-400 word narration. |
| **total** | **300 s** | Hard cap; WARNING logged on overrun. |

Soft budgets are diagnostic only — a run that exceeds the per-stage
budget but stays inside the 300 s total is still a successful job.

## Tests
```bash
cd backend
source .venv/bin/activate
pytest tests/ -v --tb=short
```

The test suite covers:
- URL validation (`tests/test_ingest.py`)
- File conversion via `ffmpeg` (`tests/test_ingest.py`)
- Transcript parsing (`tests/test_transcribe.py`)
- LLM schema with a recorded fixture (`tests/test_extract.py`)
- TTS file creation (`tests/test_narrate.py`)
- API endpoints via `httpx.AsyncClient` (`tests/test_api_*.py`)
- Storage layer — SQLite jobs DB and on-disk MP3 helpers
  (`tests/test_storage_*.py`)
- AC-10 perf budget and `DELETE /api/jobs/{job_id}` contract
  (`tests/test_performance_ac10.py`)
- AC-11 canonical error shape (`tests/test_error_handling_ac11.py`)

End-to-end ASGI smoke tests do not require a live Whisper model,
LLM key, or ffmpeg binary; every pipeline stage is patched via
`unittest.mock` so the suite runs in CI on a stock GitHub Actions
runner.

## Deployment
Looma v1 is a Python FastAPI app with a local SQLite database and the
Whisper model loaded in-process. It is **not** a fit for one-click
Vercel/Cloudflare Pages deploys — it needs `ffmpeg`, a multi-GB native
Whisper dependency, and a writable filesystem for the SQLite file. Run
it on a long-lived host (laptop, VPS, Fly.io machine) instead.

To containerize for production:
- Base image: `python:3.11-slim` with `ffmpeg` installed.
- Bake the chosen Whisper model into the image to avoid cold-start latency.
- Mount a persistent volume at `DATA_DIR` so `data/jobs.db` and the MP3
  outputs survive restarts.
- The startup guard already validates `ffmpeg` + LLM key; a crash-loop
  on a misconfigured image surfaces a clear error in the first
  container log line.

## Project Layout
```
looma/
├── backend/
│   ├── app/
│   │   ├── main.py, config.py, models.py
│   │   ├── pipeline/{ingest,transcribe,extract,narrate,orchestrator}.py
│   │   ├── storage/{jobs,files}.py
│   │   └── prompts/{extract_system,extract_user}.txt
│   ├── tests/         # pytest suite
│   ├── requirements.txt
│   └── .env.example
├── frontend/{index.html, styles.css, app.js}
├── data/              # runtime: jobs.db, audio/, outputs/
├── docs/
│   ├── demo.png       # drop your demo screenshot here
│   ├── plans/
│   └── references/
├── .env.example
├── README.md
└── run.sh
```

See `claude.md` for the full repository layout and coding standards.

## License
Single-operator internal tool — pick a license before publishing.
