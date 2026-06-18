---
title: Looma
emoji: 🎯
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# Looma

Convert a YouTube link or an uploaded video into structured, reusable
knowledge — refined title, 3–5 line summary, 5–10 key insights, chapter
markers, and an audio-friendly narration. Looma is **not** a blind
transcriber: the LLM extraction step filters filler and reorganizes content
for both reading and listening.

> **Deployed on Hugging Face Spaces** — try it at
> [Ethan0103/Looma](https://huggingface.co/spaces/Ethan0103/Looma).

## Features
- **Two ingest paths:** paste a YouTube URL, or upload `.mp4 / .mov / .mkv /
  .webm` (≤200 MB, ≤90 minutes).
- **Whisper transcription** with configurable model size (`tiny|base|small|
  medium|large`, default `medium` on HF Spaces).
- **LLM extraction** (Anthropic Claude by default, OpenAI as fallback) that
  returns a strict JSON object with title, summary, insights, chapters, and
  a filler-free narrative.
- **Edge TTS narration** (default, free) or OpenAI TTS (opt-in) producing a
  playable MP3 bound to `GET /audio/{job_id}.mp3`.
- **Single-page UI** with tabbed input, live stage progress, audio player
  with clickable chapter timestamps, and a "Copy as Markdown" button.
- **Canonical error responses** — every non-2xx body uses
  `{"error": "<msg>", "code": "<machine_code>"}`.

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
```

Then open <http://127.0.0.1:8000/> in a browser.

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
| `DATA_DIR` | no | `./data` | Where MP3s live. |
| `HOST` | no | `127.0.0.1` | Bind address. |
| `PORT` | no | `8000` | Bind port. |

At least one of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` must be set;
the process refuses to start otherwise.

## Run
After installing, two equivalent ways to launch the dev server:

```bash
# Option A: helper script (loads .env, activates venv)
bash run.sh                # http://127.0.0.1:8000
bash run.sh --reload       # with uvicorn --reload

# Option B: invoke uvicorn directly
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

## Project Layout
```
looma/
├── backend/
│   ├── app/
│   │   ├── main.py, config.py, models.py
│   │   ├── pipeline/  # {ingest,transcribe,extract,narrate,orchestrator}.py
│   │   ├── storage/   # {jobs,files}.py
│   │   └── prompts/   # LLM system/user prompts
│   ├── tests/         # pytest suite
│   ├── requirements.txt
│   └── .env.example
├── frontend/          # {index.html, styles.css, app.js}
├── Dockerfile         # HF Spaces / Docker deployment
├── data/              # runtime: audio/, outputs/
└── docs/
```

## License
Single-operator internal tool — pick a license before publishing.
