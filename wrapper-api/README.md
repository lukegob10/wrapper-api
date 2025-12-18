# Roo OpenAI-Compatible Wrapper API

FastAPI + Uvicorn wrapper that exposes OpenAI-compatible endpoints for Roo/Cline:
- `POST /v1/chat/completions` (recommended)
- `POST /v1/completions` (legacy)
- `GET /v1/models`

## Features
- OpenAI-compatible request/response schema (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/healthz`).
- SSE streaming support (`stream=true`) for chat + completions.
- Optional API key passthrough: uses `Authorization: Bearer ...` (or `x-goog-api-key`) as the `google-genai` API key.

## Quickstart
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --proxy-headers
```

Or:
```
python main.py --reload
```

Health check:
```
curl -s http://localhost:8000/healthz
```

Chat completion:
```
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GOOGLE_API_KEY" \
  -d '{"model":"gemini-1.5-flash","messages":[{"role":"user","content":"Say hello to Roo Code"}],"max_tokens":256}'
```

Streaming (SSE):
```
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GOOGLE_API_KEY" \
  -d '{"model":"gemini-1.5-flash","messages":[{"role":"user","content":"Stream a short haiku"}],"stream":true}'
```

## Operational notes
- Run behind an L7 proxy with keep-alives and rate limiting; scale Uvicorn workers ~1–2× vCPU.
- Add observability (metrics/tracing) as needed.
