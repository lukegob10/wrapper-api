# Roo-Compatible Completion API

FastAPI + Uvicorn wrapper that exposes an OpenAI-style `/v1/completions` endpoint. Designed for Roo Code compatibility and minimal plumbing: accepts a request, forwards to `google-genai` (`genai.Client()`), and returns a normalized response.

## Features
- OpenAI-compatible request/response schema (`/v1/completions`, `/healthz`).
- Simple `google-genai` client path using environment-based defaults.
- Request ID middleware and structured logging with env-configurable log level.
- Minimal stack without extra HTTP client layers.

## Quickstart
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --proxy-headers
```

Health check:
```
curl -s http://localhost:8000/healthz
```

Completion:
```
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"text-bison@001","prompt":"Say hello to Roo Code","max_tokens":64,"temperature":0.3}'
```

## Configuration (env vars)
- `APP_NAME` (optional): FastAPI title.
- `LOG_LEVEL` (default: `info`): Logging verbosity.

## Operational notes
- Mount the PEM at runtime; keep it out of source control.
- Run behind an L7 proxy with keep-alives and rate limiting; scale Uvicorn workers ~1–2× vCPU.
- Add observability (metrics/tracing) and SSE streaming when you need real-time tokens.
