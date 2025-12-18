# Orchestration Plan

## Flow
1. Startup: configure logging and instantiate FastAPI.
2. Request handling:
   - `GET /v1/models`: return a basic OpenAI-compatible model list.
   - `POST /v1/chat/completions` (preferred) and `POST /v1/completions` (legacy):
     - Validate OpenAI-compatible payload; ignore unknown fields.
     - If provided, treat `Authorization: Bearer ...` (or `x-goog-api-key`) as the `google-genai` API key.
     - Map temperature/top_p/max_tokens/stop/n to `google.genai.types.GenerationConfig`.
     - Normalize responses to OpenAI-style objects (`chat.completion`, `text_completion`).
     - If `stream=true`, return SSE (`text/event-stream`) and terminate with `data: [DONE]`.
3. Health: `/healthz` returns `{ "status": "ok" }` for liveness/readiness probes.

## Hardening for heavy traffic
- Run Uvicorn with multiple workers: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --proxy-headers`; scale workers ~1-2x vCPU.
- Place behind an L7 proxy with keep-alive and compression; autoscale on p95 latency and error rate.
- Add metrics/tracing (latency, retries) for production.

## Roo compatibility
- Implements `POST /v1/chat/completions` for modern OpenAI-compatible clients (Roo/Cline).
- Keeps `POST /v1/completions` for legacy clients.
- Supports SSE streaming with `stream=true`.

## Deployment steps
- Build container/venv with `requirements.txt`.
- Expose port 8000 (or configured) behind your API gateway for auth/rate limiting.
