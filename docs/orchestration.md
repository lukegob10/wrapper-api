# Orchestration Plan

## Flow
1. Startup: configure logging and instantiate FastAPI.
2. Request handling (/v1/completions):
   - Validate OpenAI-compatible payload; reject `stream=true` for now.
   - Call `google-genai` via `genai.Client()` using environment defaults, mapping temperature/top_p/max_tokens/stop/n to generation_config.
   - Normalize the response to OpenAI-style `id/object/choices/usage`.
3. Health: `/healthz` returns `{ "status": "ok" }` for liveness/readiness probes.

## Hardening for heavy traffic
- RequestID middleware tags every request and propagates the ID in responses; structured logging includes the request id.
- Run Uvicorn with multiple workers: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --proxy-headers`; scale workers ~1-2x vCPU.
- Place behind an L7 proxy with keep-alive and compression; autoscale on p95 latency and error rate.
- Add metrics/tracing (latency, retries) for production.

## Roo compatibility
- Endpoint and schema match OpenAI `/v1/completions` for Roo Code.
- Returns `id`, `object=text_completion`, `choices[]`, and optional `usage` block.
- Streaming flag accepted but currently disabled; add SSE if needed.

## Deployment steps
- Build container/venv with `requirements.txt`.
- Expose port 8000 (or configured) behind your API gateway for auth/rate limiting.
