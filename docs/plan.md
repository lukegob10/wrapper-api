# Roo-Compatible Completion API Plan

## Goals
- OpenAI-style `/v1/completions` endpoint for Roo Code compatibility using FastAPI + Uvicorn.
- Keep the stack minimal while still emitting request IDs and structured logs.

## Layout
- app/main.py - FastAPI app and routing
- app/internal_client.py - genai client calls and response normalization
- app/config.py - Minimal env-driven settings
- app/schemas.py - Request/response Pydantic models
- requirements.txt - Dependencies
- docs/orchestration.md - Operational flow and scaling notes
- README.md - Quickstart and configuration guide

## Configuration
- LOG_LEVEL (default: info): Logging verbosity.

## Run
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --proxy-headers
- Health: GET /healthz
- Completions: POST /v1/completions

Example request:

```
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"text-bison@001","prompt":"Say hello to Roo Code","max_tokens":64,"temperature":0.3}'
```

## Notes / Next Steps
- Add SSE streaming support when needed.
- Wire metrics/tracing and centralize secret management.
- Add safety filters, prompt limits, and per-model routing as policies firm up.
