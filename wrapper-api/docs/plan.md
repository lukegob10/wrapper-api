# Roo-Compatible Completion API Plan

## Goals
- OpenAI-compatible API for Roo/Cline using FastAPI + Uvicorn.
- Support both chat (`/v1/chat/completions`) and legacy (`/v1/completions`) flows, including SSE streaming.

## Layout
- app/main.py - FastAPI app and routing
- app/internal_client.py - genai client calls and response normalization
- app/schemas.py - Request/response Pydantic models
- requirements.txt - Dependencies
- docs/orchestration.md - Operational flow and scaling notes
- README.md - Quickstart and configuration guide

## Run
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --proxy-headers
- Health: GET /healthz
- Models: GET /v1/models
- Chat: POST /v1/chat/completions
- Completions (legacy): POST /v1/completions

Example request:

```
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-1.5-flash","messages":[{"role":"user","content":"Say hello to Roo Code"}],"max_tokens":256}'
```

## Notes / Next Steps
- Wire metrics/tracing and centralize secret management.
- Add safety filters, prompt limits, and per-model routing as policies firm up.
