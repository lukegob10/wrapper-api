# Roo-Compatible Completion API Plan

## Goals
- OpenAI-style `/v1/completions` endpoint for Roo Code compatibility using FastAPI + Uvicorn.
- Wrap Google Vertex via `google-genai`, issuing Helix access tokens and honoring custom base URL/headers.
- Handle heavy traffic with connection pooling, retries, token caching, request IDs, and structured logging.

## Layout
- app/main.py - FastAPI app, routing, lifespan-managed HTTP client
- app/internal_client.py - Helix token manager and Vertex/internal clients
- app/config.py - Env-driven settings
- app/schemas.py - Request/response Pydantic models
- app/logging_utils.py - Request ID middleware + logging config
- requirements.txt - Dependencies
- docs/orchestration.md - Operational flow and scaling notes
- README.md - Quickstart and configuration guide

## Configuration
- VERTEX_PROJECT (required): GCP project id for Vertex AI.
- VERTEX_LOCATION (default: us-central1): Region for the model.
- VERTEX_MODEL (default: text-bison@001): Vertex publisher model id.
- VERTEX_BASE_URL (optional): Override Vertex gateway base URL (e.g., via proxy).
- LM_STUDIO_BASE_URL (optional): Local LM Studio OpenAI-compatible gateway for pre-deployment testing.
- HELIX_PROFILE (optional): Helix CLI profile to issue tokens (appended to the command when set).
- HELIX_ACCESS_TOKEN_CMD (default: helix auth access-token print -a): Command used to fetch access tokens.
- HELIX_TOKEN_TTL (default: 600): Token lifetime in seconds.
- HELIX_REFRESH_MARGIN (default: 60): Seconds before expiry to refresh.
- SSL_CERT_FILE (default: CAChain_PROD.pem): Cert bundle exported for TLS to Vertex gateway.
- CUSTOM_USER_ID (default: $USER) / CUSTOM_HEADER_NAME (default: x-custom-userid): Metadata header for Vertex gateway.
- REQUEST_TIMEOUT_SECONDS (default: 30.0): Upstream timeout per request.
- REQUEST_MAX_RETRIES (default: 2): Retries for upstream POSTs.
- HTTPX_MAX_CONNECTIONS (default: 100) / HTTPX_MAX_KEEPALIVE (default: 20): Connection pool sizing.
- INTERNAL_COMPLETIONS_URL (optional): Forward requests to this service when set.
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
