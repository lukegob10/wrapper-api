# Roo-Compatible Completion API

FastAPI + Uvicorn wrapper that exposes an OpenAI-style `/v1/completions` endpoint backed by Google Vertex AI (or an internal completion endpoint) with Helix-issued tokens. Designed for Roo Code compatibility and production hardening (connection pooling, retries, cached tokens, request IDs).

## Features
- OpenAI-compatible request/response schema (`/v1/completions`, `/healthz`).
- Vertex AI calls via `google-genai` client (configurable base URL + headers) with stop sequences, candidate count, and safety timeouts.
- Helix access-token issuance (`helix auth access-token print -a` by default) with caching; supports custom profile and TTL/refresh window.
- Request ID middleware and structured logging with env-configurable log level.
- Pooled `httpx.AsyncClient` managed via FastAPI lifespan for safe startup/shutdown.

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
- `VERTEX_PROJECT` (required when using Vertex): GCP project id for Vertex AI.
- `VERTEX_LOCATION` (default: `us-central1`): Region for the model.
- `VERTEX_MODEL` (default: `text-bison@001`): Vertex model id.
- `VERTEX_BASE_URL` (optional): Override base URL for Vertex gateway (e.g., `https://url.net/vertex`).
- `LM_STUDIO_BASE_URL` (optional): When set, route completions to a local LM Studio OpenAI-compatible server (e.g., `http://localhost:1234`).
- `HELIX_PROFILE` (optional): Helix CLI profile to issue tokens (appended to the access-token command if set).
- `HELIX_ACCESS_TOKEN_CMD` (default: `helix auth access-token print -a`): Command used to fetch an access token.
- `HELIX_TOKEN_TTL` (default: `600`), `HELIX_REFRESH_MARGIN` (default: `60`): Token lifetime/refresh window.
- `SSL_CERT_FILE` (default: `CAChain_PROD.pem`): Cert bundle path exported for TLS to the Vertex gateway.
- `CUSTOM_USER_ID` (default: `$USER`), `CUSTOM_HEADER_NAME` (default: `x-custom-userid`): Header/value sent to the Vertex gateway.
- `INTERNAL_COMPLETIONS_URL` (optional): If set, forward to this internal API instead of Vertex.
- `REQUEST_TIMEOUT_SECONDS` (default: `30.0`), `REQUEST_MAX_RETRIES` (default: `2`): Upstream call protection.
- `HTTPX_MAX_CONNECTIONS` (default: `100`), `HTTPX_MAX_KEEPALIVE` (default: `20`): Connection pool sizing.
- `LOG_LEVEL` (default: `info`): Logging verbosity.

## Operational notes
- Mount the PEM at runtime; keep it out of source control.
- Run behind an L7 proxy with keep-alives and rate limiting; scale Uvicorn workers ~1–2× vCPU.
- Add observability (metrics/tracing) and SSE streaming when you need real-time tokens.
