# Orchestration Plan

## Flow
1. Startup: load env settings, initialize TokenManager, and build a pooled async HTTPX client with configured timeouts/limits.
2. Request handling (/v1/completions):
   - Validate OpenAI-compatible payload; reject `stream=true` for now.
   - Acquire Helix access token (cached; refreshes with a safety margin) using the configured command/profile.
   - If `INTERNAL_COMPLETIONS_URL` is set, POST there with the same payload + Bearer token.
   - Otherwise, if `LM_STUDIO_BASE_URL` is set, POST to `LM_STUDIO_BASE_URL/v1/completions` (OpenAI-compatible) and normalize the response.
   - Otherwise, call Vertex via `google-genai` client with configured base URL and headers, mapping temperature/top_p/max_tokens/stop/n to generation_config.
   - Retry transient failures (timeouts/5xx) up to `REQUEST_MAX_RETRIES` with backoff.
   - Normalize the upstream response to OpenAI-style `id/object/choices/usage`.
3. Health: `/healthz` returns `{ "status": "ok" }` for liveness/readiness probes.

## Hardening for heavy traffic
- HTTPX pooled async client sized via `HTTPX_MAX_CONNECTIONS` and `HTTPX_MAX_KEEPALIVE`.
- TokenManager uses a lock to avoid thundering herds during refresh.
- Timeouts enforced via `REQUEST_TIMEOUT_SECONDS`; retries on transient upstream errors.
- RequestID middleware tags every request and propagates the ID in responses; structured logging includes the request id.
- TLS trust is set via `SSL_CERT_FILE`; ensure the CA chain is mounted and readable.
- Run Uvicorn with multiple workers: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --proxy-headers`; scale workers ~1-2x vCPU.
- Place behind an L7 proxy with keep-alive and compression; autoscale on p95 latency and error rate.
- Add metrics/tracing (latency, retries, token refresh counts) for production.

## Security
- Keep the PEM out of the repo; mount it and point `HELIX_PEM_PATH` to the mounted path.
- Avoid logging tokens or sensitive prompts; ensure Helix profile has least privilege (Vertex predict only).
- Feed secrets from a secret manager rather than static env vars when possible.

## Roo compatibility
- Endpoint and schema match OpenAI `/v1/completions` for Roo Code.
- Returns `id`, `object=text_completion`, `choices[]`, and optional `usage` block.
- Streaming flag accepted but currently disabled; add SSE if needed.

## Deployment steps
- Build container/venv with `docs/requirements.txt`.
- Set required env vars (VERTEX_PROJECT, HELIX_PROFILE, PEM path, model id).
- Mount PEM at runtime with restrictive permissions.
- Expose port 8000 (or configured) behind your API gateway for auth/rate limiting.
