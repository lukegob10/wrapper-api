from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status

from .config import Settings
from .internal_client import (
    TokenManager,
    call_vertex_completion,
    forward_to_internal_completion,
    call_lm_studio_completion,
)
from .logging_utils import RequestIDMiddleware, configure_logging
from .schemas import CompletionRequest, CompletionResponse, HealthStatus

settings = Settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)
token_manager = TokenManager(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared HTTP client and close it cleanly on shutdown."""
    timeout = httpx.Timeout(settings.request_timeout)
    limits = httpx.Limits(
        max_connections=settings.pool_connections,
        max_keepalive_connections=settings.pool_keepalive,
    )
    http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
    app.state.http_client = http_client
    logger.info("startup: http client ready", extra={"max_connections": settings.pool_connections})
    try:
        yield
    finally:
        await http_client.aclose()
        logger.info("shutdown: http client closed")


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)


async def get_http_client(request: Request) -> httpx.AsyncClient:
    """Dependency that provides the shared AsyncClient instance."""
    client = getattr(request.app.state, "http_client", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="HTTP client not ready",
        )
    return client


@app.get("/healthz", response_model=HealthStatus)
async def healthcheck() -> HealthStatus:
    """Liveness/readiness probe."""
    return HealthStatus(status="ok")


@app.post(
    "/v1/completions",
    response_model=CompletionResponse,
    status_code=status.HTTP_200_OK,
)
async def create_completion(
    payload: CompletionRequest,
    client: httpx.AsyncClient = Depends(get_http_client),
) -> CompletionResponse:
    """OpenAI-style completions endpoint for Roo compatibility."""
    logger.info(
        "completions.request",
        extra={"model": payload.model, "stream": payload.stream, "n": payload.n},
    )

    if payload.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stream=true is not yet supported",
        )

    if settings.internal_endpoint:
        return await forward_to_internal_completion(
            request=payload,
            settings=settings,
            client=client,
            token_manager=token_manager,
        )

    if settings.lm_studio_base_url:
        return await call_lm_studio_completion(
            request=payload,
            settings=settings,
            client=client,
        )

    return await call_vertex_completion(
        request=payload,
        settings=settings,
        client=client,
        token_manager=token_manager,
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
