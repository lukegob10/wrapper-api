from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, HTTPException, status

from .internal_client import call_vertex_completion
from .schemas import CompletionRequest, CompletionResponse, HealthStatus

APP_NAME = "roo-compatible-completions-api"
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

app = FastAPI(title=APP_NAME, version="0.1.0")


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
) -> CompletionResponse:
    """OpenAI-style completions endpoint for Roo compatibility."""
    if payload.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stream=true is not yet supported",
        )

    return await call_vertex_completion(
        request=payload,
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
