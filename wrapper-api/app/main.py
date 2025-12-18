from __future__ import annotations

import anyio
import uvicorn
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from .internal_client import (
    call_openai_chat_completion,
    call_openai_completion,
    stream_openai_chat_completion,
    stream_openai_completion,
)
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    HealthStatus,
    ModelCard,
    ModelListResponse,
)

APP_NAME = "roo-openai-compatible-wrapper"

DEFAULT_MODELS = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "text-bison@001",
]

app = FastAPI(title=APP_NAME, version="0.2.0")


def _extract_api_key(
    authorization: str | None,
    x_api_key: str | None,
    x_goog_api_key: str | None,
) -> str | None:
    if x_api_key:
        return x_api_key
    if x_goog_api_key:
        return x_goog_api_key
    if not authorization:
        return None
    parts = authorization.strip().split()
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1]:
        return parts[1]
    return None


@app.get("/healthz", response_model=HealthStatus)
async def healthcheck() -> HealthStatus:
    """Liveness/readiness probe."""
    return HealthStatus(status="ok")


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    return ModelListResponse(data=[ModelCard(id=model_id) for model_id in DEFAULT_MODELS])


@app.get("/v1/models/{model_id}", response_model=ModelCard)
async def get_model(model_id: str) -> ModelCard:
    return ModelCard(id=model_id)


@app.post(
    "/v1/completions",
    response_model=CompletionResponse,
    status_code=status.HTTP_200_OK,
)
async def create_completion(
    payload: CompletionRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    x_goog_api_key: str | None = Header(default=None, alias="x-goog-api-key"),
) -> CompletionResponse | StreamingResponse:
    """OpenAI-style completions endpoint for Roo compatibility."""
    api_key = _extract_api_key(authorization, x_api_key, x_goog_api_key)
    if payload.stream:
        return StreamingResponse(
            stream_openai_completion(payload, api_key=api_key),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
        )

    return await anyio.to_thread.run_sync(lambda: call_openai_completion(payload, api_key=api_key))


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    status_code=status.HTTP_200_OK,
)
async def create_chat_completion(
    payload: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    x_goog_api_key: str | None = Header(default=None, alias="x-goog-api-key"),
) -> ChatCompletionResponse | StreamingResponse:
    api_key = _extract_api_key(authorization, x_api_key, x_goog_api_key)
    if payload.stream:
        return StreamingResponse(
            stream_openai_chat_completion(payload, api_key=api_key),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
        )

    return await anyio.to_thread.run_sync(lambda: call_openai_chat_completion(payload, api_key=api_key))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
