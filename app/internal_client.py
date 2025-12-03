from __future__ import annotations

import asyncio
import os
import secrets
import shlex
import time
from typing import Any, Dict, List, Sequence, Tuple

import httpx
from fastapi import HTTPException, status
from google import genai
from google.genai import types as genai_types
from google.oauth2.credentials import Credentials
from pydantic import ValidationError

from .config import Settings
from .schemas import Choice, CompletionRequest, CompletionResponse, Usage


class TokenManager:
    """Caches Helix access tokens to avoid reissuing on every request."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def _issue_token(self) -> tuple[str, float]:
        cmd_parts = shlex.split(self._settings.helix_access_token_cmd)
        if self._settings.helix_profile:
            cmd_parts.extend(["--profile", self._settings.helix_profile])

        proc = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            message = stderr.decode().strip() or stdout.decode().strip() or "unknown error"
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Helix token issuance failed: {message}",
            )

        token = stdout.decode().strip()
        lifetime = max(self._settings.helix_token_ttl - self._settings.helix_refresh_margin, 1)
        expires_at = time.time() + lifetime
        return token, expires_at

    async def get_token(self) -> str:
        async with self._lock:
            now = time.time()
            if self._token and now < self._expires_at:
                return self._token

            token, expires_at = await self._issue_token()
            self._token = token
            self._expires_at = expires_at
            return token


async def _post_with_retries(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    json_payload: Dict[str, Any],
    timeout_seconds: float,
    retries: int,
) -> httpx.Response:
    """Send POST with basic retry/backoff for transient failures."""
    backoff = 0.25
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = await client.post(
                url,
                headers=headers,
                json=json_payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if 400 <= status_code < 500:
                detail = exc.response.text.strip() or exc.response.reason_phrase
                raise HTTPException(
                    status_code=status_code,
                    detail=detail,
                ) from exc
            last_exc = exc
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            last_exc = exc

        await asyncio.sleep(backoff * (attempt + 1))

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"Upstream request failed after retries: {last_exc}",
    )


def _candidate_text(candidate: Any) -> str:
    """Extract text from a genai candidate payload."""
    if hasattr(candidate, "text"):
        return str(getattr(candidate, "text"))

    content = getattr(candidate, "content", None)
    if content is not None:
        parts = getattr(content, "parts", None)
        if parts:
            texts: List[str] = []
            for part in parts:
                if hasattr(part, "text"):
                    texts.append(str(getattr(part, "text")))
                elif isinstance(part, str):
                    texts.append(part)
            if texts:
                return " ".join(texts).strip()
        if hasattr(content, "text"):
            return str(getattr(content, "text"))

    return str(candidate)


def _extract_usage(response: Any) -> Usage | None:
    usage_meta = getattr(response, "usage_metadata", None)
    if not usage_meta:
        return None

    return Usage(
        prompt_tokens=getattr(usage_meta, "prompt_token_count", None),
        completion_tokens=getattr(usage_meta, "candidates_token_count", None),
        total_tokens=getattr(usage_meta, "total_token_count", None),
    )


def _aggregate_usage(usages: Sequence[Usage | None]) -> Usage | None:
    prompt_tokens = completion_tokens = total_tokens = 0
    seen = False
    for usage in usages:
        if usage is None:
            continue
        seen = True
        prompt_tokens += usage.prompt_tokens or 0
        completion_tokens += usage.completion_tokens or 0
        total_tokens += usage.total_tokens or 0
    if not seen:
        return None
    return Usage(
        prompt_tokens=prompt_tokens or None,
        completion_tokens=completion_tokens or None,
        total_tokens=total_tokens or None,
    )


def _extract_choices(response: Any, start_index: int) -> Tuple[List[Choice], Usage | None, int]:
    choices: List[Choice] = []
    index = start_index
    for candidate in getattr(response, "candidates", []) or []:
        finish_reason = getattr(candidate, "finish_reason", None)
        choices.append(
            Choice(
                text=_candidate_text(candidate),
                index=index,
                logprobs=None,
                finish_reason=finish_reason,
            )
        )
        index += 1
    usage = _extract_usage(response)
    return choices, usage, index


def _coerce_completion_response(payload: Dict[str, Any], request_model: str) -> CompletionResponse:
    """Fallback normalizer for LM Studio-style OpenAI responses."""
    try:
        return CompletionResponse.model_validate(payload)
    except ValidationError:
        pass

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LM Studio response was not valid JSON object",
        )

    raw_choices = payload.get("choices")
    if not isinstance(raw_choices, list):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LM Studio response missing choices",
        )

    choices: List[Choice] = []
    for idx, choice in enumerate(raw_choices):
        if not isinstance(choice, dict):
            continue
        choices.append(
            Choice(
                text=choice.get("text", ""),
                index=choice.get("index", idx),
                logprobs=choice.get("logprobs"),
                finish_reason=choice.get("finish_reason") or choice.get("finishReason"),
            )
        )

    if not choices:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LM Studio response contained no usable choices",
        )

    usage = None
    usage_raw = payload.get("usage")
    if isinstance(usage_raw, dict):
        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens") or usage_raw.get("promptTokens"),
            completion_tokens=usage_raw.get("completion_tokens") or usage_raw.get("completionTokens"),
            total_tokens=usage_raw.get("total_tokens") or usage_raw.get("totalTokens"),
        )

    return CompletionResponse(
        id=payload.get("id") or f"cmpl-{secrets.token_hex(12)}",
        object=payload.get("object", "text_completion"),
        created=payload.get("created") or int(time.time()),
        model=payload.get("model") or request_model,
        choices=choices,
        usage=usage,
    )


def _build_generation_config(request: CompletionRequest) -> genai_types.GenerationConfig:
    stop_sequences = None
    if request.stop:
        stop_sequences = [request.stop] if isinstance(request.stop, str) else list(request.stop)

    return genai_types.GenerationConfig(
        temperature=request.temperature,
        top_p=request.top_p,
        max_output_tokens=request.max_tokens,
        candidate_count=max(request.n, 1),
        stop_sequences=stop_sequences,
    )


def _build_genai_client(token: str, settings: Settings) -> genai.Client:
    if settings.ssl_cert_file:
        os.environ["SSL_CERT_FILE"] = settings.ssl_cert_file

    http_options = genai_types.HttpOptions(
        base_url=settings.vertex_base_url,
        headers={settings.custom_header_name: settings.custom_user_id},
    )

    return genai.Client(
        vertexai=True,
        project=settings.vertex_project,
        http_options=http_options,
        location=settings.vertex_location,
        credentials=Credentials(token),
    )


async def forward_to_internal_completion(
    request: CompletionRequest,
    settings: Settings,
    client: httpx.AsyncClient,
    token_manager: TokenManager,
) -> CompletionResponse:
    if not settings.internal_endpoint:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="INTERNAL_COMPLETIONS_URL not configured",
        )

    token = await token_manager.get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    response = await _post_with_retries(
        client=client,
        url=settings.internal_endpoint,
        headers=headers,
        json_payload=request.model_dump(),
        timeout_seconds=settings.request_timeout,
        retries=settings.max_retries,
    )

    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Internal completion response was not valid JSON",
        ) from exc

    try:
        return CompletionResponse.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Internal completion response did not match the expected schema",
        ) from exc


async def _generate_for_prompt(
    client: genai.Client,
    model: str,
    prompt: str,
    gen_config: genai_types.GenerationConfig,
) -> Any:
    """Run genai generate_content in a worker thread to avoid blocking the event loop."""
    return await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=[prompt],
        generation_config=gen_config,
    )


async def call_vertex_completion(
    request: CompletionRequest,
    settings: Settings,
    _client: httpx.AsyncClient,
    token_manager: TokenManager,
) -> CompletionResponse:
    """Invoke Vertex AI via google-genai client and normalize to OpenAI-style response."""
    if not settings.vertex_project:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="VERTEX_PROJECT not configured",
        )

    token = await token_manager.get_token()
    genai_client = _build_genai_client(token=token, settings=settings)
    gen_config = _build_generation_config(request)

    prompts: List[str] = [request.prompt] if isinstance(request.prompt, str) else list(request.prompt)

    choices: List[Choice] = []
    usage_list: List[Usage | None] = []
    index = 0
    for prompt in prompts:
        try:
            response = await _generate_for_prompt(
                client=genai_client,
                model=settings.vertex_model,
                prompt=prompt,
                gen_config=gen_config,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Vertex generation failed: {exc}",
            ) from exc
        prompt_choices, usage, index = _extract_choices(response, start_index=index)
        choices.extend(prompt_choices)
        usage_list.append(usage)

    if not choices:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vertex response contained no candidates",
        )

    completion_id = f"cmpl-{secrets.token_hex(12)}"
    created_at = int(time.time())
    usage = _aggregate_usage(usage_list)

    return CompletionResponse(
        id=completion_id,
        object="text_completion",
        created=created_at,
        model=request.model,
        choices=choices,
        usage=usage,
    )


async def call_lm_studio_completion(
    request: CompletionRequest,
    settings: Settings,
    client: httpx.AsyncClient,
) -> CompletionResponse:
    """Invoke a local LM Studio OpenAI-compatible completion endpoint."""
    if not settings.lm_studio_base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LM_STUDIO_BASE_URL not configured",
        )

    url = settings.lm_studio_base_url.rstrip("/") + "/v1/completions"
    response = await _post_with_retries(
        client=client,
        url=url,
        headers={"Content-Type": "application/json"},
        json_payload=request.model_dump(),
        timeout_seconds=settings.request_timeout,
        retries=settings.max_retries,
    )

    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LM Studio response was not valid JSON",
        ) from exc

    if isinstance(payload, dict):
        try:
            return CompletionResponse.model_validate(payload)
        except ValidationError:
            return _coerce_completion_response(payload, request_model=request.model)

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="LM Studio response not understood",
    )
