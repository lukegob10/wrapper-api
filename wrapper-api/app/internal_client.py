from __future__ import annotations

import json
import secrets
import time
from typing import Any, Iterable, List, Sequence, Tuple

from fastapi import HTTPException, status
from google import genai
from google.genai import types as genai_types

from .schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatMessageOut,
    Choice,
    CompletionRequest,
    CompletionResponse,
    Usage,
)

# Centralized google-genai client configuration.
#
# Put all `genai.Client(...)` configuration in this one place.
# Examples (uncomment and fill in if you use Vertex AI):
#   GENAI_CLIENT_KWARGS = {"vertexai": True, "project": "my-gcp-project", "location": "us-central1"}
GENAI_CLIENT_KWARGS: dict[str, Any] = {}

# Optional default API key for the wrapper. Prefer passing a per-request key via:
# - `Authorization: Bearer ...`
# - `x-goog-api-key: ...`
DEFAULT_GENAI_API_KEY: str | None = None


def _init_genai_client(api_key: str | None) -> genai.Client:
    kwargs = dict(GENAI_CLIENT_KWARGS)
    if api_key:
        kwargs["api_key"] = api_key
    return genai.Client(**kwargs)


DEFAULT_GENAI_CLIENT = _init_genai_client(DEFAULT_GENAI_API_KEY)


def get_genai_client(api_key: str | None) -> genai.Client:
    if not api_key or (DEFAULT_GENAI_API_KEY and api_key == DEFAULT_GENAI_API_KEY):
        return DEFAULT_GENAI_CLIENT
    return _init_genai_client(api_key)


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


def _normalize_finish_reason(reason: Any) -> str | None:
    if reason is None:
        return None
    normalized = str(reason).strip().lower()
    if not normalized:
        return None
    if "stop" in normalized:
        return "stop"
    if "length" in normalized or "max" in normalized:
        return "length"
    return normalized


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
        finish_reason = _normalize_finish_reason(getattr(candidate, "finish_reason", None)) or "stop"
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


def _coerce_stop_sequences(stop: str | list[str] | None) -> list[str] | None:
    if not stop:
        return None
    return [stop] if isinstance(stop, str) else list(stop)


def _build_generation_config(
    *,
    temperature: float,
    top_p: float,
    max_output_tokens: int | None,
    n: int,
    stop: str | list[str] | None,
) -> genai_types.GenerationConfig:
    stop_sequences = _coerce_stop_sequences(stop)
    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "candidate_count": max(n, 1),
        "stop_sequences": stop_sequences,
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    return genai_types.GenerationConfig(**kwargs)


def _generate_content(
    client: genai.Client,
    model: str,
    prompt: str,
    gen_config: genai_types.GenerationConfig,
) -> Any:
    return client.models.generate_content(
        model=model,
        contents=[prompt],
        generation_config=gen_config,
    )


def _generate_content_stream(
    client: genai.Client,
    model: str,
    prompt: str,
    gen_config: genai_types.GenerationConfig,
) -> Iterable[Any]:
    if hasattr(client.models, "generate_content_stream"):
        return client.models.generate_content_stream(
            model=model,
            contents=[prompt],
            generation_config=gen_config,
        )
    try:
        return client.models.generate_content(
            model=model,
            contents=[prompt],
            generation_config=gen_config,
            stream=True,
        )
    except TypeError as exc:  # noqa: BLE001
        raise RuntimeError("Streaming is not supported by the configured google-genai client") from exc


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    parts.append(str(part["text"]))
                elif "text" in part:
                    parts.append(str(part["text"]))
        return "\n".join([p for p in parts if p]).strip()
    return str(content)


def chat_messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
    lines: List[str] = []
    for message in messages:
        role = (message.role or "user").strip()
        content = _message_content_to_text(message.content).strip()
        if not content:
            continue
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines).strip()


def _sse_data(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def call_openai_completion(
    request: CompletionRequest,
    *,
    api_key: str | None = None,
) -> CompletionResponse:
    """Invoke google-genai client and normalize to OpenAI-style response."""
    genai_client = get_genai_client(api_key)
    gen_config = _build_generation_config(
        temperature=request.temperature,
        top_p=request.top_p,
        max_output_tokens=request.max_tokens,
        n=request.n,
        stop=request.stop,
    )

    prompts: List[str] = [request.prompt] if isinstance(request.prompt, str) else list(request.prompt)

    choices: List[Choice] = []
    usage_list: List[Usage | None] = []
    index = 0
    for prompt in prompts:
        try:
            response = _generate_content(
                client=genai_client,
                model=request.model,
                prompt=prompt,
                gen_config=gen_config,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"genai generation failed: {exc}",
            ) from exc
        prompt_choices, usage, index = _extract_choices(response, start_index=index)
        choices.extend(prompt_choices)
        usage_list.append(usage)

    if not choices:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="genai response contained no candidates",
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


def stream_openai_completion(
    request: CompletionRequest,
    *,
    api_key: str | None = None,
) -> Iterable[str]:
    if request.n != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stream=true currently supports only n=1",
        )
    if not isinstance(request.prompt, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stream=true currently supports only a single prompt string",
        )

    genai_client = get_genai_client(api_key)
    gen_config = _build_generation_config(
        temperature=request.temperature,
        top_p=request.top_p,
        max_output_tokens=request.max_tokens,
        n=1,
        stop=request.stop,
    )

    completion_id = f"cmpl-{secrets.token_hex(12)}"
    created_at = int(time.time())

    seen_text = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    try:
        for chunk in _generate_content_stream(
            client=genai_client,
            model=request.model,
            prompt=request.prompt,
            gen_config=gen_config,
        ):
            usage = _extract_usage(chunk) or usage
            candidate = (getattr(chunk, "candidates", None) or [None])[0]
            if candidate is None:
                continue

            finish_reason = _normalize_finish_reason(getattr(candidate, "finish_reason", None)) or finish_reason
            current_text = _candidate_text(candidate)
            if not current_text:
                continue

            if current_text.startswith(seen_text):
                delta = current_text[len(seen_text) :]
                seen_text = current_text
            else:
                delta = current_text
                seen_text += delta

            if not delta:
                continue

            yield _sse_data(
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created_at,
                    "model": request.model,
                    "choices": [
                        {
                            "text": delta,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"genai streaming failed: {exc}",
        ) from exc

    yield _sse_data(
        {
            "id": completion_id,
            "object": "text_completion",
            "created": created_at,
            "model": request.model,
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason or "stop",
                }
            ],
        }
    )
    yield _sse_data("[DONE]")


def call_openai_chat_completion(
    request: ChatCompletionRequest,
    *,
    api_key: str | None = None,
) -> ChatCompletionResponse:
    genai_client = get_genai_client(api_key)
    max_output_tokens = request.max_completion_tokens or request.max_tokens
    gen_config = _build_generation_config(
        temperature=request.temperature,
        top_p=request.top_p,
        max_output_tokens=max_output_tokens,
        n=request.n,
        stop=request.stop,
    )

    prompt = chat_messages_to_prompt(request.messages)
    try:
        response = _generate_content(
            client=genai_client,
            model=request.model,
            prompt=prompt,
            gen_config=gen_config,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"genai generation failed: {exc}",
        ) from exc

    created_at = int(time.time())
    completion_id = f"chatcmpl-{secrets.token_hex(12)}"
    usage = _extract_usage(response)

    choices: List[ChatChoice] = []
    for index, candidate in enumerate(getattr(response, "candidates", []) or []):
        choices.append(
            ChatChoice(
                index=index,
                message=ChatMessageOut(
                    role="assistant",
                    content=_candidate_text(candidate),
                ),
                finish_reason=_normalize_finish_reason(getattr(candidate, "finish_reason", None)) or "stop",
            )
        )

    if not choices:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="genai response contained no candidates",
        )

    return ChatCompletionResponse(
        id=completion_id,
        created=created_at,
        model=request.model,
        choices=choices,
        usage=usage,
    )


def stream_openai_chat_completion(
    request: ChatCompletionRequest,
    *,
    api_key: str | None = None,
) -> Iterable[str]:
    if request.n != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stream=true currently supports only n=1",
        )

    genai_client = get_genai_client(api_key)
    max_output_tokens = request.max_completion_tokens or request.max_tokens
    gen_config = _build_generation_config(
        temperature=request.temperature,
        top_p=request.top_p,
        max_output_tokens=max_output_tokens,
        n=1,
        stop=request.stop,
    )
    prompt = chat_messages_to_prompt(request.messages)

    completion_id = f"chatcmpl-{secrets.token_hex(12)}"
    created_at = int(time.time())

    yield _sse_data(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_at,
            "model": request.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )

    seen_text = ""
    finish_reason: str | None = None
    try:
        for chunk in _generate_content_stream(
            client=genai_client,
            model=request.model,
            prompt=prompt,
            gen_config=gen_config,
        ):
            candidate = (getattr(chunk, "candidates", None) or [None])[0]
            if candidate is None:
                continue

            finish_reason = _normalize_finish_reason(getattr(candidate, "finish_reason", None)) or finish_reason
            current_text = _candidate_text(candidate)
            if not current_text:
                continue

            if current_text.startswith(seen_text):
                delta = current_text[len(seen_text) :]
                seen_text = current_text
            else:
                delta = current_text
                seen_text += delta

            if not delta:
                continue

            yield _sse_data(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                }
            )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"genai streaming failed: {exc}",
        ) from exc

    yield _sse_data(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_at,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}],
        }
    )
    yield _sse_data("[DONE]")
