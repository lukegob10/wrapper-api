from __future__ import annotations

import secrets
import time
from typing import Any, List, Sequence, Tuple

from fastapi import HTTPException, status
from google import genai
from google.genai import types as genai_types
from pydantic import ValidationError

from .schemas import Choice, CompletionRequest, CompletionResponse, Usage
GENAI_CLIENT = genai.Client()


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

async def _generate_for_prompt(
    client: genai.Client,
    model: str,
    prompt: str,
    gen_config: genai_types.GenerationConfig,
) -> Any:
    """Call genai generate_content."""
    return client.models.generate_content(
        model=model,
        contents=[prompt],
        generation_config=gen_config,
    )


async def call_vertex_completion(
    request: CompletionRequest,
) -> CompletionResponse:
    """Invoke google-genai client and normalize to OpenAI-style response."""
    genai_client = GENAI_CLIENT
    gen_config = _build_generation_config(request)

    prompts: List[str] = [request.prompt] if isinstance(request.prompt, str) else list(request.prompt)

    choices: List[Choice] = []
    usage_list: List[Usage | None] = []
    index = 0
    for prompt in prompts:
        try:
            response = await _generate_for_prompt(
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
