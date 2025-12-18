from __future__ import annotations

import datetime as dt
import json
import os
import queue
import secrets
import subprocess
import threading
import time
from typing import Any, Iterable, List, Sequence, Tuple

from fastapi import HTTPException, status
from google import genai
from google.genai import types as genai_types
from google.auth.credentials import Credentials

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

#
# GenAI client configuration (single place).
#
# Put your exact `genai.Client(...)` kwargs here (vertexai/project/location/etc).
GENAI_CLIENT_KWARGS: dict[str, Any] = {}

# Helix CLI command that prints an access token.
# - If it's a string, it's run via the shell.
# - If it's a list, it's executed directly (recommended).
HELIX_TOKEN_COMMAND: str | Sequence[str] | None = None
HELIX_TOKEN_TIMEOUT_SECONDS = 15
HELIX_TOKEN_TTL_SECONDS = 3000

# Optional: path to your router/CA PEM. This is applied to the process env so
# the underlying HTTP/TLS stack can pick it up.
SSL_CERT_PEM_PATH: str | None = None
SSL_CERT_ENV_KEYS: Sequence[str] = (
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT",
    "SSL CERT",
)

# Roo/Cline often sends a dummy "API key" in the Authorization header. Ignore it by default.
ALLOW_REQUEST_API_KEY = False

# If your upstream/router streaming never terminates cleanly, set this to False
# to "simulate" streaming via a single non-streaming call + SSE framing.
USE_UPSTREAM_STREAMING = False

# Safeties for upstream streaming: some routers never close the stream even after
# the model is done. These timeouts prevent clients (e.g., Roo) from spinning forever.
UPSTREAM_STREAM_START_TIMEOUT_SECONDS = 90.0
UPSTREAM_STREAM_IDLE_TIMEOUT_SECONDS = 15.0


def _run_command(command: str | Sequence[str]) -> str:
    """Run command and return decoded stdout.

    Uses the `check_output(...).decode().strip()` style expected by your Helix flow.
    """
    try:
        if isinstance(command, str):
            output = subprocess.check_output(
                command,
                shell=True,
                stderr=subprocess.STDOUT,
                timeout=HELIX_TOKEN_TIMEOUT_SECONDS,
            )
        else:
            output = subprocess.check_output(
                list(command),
                stderr=subprocess.STDOUT,
                timeout=HELIX_TOKEN_TIMEOUT_SECONDS,
            )
    except subprocess.CalledProcessError as exc:  # noqa: BLE001
        message = (exc.output or b"").decode(errors="replace").strip()
        raise RuntimeError(f"token command failed (exit {exc.returncode}): {message}") from exc

    stdout = (output or b"").decode(errors="replace").strip()
    if not stdout:
        raise RuntimeError("token command returned empty output")
    return stdout


def _apply_ssl_cert_env() -> None:
    if not SSL_CERT_PEM_PATH:
        return
    for key in SSL_CERT_ENV_KEYS:
        os.environ[key] = SSL_CERT_PEM_PATH


def _fetch_helix_token(command: str | Sequence[str]) -> tuple[str, dt.datetime | None]:
    _apply_ssl_cert_env()
    stdout = _run_command(command)

    token = stdout.splitlines()[0].strip()
    expiry = dt.datetime.utcnow() + dt.timedelta(seconds=HELIX_TOKEN_TTL_SECONDS)

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict) and data.get("access_token"):
        token = str(data["access_token"]).strip()
        expires_in = data.get("expires_in")
        if expires_in is not None:
            try:
                expiry = dt.datetime.utcnow() + dt.timedelta(seconds=int(expires_in))
            except Exception:  # noqa: BLE001
                pass

    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    if not token:
        raise RuntimeError("helix token command did not return a token")

    return token, expiry


class HelixTokenCredentials(Credentials):
    def __init__(self, command: str | Sequence[str]) -> None:
        super().__init__()
        self._command = command
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expiry: dt.datetime | None = None

    @property
    def token(self) -> str | None:  # type: ignore[override]
        return self._token

    @property
    def expiry(self) -> dt.datetime | None:  # type: ignore[override]
        return self._expiry

    @property
    def expired(self) -> bool:  # type: ignore[override]
        if not self._token:
            return True
        if self._expiry is None:
            return False
        return dt.datetime.utcnow() >= self._expiry

    @property
    def valid(self) -> bool:  # type: ignore[override]
        return bool(self._token) and not self.expired

    @property
    def requires_scopes(self) -> bool:  # type: ignore[override]
        return False

    def refresh(self, request: Any) -> None:  # type: ignore[override]
        with self._lock:
            token, expiry = _fetch_helix_token(self._command)
            self._token = token
            self._expiry = expiry


_DEFAULT_GENAI_CLIENT: genai.Client | None = None
_DEFAULT_CLIENT_LOCK = threading.Lock()


def get_genai_client(api_key: str | None) -> genai.Client:
    _apply_ssl_cert_env()
    if ALLOW_REQUEST_API_KEY and api_key:
        kwargs = dict(GENAI_CLIENT_KWARGS)
        kwargs["api_key"] = api_key
        return genai.Client(**kwargs)

    global _DEFAULT_GENAI_CLIENT
    if _DEFAULT_GENAI_CLIENT is None:
        with _DEFAULT_CLIENT_LOCK:
            if _DEFAULT_GENAI_CLIENT is None:
                kwargs = dict(GENAI_CLIENT_KWARGS)
                if HELIX_TOKEN_COMMAND:
                    kwargs["credentials"] = HelixTokenCredentials(HELIX_TOKEN_COMMAND)
                _DEFAULT_GENAI_CLIENT = genai.Client(**kwargs)
    return _DEFAULT_GENAI_CLIENT


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


def _request_include_usage(request: Any) -> bool:
    stream_options = getattr(request, "stream_options", None)
    if not isinstance(stream_options, dict):
        return False
    return bool(stream_options.get("include_usage"))


def _usage_dict(usage: Usage | None) -> dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": usage.prompt_tokens or 0,
        "completion_tokens": usage.completion_tokens or 0,
        "total_tokens": usage.total_tokens or 0,
    }


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
    include_usage = _request_include_usage(request)

    try:
        if not USE_UPSTREAM_STREAMING:
            response = _generate_content(
                client=genai_client,
                model=request.model,
                prompt=request.prompt,
                gen_config=gen_config,
            )

            choices, usage, _ = _extract_choices(response, start_index=0)
            first = choices[0]
            if first.text:
                yield _sse_data(
                    {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created_at,
                        "model": request.model,
                        "choices": [
                            {
                                "text": first.text,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            final_payload: dict[str, Any] = {
                "id": completion_id,
                "object": "text_completion",
                "created": created_at,
                "model": request.model,
                "choices": [
                    {
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": first.finish_reason or "stop",
                    }
                ],
            }
            if usage is not None:
                final_payload["usage"] = usage.model_dump(exclude_none=True)
            yield _sse_data(final_payload)
            if include_usage:
                yield _sse_data(
                    {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created_at,
                        "model": request.model,
                        "choices": [],
                        "usage": _usage_dict(usage),
                    }
                )
            yield _sse_data("[DONE]")
            return
    except Exception as exc:  # noqa: BLE001
        error_text = f"Error: genai request failed: {exc}"
        yield _sse_data(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created_at,
                "model": request.model,
                "choices": [
                    {
                        "text": error_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
        )
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
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        yield _sse_data("[DONE]")
        return

    seen_text = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    stream = _generate_content_stream(
        client=genai_client,
        model=request.model,
        prompt=request.prompt,
        gen_config=gen_config,
    )
    stream_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def _producer() -> None:
        try:
            for item in stream:
                stream_queue.put(("chunk", item))
        except BaseException as exc:  # noqa: BLE001
            stream_queue.put(("error", exc))
        finally:
            stream_queue.put(("done", None))

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    stream_error: BaseException | None = None
    got_any_chunk = False
    while True:
        timeout = (
            UPSTREAM_STREAM_IDLE_TIMEOUT_SECONDS
            if got_any_chunk
            else UPSTREAM_STREAM_START_TIMEOUT_SECONDS
        )
        try:
            kind, payload = stream_queue.get(timeout=timeout)
        except queue.Empty:
            if seen_text:
                break
            stream_error = TimeoutError(f"upstream stream idle for {timeout:.0f}s")
            break

        if kind == "chunk":
            got_any_chunk = True
            chunk = payload
            usage = _extract_usage(chunk) or usage
            candidate = (getattr(chunk, "candidates", None) or [None])[0]
            if candidate is None:
                continue

            chunk_finish_reason = _normalize_finish_reason(getattr(candidate, "finish_reason", None))
            finish_reason = chunk_finish_reason or finish_reason
            current_text = _candidate_text(candidate)
            if not current_text:
                if chunk_finish_reason:
                    break
                continue

            if current_text.startswith(seen_text):
                delta = current_text[len(seen_text) :]
                seen_text = current_text
            else:
                delta = current_text
                seen_text += delta

            if not delta:
                if chunk_finish_reason:
                    break
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
            if chunk_finish_reason:
                break
            continue

        if kind == "error":
            stream_error = payload if isinstance(payload, BaseException) else RuntimeError(str(payload))
            break

        if kind == "done":
            break

    close = getattr(stream, "close", None)
    if callable(close):
        close()

    if stream_error is not None and not seen_text:
        error_text = f"Error: genai streaming failed: {stream_error}"
        yield _sse_data(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created_at,
                "model": request.model,
                "choices": [
                    {
                        "text": error_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
        )

    final_payload = {
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
    if usage is not None:
        final_payload["usage"] = usage.model_dump(exclude_none=True)
    yield _sse_data(final_payload)
    if include_usage:
        yield _sse_data(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created_at,
                "model": request.model,
                "choices": [],
                "usage": _usage_dict(usage),
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
    include_usage = _request_include_usage(request)

    yield _sse_data(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_at,
            "model": request.model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
    )

    try:
        if not USE_UPSTREAM_STREAMING:
            response = _generate_content(
                client=genai_client,
                model=request.model,
                prompt=prompt,
                gen_config=gen_config,
            )

            candidate = (getattr(response, "candidates", None) or [None])[0]
            if candidate is None:
                raise RuntimeError("genai response contained no candidates")

            text = _candidate_text(candidate)
            finish_reason = _normalize_finish_reason(getattr(candidate, "finish_reason", None)) or "stop"
            usage = _extract_usage(response)
            if text:
                yield _sse_data(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_at,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                    }
                )

            final_payload: dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_at,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": finish_reason}],
            }
            if usage is not None:
                final_payload["usage"] = usage.model_dump(exclude_none=True)
            yield _sse_data(final_payload)
            if include_usage:
                yield _sse_data(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_at,
                        "model": request.model,
                        "choices": [],
                        "usage": _usage_dict(usage),
                    }
                )
            yield _sse_data("[DONE]")
            return
    except Exception as exc:  # noqa: BLE001
        error_text = f"Error: genai request failed: {exc}"
        yield _sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_at,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": error_text}, "finish_reason": None}],
            }
        )
        yield _sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_at,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
            }
        )
        yield _sse_data("[DONE]")
        return

    seen_text = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    stream = _generate_content_stream(
        client=genai_client,
        model=request.model,
        prompt=prompt,
        gen_config=gen_config,
    )
    stream_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def _producer() -> None:
        try:
            for item in stream:
                stream_queue.put(("chunk", item))
        except BaseException as exc:  # noqa: BLE001
            stream_queue.put(("error", exc))
        finally:
            stream_queue.put(("done", None))

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    stream_error: BaseException | None = None
    got_any_chunk = False
    while True:
        timeout = (
            UPSTREAM_STREAM_IDLE_TIMEOUT_SECONDS
            if got_any_chunk
            else UPSTREAM_STREAM_START_TIMEOUT_SECONDS
        )
        try:
            kind, payload = stream_queue.get(timeout=timeout)
        except queue.Empty:
            if seen_text:
                break
            stream_error = TimeoutError(f"upstream stream idle for {timeout:.0f}s")
            break

        if kind == "chunk":
            got_any_chunk = True
            chunk = payload
            usage = _extract_usage(chunk) or usage
            candidate = (getattr(chunk, "candidates", None) or [None])[0]
            if candidate is None:
                continue

            chunk_finish_reason = _normalize_finish_reason(getattr(candidate, "finish_reason", None))
            finish_reason = chunk_finish_reason or finish_reason
            current_text = _candidate_text(candidate)
            if not current_text:
                if chunk_finish_reason:
                    break
                continue

            if current_text.startswith(seen_text):
                delta = current_text[len(seen_text) :]
                seen_text = current_text
            else:
                delta = current_text
                seen_text += delta

            if not delta:
                if chunk_finish_reason:
                    break
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
            if chunk_finish_reason:
                break
            continue

        if kind == "error":
            stream_error = payload if isinstance(payload, BaseException) else RuntimeError(str(payload))
            break

        if kind == "done":
            break

    close = getattr(stream, "close", None)
    if callable(close):
        close()

    if stream_error is not None and not seen_text:
        error_text = f"Error: genai streaming failed: {stream_error}"
        yield _sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_at,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": error_text}, "finish_reason": None}],
            }
        )

    final_payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_at,
        "model": request.model,
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": finish_reason or "stop"}],
    }
    if usage is not None:
        final_payload["usage"] = usage.model_dump(exclude_none=True)
    yield _sse_data(final_payload)
    if include_usage:
        yield _sse_data(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_at,
                "model": request.model,
                "choices": [],
                "usage": _usage_dict(usage),
            }
        )
    yield _sse_data("[DONE]")
