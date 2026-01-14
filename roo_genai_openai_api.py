# CHANGES IN THIS ITERATION (high-level)
# 1) Added dual-mode client initialization:
#    - PERSONAL mode: uses GOOGLE_API_KEY / GEMINI_API_KEY (as before)
#    - WORK mode: uses Helix CLI token + optional endpoint/project/location/vertex settings
# 2) Client construction is now "signature-safe": we only pass kwargs supported by your installed google-genai version.
# 3) Token refresh/re-init is built-in: if auth/token expires, cached client is reset and the call is retried once.
#
# WHAT YOU NEED TO SET (env vars)
# - GENAI_AUTH_MODE = "personal" (default) OR "work"
#
# PERSONAL MODE:
# - GOOGLE_API_KEY or GEMINI_API_KEY
#
# WORK MODE (examples; set what you actually have):
# - HELIX_TOKEN_COMMAND="helix token"   (whatever prints a token to stdout)
# - GENAI_API_ENDPOINT="https://your.internal.endpoint"   (optional)
# - GENAI_VERTEXAI="1"                  (optional; if using Vertex AI auth mode)
# - GENAI_PROJECT="your-gcp-project"    (optional; for vertexai=True)
# - GENAI_LOCATION="us-central1"        (optional; for vertexai=True)
#
# Note: This file stays compatible even if your google-genai Client signature differs across environments.

import math
import json
import os
import re
import time
import uuid
import asyncio
import subprocess
import shlex
import inspect
from collections.abc import AsyncIterator, Iterable
from typing import Any

from fastapi import Header, HTTPException, Request
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    from google.genai import errors as genai_errors  # type: ignore
except Exception:  # pragma: no cover
    genai = None
    types = None
    genai_errors = None


APP_TITLE = "GenAI OpenAI-Compatible API"


def _load_dotenv(path: str = ".env") -> None:
    """
    Minimal .env loader (no dependencies).
    - Supports KEY=VALUE lines and optional quotes.
    - By default does NOT override existing environment variables.
    - Set DOTENV_OVERRIDE=1 to override existing env vars.
    """
    override = (os.getenv("DOTENV_OVERRIDE") or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                # strip accidental whitespace
                value = value.strip()

                if not override and key in os.environ:
                    continue
                os.environ[key] = value
    except FileNotFoundError:
        return


_load_dotenv()
# Keep a default only for `/v1/models` advertising and manual curl tests.
# Chat completions should use the request's `model`.
DEFAULT_MODEL = "gemini-2.0-flash"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
LOG_REQUESTS = (os.getenv("LOG_REQUESTS") or "").strip().lower() in {"1", "true", "yes", "y", "on"}

OPENAI_RATELIMIT_LIMIT_REQUESTS = int(os.getenv("OPENAI_RATELIMIT_LIMIT_REQUESTS", "9999"))
OPENAI_RATELIMIT_LIMIT_TOKENS = int(os.getenv("OPENAI_RATELIMIT_LIMIT_TOKENS", "999999"))
OPENAI_RATELIMIT_RESET_SECONDS = int(os.getenv("OPENAI_RATELIMIT_RESET_SECONDS", "1"))

app = FastAPI(title=APP_TITLE, version="0.1.0")

_client: Any | None = None
_client_lock = asyncio.Lock()


def _openai_rate_limit_headers(*, status_code: int, retry_after_seconds: int | None = None) -> dict[str, str]:
    reset_seconds = (
        retry_after_seconds
        if isinstance(retry_after_seconds, int) and retry_after_seconds > 0
        else OPENAI_RATELIMIT_RESET_SECONDS
    )
    remaining_requests = 0 if status_code == 429 else OPENAI_RATELIMIT_LIMIT_REQUESTS
    remaining_tokens = 0 if status_code == 429 else OPENAI_RATELIMIT_LIMIT_TOKENS
    reset_str = f"{reset_seconds}s"
    return {
        "x-ratelimit-limit-requests": str(OPENAI_RATELIMIT_LIMIT_REQUESTS),
        "x-ratelimit-remaining-requests": str(max(0, remaining_requests)),
        "x-ratelimit-reset-requests": reset_str,
        "x-ratelimit-limit-tokens": str(OPENAI_RATELIMIT_LIMIT_TOKENS),
        "x-ratelimit-remaining-tokens": str(max(0, remaining_tokens)),
        "x-ratelimit-reset-tokens": reset_str,
    }


@app.middleware("http")
async def _add_openai_ratelimit_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
    response = await call_next(request)

    retry_after_seconds: int | None = None
    if response.status_code == 429:
        ra = response.headers.get("Retry-After")
        if ra:
            try:
                retry_after_seconds = int(math.ceil(float(ra)))
            except Exception:
                retry_after_seconds = None

    for k, v in _openai_rate_limit_headers(status_code=response.status_code, retry_after_seconds=retry_after_seconds).items():
        if k not in response.headers:
            response.headers[k] = v
    return response


def _get_auth_token(authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key
    if not authorization:
        return None
    prefix = "bearer "
    if authorization.lower().startswith(prefix):
        return authorization[len(prefix) :].strip()
    return None


def _require_wrapper_auth(token: str | None) -> None:
    required = os.getenv("WRAPPER_API_KEY")
    if required and token != required:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _google_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="Missing GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable.",
        )
    return key


# -----------------------------
# NEW: Dual-mode client builder
# -----------------------------

def _auth_mode() -> str:
    """
    Determines how we authenticate to GenAI.
    - personal: uses GOOGLE_API_KEY / GEMINI_API_KEY
    - work: uses Helix token + optional internal endpoint / vertex parameters
    """
    mode = (os.getenv("GENAI_AUTH_MODE") or "personal").strip().lower()
    if mode not in {"personal", "work"}:
        mode = "personal"
    return mode


def _helix_token() -> str:
    """
    Fetch a short-lived token from Helix CLI (or any internal tool).
    Command must print a token to stdout.
    """
    cmd = (os.getenv("HELIX_TOKEN_COMMAND") or "").strip()
    if not cmd:
        raise HTTPException(
            status_code=500,
            detail="GENAI_AUTH_MODE=work but HELIX_TOKEN_COMMAND is not set.",
        )

    try:
        # Use shlex to support quoted args in env var
        args = shlex.split(cmd)
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run HELIX_TOKEN_COMMAND: {exc}") from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise HTTPException(
            status_code=500,
            detail=f"HELIX_TOKEN_COMMAND failed (rc={proc.returncode}): {stderr or 'no stderr'}",
        )

    token = (proc.stdout or "").strip()
    if not token:
        raise HTTPException(status_code=500, detail="HELIX_TOKEN_COMMAND returned empty stdout.")
    return token


def _filter_kwargs_for_client(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Only pass kwargs that your installed google-genai Client accepts.
    This keeps one script working across:
      - personal machines (api_key only)
      - corp installs (extra vertex/client_options/custom endpoint/etc.)
    """
    if genai is None:
        return kwargs

    try:
        sig = inspect.signature(genai.Client)  # type: ignore[attr-defined]
    except Exception:
        return kwargs

    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted and v is not None}


def _build_personal_client() -> Any:
    if genai is None:
        raise HTTPException(
            status_code=500,
            detail="google-genai is not installed. Install with: pip install google-genai",
        )
    kwargs = {"api_key": _google_api_key()}
    kwargs = _filter_kwargs_for_client(kwargs)
    if LOG_REQUESTS:
        k = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        print(f"Auth mode=personal; api_key_present={bool(k)}; api_key_suffix={k[-6:] if k else ''}")

    return genai.Client(**kwargs)  # type: ignore[misc]


def _build_work_client() -> Any:
    """
    Work-mode client:
      - gets token from Helix CLI
      - optionally points to internal endpoint
      - optionally uses vertexai=True with project/location if your environment needs it
    """
    if genai is None:
        raise HTTPException(
            status_code=500,
            detail="google-genai is not installed. Install with: pip install google-genai",
        )

    token = _helix_token()

    # Optional: internal endpoint override
    endpoint = (os.getenv("GENAI_API_ENDPOINT") or "").strip()
    client_options = None
    if endpoint:
        # Some google libs use "client_options={'api_endpoint': ...}"
        # This may or may not be supported by google-genai.Client; we filter via signature.
        client_options = {"api_endpoint": endpoint}

    # Optional: Vertex routing/fields (common in corp setups)
    vertexai_flag = (os.getenv("GENAI_VERTEXAI") or "").strip().lower() in {"1", "true", "yes", "y", "on"}
    project = (os.getenv("GENAI_PROJECT") or "").strip() or None
    location = (os.getenv("GENAI_LOCATION") or "").strip() or None

    kwargs: dict[str, Any] = {}

    # Many corp setups treat the short-lived token as the "api_key" value
    # (even if it isn't literally a Google API key).
    kwargs["api_key"] = token

    # Vertex-specific flags (only applied if supported by your installed client)
    kwargs["vertexai"] = vertexai_flag
    kwargs["project"] = project
    kwargs["location"] = location

    # Endpoint override (only if supported)
    kwargs["client_options"] = client_options

    kwargs = _filter_kwargs_for_client(kwargs)

    if LOG_REQUESTS:
        safe = {k: ("***" if k in {"api_key"} else v) for k, v in kwargs.items()}
        print("Initializing WORK GenAI client with kwargs:", safe)

    return genai.Client(**kwargs)  # type: ignore[misc]


def _get_client() -> Any:
    global _client
    if _client is not None:
        return _client

    mode = _auth_mode()
    if mode == "work":
        _client = _build_work_client()
        return _client

    _client = _build_personal_client()
    return _client


def _looks_like_expired_token_error(exc: Exception) -> bool:
    msg = str(exc).lower()

    needles = (
        "unauthenticated",
        "authentication failed",
        "invalid authentication",
        "invalid credentials",
        "expired",
        "token has expired",
        "access token expired",
        "permission denied",
        "401",
        "403",
        "refresh",
        "oauth",
    )
    if any(n in msg for n in needles):
        return True

    if genai_errors is not None:
        status_code = getattr(exc, "status_code", None)
        if status_code in (401, 403):
            return True

    return False


async def _reset_client() -> Any:
    global _client
    async with _client_lock:
        _client = None
        return _get_client()


async def _call_genai_with_reauth(fn, *, retry: bool = True):
    """
    Call a sync genai function in a threadpool.
    If token/auth is expired, reset client and retry once.
    """
    try:
        return await run_in_threadpool(fn)
    except Exception as exc:
        if retry and _looks_like_expired_token_error(exc):
            if LOG_REQUESTS:
                print("Auth/token failure detected; resetting GenAI client and retrying once:", repr(exc))
            await _reset_client()
            return await _call_genai_with_reauth(fn, retry=False)
        raise


# -----------------------------
# Rest of your original helpers
# -----------------------------

def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # Be permissive: some clients send {"type":"text","text":"..."}.
        if isinstance(content.get("text"), str):
            return content["text"]
        return json.dumps(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if (item_type in {None, "text", "input_text"}) and isinstance(item.get("text"), str):
                parts.append(item["text"])
                continue
            raise HTTPException(
                status_code=400,
                detail="Only text content parts are supported for MVP.",
            )
        return "".join(parts)
    raise HTTPException(status_code=400, detail="Invalid message.content")


def _part_text(text: str) -> Any:
    if types is not None:
        try:
            return types.Part(text=text)
        except Exception:
            pass
    return {"text": text}


def _part_function_call(name: str, args: dict[str, Any]) -> Any:
    if types is not None:
        try:
            return types.Part(function_call=types.FunctionCall(name=name, args=args))
        except Exception:
            pass
    return {"function_call": {"name": name, "args": args}}


def _part_function_response(name: str, response: Any) -> Any:
    if types is not None:
        try:
            return types.Part(function_response=types.FunctionResponse(name=name, response=response))
        except Exception:
            pass
    return {"function_response": {"name": name, "response": response}}


def _parse_tool_result(content: Any) -> Any:
    if content is None:
        return {}
    if isinstance(content, (dict, list, int, float, bool)):
        return content
    if not isinstance(content, str):
        return {"content": str(content)}
    text = content.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {"content": content}


def _messages_to_genai(
    messages: list[dict[str, Any]], *, openai_to_genai: dict[str, str] | None = None
) -> tuple[str | None, list[dict[str, Any]]]:
    system_chunks: list[str] = []
    contents: list[dict[str, Any]] = []
    tool_call_id_to_name: dict[str, str] = {}
    openai_to_genai = openai_to_genai or {}

    for message in messages:
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="Each message must be an object.")

        role = message.get("role")
        if role in {"system", "developer"}:
            system_chunks.append(_message_text(message))
            continue

        if role == "assistant":
            tool_calls = message.get("tool_calls")
            # Legacy OpenAI shape: {"role":"assistant","function_call":{...}}
            if tool_calls is None and isinstance(message.get("function_call"), dict):
                fc = message["function_call"]
                tool_calls = [
                    {
                        "id": f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": fc.get("name"),
                            "arguments": fc.get("arguments", "{}"),
                        },
                    }
                ]
            assistant_text = _message_text(message)
            parts: list[Any] = []
            if assistant_text:
                parts.append(_part_text(assistant_text))
            if tool_calls is not None:
                if not isinstance(tool_calls, list):
                    raise HTTPException(status_code=400, detail="assistant.tool_calls must be an array.")
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    if tc.get("type") != "function":
                        raise HTTPException(status_code=400, detail="Only function tool calls are supported.")
                    fn = tc.get("function")
                    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                        raise HTTPException(status_code=400, detail="tool_call.function.name is required.")
                    openai_name = fn["name"]
                    name = openai_to_genai.get(openai_name, openai_name)
                    args_raw = fn.get("arguments") or "{}"
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw) if args_raw.strip() else {}
                        except Exception:
                            args = {"_raw": args_raw}
                    elif isinstance(args_raw, dict):
                        args = args_raw
                    else:
                        args = {"_raw": str(args_raw)}
                    tc_id = tc.get("id")
                    if isinstance(tc_id, str):
                        tool_call_id_to_name[tc_id] = openai_name
                    parts.append(_part_function_call(name, args))

            contents.append({"role": "model", "parts": parts or [_part_text("")]})
            continue

        if role in {"tool", "function"}:
            tool_call_id = message.get("tool_call_id")
            name = None
            if isinstance(message.get("name"), str):
                name = message["name"]
            elif isinstance(tool_call_id, str):
                name = tool_call_id_to_name.get(tool_call_id)
            if not name:
                raise HTTPException(
                    status_code=400,
                    detail="tool messages must include name or tool_call_id matching a prior tool_call.",
                )

            genai_name = openai_to_genai.get(name, name)
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        _part_function_response(genai_name, _parse_tool_result(message.get("content")))
                    ],
                }
            )
            continue

        if role == "user":
            genai_role = "user"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported role: {role!r}")

        text = _message_text(message)
        contents.append({"role": genai_role, "parts": [_part_text(text)]})

    system_instruction = "\n\n".join([c for c in system_chunks if c.strip()]) or None
    return system_instruction, contents


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text

    # Best-effort fallback if `.text` isn't present.
    candidates = getattr(response, "candidates", None)
    try:
        first_candidate = candidates[0] if candidates else None
    except Exception:
        first_candidate = None

    if first_candidate is None and candidates is not None:
        try:
            first_candidate = next(iter(candidates))
        except Exception:
            first_candidate = None

    if first_candidate is not None:
        content = getattr(first_candidate, "content", None)
        parts = getattr(content, "parts", None)
        if parts is not None:
            out: list[str] = []
            try:
                parts_iter = iter(parts)
            except Exception:
                parts_iter = iter(())
            for part in parts_iter:
                part_text = getattr(part, "text", None)
                if part_text is None and isinstance(part, dict):
                    part_text = part.get("text")
                if isinstance(part_text, str):
                    out.append(part_text)
            if out:
                return "".join(out)

    return ""


def _extract_function_calls(response: Any, *, genai_to_openai: dict[str, str] | None = None) -> list[dict[str, Any]]:
    genai_to_openai = genai_to_openai or {}

    # Some versions expose function calls directly.
    direct_calls = getattr(response, "function_calls", None)
    if direct_calls is not None:
        calls: list[dict[str, Any]] = []
        try:
            direct_iter = iter(direct_calls)
        except Exception:
            direct_iter = iter(())

        for fn_call in direct_iter:
            name = getattr(fn_call, "name", None) if not isinstance(fn_call, dict) else fn_call.get("name")
            args = getattr(fn_call, "args", None) if not isinstance(fn_call, dict) else fn_call.get("args")
            if not isinstance(name, str) or not name:
                continue
            openai_name = genai_to_openai.get(name, name)
            if args is None:
                args = {}
            if not isinstance(args, dict):
                args = {"_raw": args}
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": openai_name, "arguments": json.dumps(args)},
                }
            )
        if calls:
            return calls

    candidates = getattr(response, "candidates", None)
    try:
        first_candidate = candidates[0] if candidates else None
    except Exception:
        first_candidate = None

    if first_candidate is None and candidates is not None:
        try:
            first_candidate = next(iter(candidates))
        except Exception:
            first_candidate = None

    if first_candidate is None:
        return []
    content = getattr(first_candidate, "content", None)
    parts = getattr(content, "parts", None)
    if parts is None:
        return []

    calls: list[dict[str, Any]] = []
    try:
        parts_iter = iter(parts)
    except Exception:
        parts_iter = iter(())
    for part in parts_iter:
        fn_call = getattr(part, "function_call", None)
        if fn_call is None and isinstance(part, dict):
            fn_call = part.get("function_call") or part.get("functionCall")
        if fn_call is None:
            continue

        name = getattr(fn_call, "name", None) if not isinstance(fn_call, dict) else fn_call.get("name")
        args = getattr(fn_call, "args", None) if not isinstance(fn_call, dict) else fn_call.get("args")
        if not isinstance(name, str) or not name:
            continue
        openai_name = genai_to_openai.get(name, name)
        if args is None:
            args = {}
        if not isinstance(args, dict):
            args = {"_raw": args}

        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {"name": openai_name, "arguments": json.dumps(args)},
            }
        )

    return calls


def _genai_tools(openai_tools: Any) -> Any:
    if openai_tools is None:
        return None
    if not isinstance(openai_tools, list):
        raise HTTPException(status_code=400, detail="tools must be an array.")

    declarations: list[Any] = []
    for tool in openai_tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        name = fn["name"]
        description = fn.get("description") if isinstance(fn.get("description"), str) else None
        parameters = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object"}
        declarations.append({"name": name, "description": description, "parameters": parameters})

    if not declarations:
        return None

    if types is not None:
        try:
            fn_decls = [
                types.FunctionDeclaration(
                    name=d["name"],
                    description=d.get("description"),
                    parameters=d.get("parameters"),
                )
                for d in declarations
            ]
            return [types.Tool(function_declarations=fn_decls)]
        except Exception:
            pass

    return [{"function_declarations": declarations}]


def _sanitize_json_schema(schema: Any, *, depth: int = 0) -> dict[str, Any]:
    if depth > 20:
        return {"type": "string"}
    if not isinstance(schema, dict):
        return {"type": "string"}

    if "$ref" in schema:
        out: dict[str, Any] = {"type": "string"}
        if isinstance(schema.get("description"), str):
            out["description"] = schema["description"]
        return out

    for union_key in ("oneOf", "anyOf", "allOf"):
        if isinstance(schema.get(union_key), list) and schema[union_key]:
            return _sanitize_json_schema(schema[union_key][0], depth=depth + 1)

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        # Gemini function schemas expect a single type string.
        type_candidates = [t for t in schema_type if isinstance(t, str)]
        if "null" in type_candidates and len(type_candidates) > 1:
            type_candidates = [t for t in type_candidates if t != "null"]
        schema_type = type_candidates[0] if type_candidates else None
    if schema_type is None and "properties" in schema:
        schema_type = "object"
    if schema_type is None and "items" in schema:
        schema_type = "array"
    if schema_type is None:
        schema_type = "string"
    if isinstance(schema_type, str):
        schema_type = schema_type.lower()
    if schema_type not in {"string", "number", "integer", "boolean", "object", "array"}:
        schema_type = "string"

    out: dict[str, Any] = {"type": schema_type}
    if isinstance(schema.get("description"), str):
        out["description"] = schema["description"]
    if "enum" in schema and isinstance(schema["enum"], list) and all(
        isinstance(v, (str, int, float, bool)) or v is None for v in schema["enum"]
    ):
        out["enum"] = schema["enum"]

    if schema_type == "object":
        props_in = schema.get("properties")
        props_out: dict[str, Any] = {}
        if isinstance(props_in, dict):
            for key, value in props_in.items():
                if isinstance(key, str) and key:
                    props_out[key] = _sanitize_json_schema(value, depth=depth + 1)
        out["properties"] = props_out

        required_in = schema.get("required")
        if isinstance(required_in, list):
            required_out = [r for r in required_in if isinstance(r, str) and r in props_out]
            if required_out:
                out["required"] = required_out
        return out

    if schema_type == "array":
        out["items"] = _sanitize_json_schema(schema.get("items"), depth=depth + 1)
        return out

    if isinstance(schema.get("format"), str):
        out["format"] = schema["format"]
    return out


def _apply_tool_name_map_to_tool_choice(tool_choice: Any, openai_to_genai: dict[str, str]) -> Any:
    if not isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice.get("type") == "function" and isinstance(tool_choice.get("function"), dict):
        fn = dict(tool_choice["function"])
        if isinstance(fn.get("name"), str):
            fn["name"] = openai_to_genai.get(fn["name"], fn["name"])
        return {"type": "function", "function": fn}
    if isinstance(tool_choice.get("name"), str):
        mapped = dict(tool_choice)
        mapped["name"] = openai_to_genai.get(mapped["name"], mapped["name"])
        return mapped
    return tool_choice


def _normalize_openai_tools(body: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = body.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise HTTPException(status_code=400, detail="tools must be an array.")
        return tools

    functions = body.get("functions")
    if functions is None:
        return None
    if not isinstance(functions, list):
        raise HTTPException(status_code=400, detail="functions must be an array.")

    normalized: list[dict[str, Any]] = []
    for fn in functions:
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        normalized.append({"type": "function", "function": fn})
    return normalized or None


def _build_tool_name_maps(openai_tools: list[dict[str, Any]] | None) -> tuple[dict[str, str], dict[str, str]]:
    if not openai_tools:
        return {}, {}

    used: set[str] = set()
    o2g: dict[str, str] = {}
    g2o: dict[str, str] = {}

    for tool in openai_tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        openai_name = fn["name"]
        base = re.sub(r"[^A-Za-z0-9_]", "_", openai_name)
        if not re.match(r"^[A-Za-z_]", base):
            base = f"fn_{base}"
        base = base[:64] or "fn"
        genai_name = base
        if genai_name in used:
            genai_name = (base[:57] + "_" + uuid.uuid4().hex[:6])[:64]
        used.add(genai_name)
        o2g[openai_name] = genai_name
        g2o[genai_name] = openai_name

    return o2g, g2o


def _apply_tool_name_map_to_openai_tools(
    openai_tools: list[dict[str, Any]] | None, openai_to_genai: dict[str, str]
) -> list[dict[str, Any]] | None:
    if not openai_tools:
        return None
    out: list[dict[str, Any]] = []
    for tool in openai_tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        fn = dict(fn)
        fn["name"] = openai_to_genai.get(fn["name"], fn["name"])
        if isinstance(fn.get("description"), str) and len(fn["description"]) > 2000:
            fn["description"] = fn["description"][:2000]
        params = fn.get("parameters")
        fn["parameters"] = _sanitize_json_schema(params) if params is not None else {"type": "object"}
        out.append({"type": "function", "function": fn})
    return out or None


def _normalize_tool_choice(body: dict[str, Any]) -> Any:
    tool_choice = body.get("tool_choice")
    if tool_choice is None and "function_call" in body:
        tool_choice = body.get("function_call")
    return tool_choice


def _genai_tool_config(tool_choice: Any, *, allowed_names: list[str] | None) -> Any:
    if tool_choice in (None, "auto"):
        return None
    mode = None
    if tool_choice == "none":
        mode = "NONE"
    elif tool_choice == "required":
        mode = "ANY"
    elif isinstance(tool_choice, dict):
        fn = tool_choice.get("function") if tool_choice.get("type") == "function" else tool_choice
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            mode = "ANY"
            allowed_names = [fn["name"]]
    elif isinstance(tool_choice, str):
        mode = None

    if mode is None:
        return None

    if types is not None:
        ToolConfig = getattr(types, "ToolConfig", None)
        FunctionCallingConfig = getattr(types, "FunctionCallingConfig", None)
        if ToolConfig is not None and FunctionCallingConfig is not None:
            try:
                fcc_kwargs: dict[str, Any] = {"mode": mode}
                if allowed_names:
                    fcc_kwargs["allowed_function_names"] = allowed_names
                fcc = FunctionCallingConfig(**fcc_kwargs)
                return ToolConfig(function_calling_config=fcc)
            except Exception:
                pass

    cfg: dict[str, Any] = {"function_calling_config": {"mode": mode}}
    if allowed_names:
        cfg["function_calling_config"]["allowed_function_names"] = allowed_names
    return cfg


def _genai_config(
    *,
    system_instruction: str | None,
    temperature: float | None,
    top_p: float | None,
    max_output_tokens: int | None,
    stop_sequences: list[str] | None,
    tools: Any,
    tool_config: Any,
) -> Any:
    if types is None:
        cfg: dict[str, Any] = {}
        if system_instruction:
            cfg["system_instruction"] = system_instruction
        if temperature is not None:
            cfg["temperature"] = temperature
        if top_p is not None:
            cfg["top_p"] = top_p
        if max_output_tokens is not None:
            cfg["max_output_tokens"] = max_output_tokens
        if stop_sequences:
            cfg["stop_sequences"] = stop_sequences
        if tools:
            cfg["tools"] = tools
        if tool_config:
            cfg["tool_config"] = tool_config
        return cfg or None

    kwargs: dict[str, Any] = {}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    if stop_sequences:
        kwargs["stop_sequences"] = stop_sequences
    if tools:
        kwargs["tools"] = tools
    if tool_config:
        kwargs["tool_config"] = tool_config
    return types.GenerateContentConfig(**kwargs)


@app.exception_handler(HTTPException)
async def _openai_http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
    if LOG_REQUESTS:
        print(f"HTTPException {exc.status_code}: {message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    models_env = os.getenv("GENAI_MODELS")
    model_ids = [m.strip() for m in (models_env.split(",") if models_env else [DEFAULT_MODEL]) if m.strip()]
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": now, "owned_by": "google"}
            for model_id in model_ids
        ],
    }


def _sse(data: Any) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def _openai_error(
    status_code: int,
    message: str,
    *,
    error_type: str = "api_error",
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


def _parse_retry_after_seconds(message: str) -> int | None:
    if not message:
        return None

    m = re.search(r"retryDelay[\"']\s*:\s*[\"'](\d+)s[\"']", message)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    m = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", message, flags=re.IGNORECASE)
    if m:
        try:
            return max(1, int(math.ceil(float(m.group(1)))))
        except Exception:
            return None

    return None


def _maybe_handle_genai_error(exc: Exception) -> JSONResponse | None:
    if genai_errors is None:
        return None

    if isinstance(exc, getattr(genai_errors, "ClientError", ())):
        status_code = getattr(exc, "status_code", None)
        if not isinstance(status_code, int):
            status_code = 400
        headers: dict[str, str] | None = None
        if status_code == 429:
            retry_after = _parse_retry_after_seconds(str(exc))
            if retry_after is None:
                retry_after = OPENAI_RATELIMIT_RESET_SECONDS
            headers = {"Retry-After": str(retry_after)}
        if LOG_REQUESTS:
            print(f"GenAI ClientError {status_code}: {exc}")
        return _openai_error(
            status_code,
            str(exc),
            error_type="rate_limit_error" if status_code == 429 else "api_error",
            headers=headers,
        )

    if isinstance(exc, getattr(genai_errors, "ServerError", ())):
        status_code = getattr(exc, "status_code", None)
        if not isinstance(status_code, int):
            status_code = 502
        if LOG_REQUESTS:
            print(f"GenAI ServerError {status_code}: {exc}")
        return _openai_error(status_code, str(exc), error_type="api_error")

    return None


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> Any:
    token = _get_auth_token(authorization, x_api_key)
    _require_wrapper_auth(token)

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object.")
    if LOG_REQUESTS:
        print("REQUEST /v1/chat/completions keys:", sorted(body.keys()))
        messages_preview = body.get("messages")
        if isinstance(messages_preview, list):
            print("REQUEST messages count:", len(messages_preview))
            for i, m in enumerate(messages_preview[:8]):
                if not isinstance(m, dict):
                    print(f"  msg[{i}] non-dict:", type(m))
                    continue
                role = m.get("role")
                content = m.get("content")
                ctype = type(content).__name__
                clen = len(content) if isinstance(content, str) else None
                print(f"  msg[{i}] role={role!r} content_type={ctype} content_len={clen}")
                if isinstance(content, list):
                    part_types = []
                    for part in content[:8]:
                        if isinstance(part, dict):
                            part_types.append(str(part.get("type")))
                        else:
                            part_types.append(type(part).__name__)
                    print(f"    content part types: {part_types}")
                if role == "assistant":
                    if "tool_calls" in m:
                        tc = m.get("tool_calls")
                        print(f"    tool_calls type={type(tc).__name__}")
                    if "function_call" in m:
                        fc = m.get("function_call")
                        print(f"    function_call type={type(fc).__name__}")
                if role in {"tool", "function"}:
                    print(f"    tool name={m.get('name')!r} tool_call_id={m.get('tool_call_id')!r}")
        else:
            print("REQUEST messages not list:", type(messages_preview).__name__)

    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(status_code=400, detail="model must be a non-empty string.")
    model = model.strip()

    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be an array.")

    n = body.get("n", 1)
    try:
        n = int(n)
    except Exception:
        n = 1
    if n < 1:
        n = 1

    stream = bool(body.get("stream", False))

    temperature = body.get("temperature")
    top_p = body.get("top_p")

    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="max_tokens must be an integer.") from exc

    stop = body.get("stop")
    if stop is None:
        stop_sequences = None
    elif isinstance(stop, str):
        stop_sequences = [stop]
    elif isinstance(stop, list) and all(isinstance(s, str) for s in stop):
        stop_sequences = stop
    else:
        raise HTTPException(status_code=400, detail="stop must be a string or array of strings.")

    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    openai_tools_raw = _normalize_openai_tools(body)
    if LOG_REQUESTS:
        print("REQUEST tools present:", bool(openai_tools_raw), "count:", len(openai_tools_raw or []))
        print("REQUEST tool_choice:", _normalize_tool_choice(body))
        if openai_tools_raw:
            names: list[str] = []
            for t in openai_tools_raw[:25]:
                if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict):
                    n_ = t["function"].get("name")
                    if isinstance(n_, str):
                        names.append(n_)
            print("REQUEST tool names (first 25):", names)

    openai_to_genai, genai_to_openai = _build_tool_name_maps(openai_tools_raw)
    openai_tools_mapped = _apply_tool_name_map_to_openai_tools(openai_tools_raw, openai_to_genai)
    tools = _genai_tools(openai_tools_mapped)

    tool_choice = _normalize_tool_choice(body)
    tool_choice = _apply_tool_name_map_to_tool_choice(tool_choice, openai_to_genai)
    tool_config = _genai_tool_config(tool_choice, allowed_names=None) if tools else None

    system_instruction, contents = _messages_to_genai(messages, openai_to_genai=openai_to_genai)

    config = _genai_config(
        system_instruction=system_instruction,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
        stop_sequences=stop_sequences,
        tools=tools,
        tool_config=tool_config,
    )

    if stream:
        if tools:
            try:
                response = await _call_genai_with_reauth(
                    lambda: _get_client().models.generate_content(model=model, contents=contents, config=config)
                )
            except Exception as exc:
                err = _maybe_handle_genai_error(exc)
                if err is not None:
                    return err
                raise

            text = _extract_text(response)
            tool_calls = _extract_function_calls(response, genai_to_openai=genai_to_openai)

            async def tool_stream() -> AsyncIterator[str]:
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                )
                delta: dict[str, Any] = {}
                if tool_calls:
                    delta["tool_calls"] = tool_calls
                if text:
                    delta["content"] = text
                if delta:
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                    )
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}],
                    }
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(tool_stream(), media_type="text/event-stream")

        async def event_stream() -> AsyncIterator[str]:
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
            )

            sent = ""

            # Prefer provider streaming if present
            def _make_stream_iter() -> Iterable[Any]:
                c = _get_client()
                stream_fn = getattr(getattr(c, "models", None), "generate_content_stream", None)
                if not callable(stream_fn):
                    raise RuntimeError("generate_content_stream not available")
                return stream_fn(model=model, contents=contents, config=config)

            try:
                stream_iter = _make_stream_iter()
            except Exception as exc:
                if _looks_like_expired_token_error(exc):
                    await _reset_client()
                    stream_iter = _make_stream_iter()
                else:
                    stream_iter = None  # type: ignore[assignment]

            if stream_iter is not None:
                try:
                    async for chunk in iterate_in_threadpool(stream_iter):
                        chunk_text = _extract_text(chunk)
                        if not chunk_text:
                            continue
                        delta = chunk_text[len(sent) :] if chunk_text.startswith(sent) else chunk_text
                        sent = chunk_text if chunk_text.startswith(sent) else (sent + delta)
                        if not delta:
                            continue
                        yield _sse(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                            }
                        )
                except Exception as exc:  # pragma: no cover
                    err = _maybe_handle_genai_error(exc)
                    if err is None:
                        raise
                    yield _sse({"error": err.body.decode("utf-8") if hasattr(err, "body") else str(exc)})
                    yield "data: [DONE]\n\n"
                    return
            else:
                try:
                    response = await _call_genai_with_reauth(
                        lambda: _get_client().models.generate_content(model=model, contents=contents, config=config)
                    )
                    full_text = _extract_text(response)
                except Exception as exc:  # pragma: no cover
                    err = _maybe_handle_genai_error(exc)
                    if err is None:
                        raise
                    yield _sse({"error": err.body.decode("utf-8") if hasattr(err, "body") else str(exc)})
                    yield "data: [DONE]\n\n"
                    return

                chunk_size = 80
                for i in range(0, len(full_text), chunk_size):
                    delta = full_text[i : i + chunk_size]
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                    )

            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        response = await _call_genai_with_reauth(
            lambda: _get_client().models.generate_content(model=model, contents=contents, config=config)
        )
    except Exception as exc:
        err = _maybe_handle_genai_error(exc)
        if err is not None:
            return err
        raise

    text = _extract_text(response)
    tool_calls = _extract_function_calls(response, genai_to_openai=genai_to_openai)
    parallel_tool_calls = body.get("parallel_tool_calls")
    if isinstance(parallel_tool_calls, bool) and not parallel_tool_calls and len(tool_calls) > 1:
        tool_calls = tool_calls[:1]

    prompt_text = "\n".join(_message_text(m) for m in messages if isinstance(m, dict))
    if openai_tools_raw:
        prompt_text = prompt_text + "\n\n" + json.dumps(openai_tools_raw)
    usage = {
        "prompt_tokens": _estimate_tokens(prompt_text),
        "completion_tokens": _estimate_tokens(text),
        "total_tokens": _estimate_tokens(prompt_text) + _estimate_tokens(text),
    }

    if tool_calls:
        if LOG_REQUESTS:
            tool_names = [tc.get("function", {}).get("name") for tc in tool_calls if isinstance(tc, dict)]
            print("RESPONSE assistant tool_calls:", tool_names, "text_len:", len(text))
        if not text:
            text = "Calling tool…"
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text, "tool_calls": tool_calls},
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": usage,
            }
        )

    if LOG_REQUESTS:
        print("RESPONSE assistant message text_len:", len(text))
    if not text:
        text = "[empty response]"
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
