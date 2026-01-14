# CHANGES IN THIS ITERATION (high-level)
# 1) Added MODEL REGISTRY (models.json) to map model -> provider + location/region.
# 2) Added Anthropic Vertex provider support (anthropic_vertex) using anthropic-sdk-python.
# 3) Added per-provider client caches: one for GenAI, one for Anthropic Vertex.
# 4) Unified retry-on-auth-expiry: on 401/403/expired/unauth -> reset provider cache -> retry once.
# 5) /v1/models now returns models from the registry (explicitly configured = explicitly available).
# 6) Added Anthropic tool call <-> OpenAI tool_calls translation (basic, but works for Roo-style loop).

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

# Anthropic Vertex (optional dependency)
try:
    # anthropic-sdk-python
    # Vertex usage is typically: from anthropic import AnthropicVertex
    from anthropic import AnthropicVertex  # type: ignore
    from anthropic import APIError as AnthropicAPIError  # type: ignore
except Exception:  # pragma: no cover
    AnthropicVertex = None
    AnthropicAPIError = None


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
                value = value.strip()

                if not override and key in os.environ:
                    continue
                os.environ[key] = value
    except FileNotFoundError:
        return


_load_dotenv()

DEFAULT_MODEL = "gemini-2.0-flash"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
LOG_REQUESTS = (os.getenv("LOG_REQUESTS") or "").strip().lower() in {"1", "true", "yes", "y", "on"}

OPENAI_RATELIMIT_LIMIT_REQUESTS = int(os.getenv("OPENAI_RATELIMIT_LIMIT_REQUESTS", "9999"))
OPENAI_RATELIMIT_LIMIT_TOKENS = int(os.getenv("OPENAI_RATELIMIT_LIMIT_TOKENS", "999999"))
OPENAI_RATELIMIT_RESET_SECONDS = int(os.getenv("OPENAI_RATELIMIT_RESET_SECONDS", "1"))

MODEL_REGISTRY_PATH = (os.getenv("MODEL_REGISTRY_PATH") or "models.json").strip()

app = FastAPI(title=APP_TITLE, version="0.2.0")

# Provider caches
_genai_clients: dict[tuple[Any, ...], Any] = {}
_anthropic_clients: dict[tuple[Any, ...], Any] = {}

_cache_lock = asyncio.Lock()

# Model registry loaded at startup
MODEL_REGISTRY: dict[str, dict[str, Any]] = {}


def _load_model_registry() -> dict[str, dict[str, Any]]:
    """
    Load models.json:
      {
        "gemini-2.5-pro": {"provider":"genai","location":"global"},
        "claude-3-5-sonnet@20240620": {"provider":"anthropic_vertex","region":"us-east5"}
      }
    """
    if not os.path.exists(MODEL_REGISTRY_PATH):
        # Allow empty registry; fallback to DEFAULT_MODEL advertised
        return {DEFAULT_MODEL: {"provider": "genai", "location": "global"}}

    try:
        with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {MODEL_REGISTRY_PATH}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(f"{MODEL_REGISTRY_PATH} must be a JSON object mapping model->config")

    out: dict[str, dict[str, Any]] = {}
    for model, cfg in raw.items():
        if not isinstance(model, str) or not model.strip():
            continue
        if not isinstance(cfg, dict):
            continue
        provider = (cfg.get("provider") or "").strip().lower()
        if provider not in {"genai", "anthropic_vertex"}:
            continue

        if provider == "genai":
            location = (cfg.get("location") or "").strip() or None
            out[model.strip()] = {"provider": "genai", "location": location or "global"}
        else:
            region = (cfg.get("region") or "").strip() or None
            out[model.strip()] = {"provider": "anthropic_vertex", "region": region or (os.getenv("ANTHROPIC_REGION") or "us-east5")}

    if not out:
        out = {DEFAULT_MODEL: {"provider": "genai", "location": "global"}}
    return out


try:
    MODEL_REGISTRY = _load_model_registry()
except Exception as e:
    # fail fast; misconfigured registry should be obvious
    raise


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


def _auth_mode() -> str:
    mode = (os.getenv("GENAI_AUTH_MODE") or "personal").strip().lower()
    if mode not in {"personal", "work"}:
        mode = "personal"
    return mode


def _google_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="Missing GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable.",
        )
    return key


def _helix_token() -> str:
    cmd = (os.getenv("HELIX_TOKEN_COMMAND") or "").strip()
    if not cmd:
        raise HTTPException(
            status_code=500,
            detail="GENAI_AUTH_MODE=work but HELIX_TOKEN_COMMAND is not set.",
        )

    try:
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


def _filter_kwargs_for_ctor(ctor: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(ctor)
    except Exception:
        return kwargs
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted and v is not None}


def _looks_like_auth_error(exc: Exception) -> bool:
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
        "api_key_invalid",
    )
    if any(n in msg for n in needles):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code in (401, 403):
        return True

    return False


async def _reset_provider(provider: str) -> None:
    async with _cache_lock:
        if provider == "genai":
            _genai_clients.clear()
        elif provider == "anthropic_vertex":
            _anthropic_clients.clear()


async def _call_with_reauth(provider: str, fn, *, retry: bool = True):
    try:
        return await run_in_threadpool(fn)
    except Exception as exc:
        if retry and _looks_like_auth_error(exc):
            if LOG_REQUESTS:
                print(f"[{provider}] Auth/token failure detected; resetting provider cache and retrying once:", repr(exc))
            await _reset_provider(provider)
            return await _call_with_reauth(provider, fn, retry=False)
        raise


def _get_genai_client(*, location: str | None) -> Any:
    if genai is None:
        raise HTTPException(status_code=500, detail="google-genai is not installed. Install with: pip install google-genai")

    mode = _auth_mode()
    endpoint = (os.getenv("GENAI_API_ENDPOINT") or "").strip() or None

    vertexai_flag = (os.getenv("GENAI_VERTEXAI") or "").strip().lower() in {"1", "true", "yes", "y", "on"}
    project = (os.getenv("GENAI_PROJECT") or "").strip() or None
    # location can come from per-model registry, fallback to env
    loc = (location or (os.getenv("GENAI_LOCATION") or "").strip() or None)

    cache_key = (mode, bool(vertexai_flag), project, loc, endpoint)

    if cache_key in _genai_clients:
        return _genai_clients[cache_key]

    if mode == "personal":
        k = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        kwargs = {"api_key": _google_api_key()}
        kwargs = _filter_kwargs_for_ctor(genai.Client, kwargs)
        if LOG_REQUESTS:
            print(f"Auth mode=personal; api_key_present={bool(k)}; api_key_suffix={k[-6:] if k else ''}")
        c = genai.Client(**kwargs)  # type: ignore[misc]
        _genai_clients[cache_key] = c
        return c

    # work mode
        # work mode
    client_options = {"api_endpoint": endpoint} if endpoint else None
    kwargs: dict[str, Any] = {}

    if vertexai_flag:
        # Vertex mode: credentials + project + location ONLY
        if not project or not loc:
            raise HTTPException(
                status_code=500,
                detail="GENAI_VERTEXAI=1 requires GENAI_PROJECT and GENAI_LOCATION",
            )

        from google.auth.credentials import Credentials  # type: ignore

        class HelixCredentials(Credentials):
            def refresh(self, request):
                self.token = _helix_token()

            @property
            def expired(self):
                return False

            @property
            def valid(self):
                return True

        creds = HelixCredentials()
        creds.token = _helix_token()

        kwargs.update(
            {
                "vertexai": True,
                "project": project,
                "location": loc,
                "credentials": creds,
            }
        )
    else:
        # Non-Vertex work mode (internal gateway pattern)
        kwargs["api_key"] = _helix_token()

    if client_options:
        kwargs["client_options"] = client_options

    kwargs = _filter_kwargs_for_ctor(genai.Client, kwargs)

    if LOG_REQUESTS:
        safe = {k: ("***" if k == "api_key" else v) for k, v in kwargs.items()}
        print("Initializing WORK GenAI client with kwargs:", safe)

    c = genai.Client(**kwargs)  # type: ignore[misc]
    _genai_clients[cache_key] = c
    return c


def _get_anthropic_vertex_client(*, region: str | None) -> Any:
    if AnthropicVertex is None:
        raise HTTPException(
            status_code=500,
            detail="anthropic-sdk-python is not installed. Install with: pip install anthropic",
        )

    mode = _auth_mode()
    endpoint = (os.getenv("ANTHROPIC_VERTEX_ENDPOINT") or "").strip() or None

    project = (os.getenv("ANTHROPIC_PROJECT") or "").strip() or (os.getenv("GENAI_PROJECT") or "").strip() or None
    if not project:
        # Vertex needs a GCP project
        raise HTTPException(status_code=500, detail="Missing ANTHROPIC_PROJECT (or GENAI_PROJECT) for Anthropic Vertex.")

    reg = (region or (os.getenv("ANTHROPIC_REGION") or "").strip() or "us-east5")

    cache_key = (mode, project, reg, endpoint)
    if cache_key in _anthropic_clients:
        return _anthropic_clients[cache_key]

    # In personal mode, Vertex auth usually relies on ADC (gcloud auth application-default login).
    # In work mode, your environment may already provide suitable Google auth OR you’ll rely on the token system.
    # We keep it signature-safe and only pass supported kwargs.
    kwargs: dict[str, Any] = {"project_id": project, "region": reg}
    if endpoint:
        # Some SDK versions support base_url / endpoint overrides; pass only if constructor accepts it
        kwargs["base_url"] = endpoint
        kwargs["endpoint"] = endpoint

    kwargs = _filter_kwargs_for_ctor(AnthropicVertex, kwargs)

    if LOG_REQUESTS:
        safe = dict(kwargs)
        print("Initializing AnthropicVertex client with kwargs:", safe)

    c = AnthropicVertex(**kwargs)  # type: ignore[misc]
    _anthropic_clients[cache_key] = c
    return c


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
            raise HTTPException(status_code=400, detail="Only text content parts are supported for MVP.")
        return "".join(parts)
    raise HTTPException(status_code=400, detail="Invalid message.content")


def _sanitize_json_schema(schema: Any, *, depth: int = 0) -> dict[str, Any]:
    if depth > 20:
        return {"type": "string"}
    if not isinstance(schema, dict):
        return {"type": "string"}
    
    schema.pop("additional_properties", None)
    schema.pop("additionalProperties", None)

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
    o2p: dict[str, str] = {}
    p2o: dict[str, str] = {}

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
        provider_name = base
        if provider_name in used:
            provider_name = (base[:57] + "_" + uuid.uuid4().hex[:6])[:64]
        used.add(provider_name)
        o2p[openai_name] = provider_name
        p2o[provider_name] = openai_name

    return o2p, p2o


def _openai_tools_to_anthropic(openai_tools_mapped: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not openai_tools_mapped:
        return None
    out: list[dict[str, Any]] = []
    for t in openai_tools_mapped:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        desc = fn.get("description")
        params = fn.get("parameters")
        tool = {
            "name": name,
            "description": desc if isinstance(desc, str) else None,
            "input_schema": _sanitize_json_schema(params) if params is not None else {"type": "object"},
        }
        out.append(tool)
    return out or None


def _normalize_tool_choice(body: dict[str, Any]) -> Any:
    tool_choice = body.get("tool_choice")
    if tool_choice is None and "function_call" in body:
        tool_choice = body.get("function_call")
    return tool_choice


def _anthropic_tool_choice(tool_choice: Any) -> Any:
    # Anthropic uses tool_choice={"type":"auto"} or {"type":"tool","name":"..."} or {"type":"any"}.
    if tool_choice in (None, "auto"):
        return {"type": "auto"}
    if tool_choice == "none":
        return {"type": "none"}
    if tool_choice == "required":
        return {"type": "any"}
    if isinstance(tool_choice, dict):
        # OpenAI shape: {"type":"function","function":{"name":"..."}}
        if tool_choice.get("type") == "function" and isinstance(tool_choice.get("function"), dict):
            nm = tool_choice["function"].get("name")
            if isinstance(nm, str) and nm:
                return {"type": "tool", "name": nm}
        # Legacy: {"name":"..."}
        nm = tool_choice.get("name")
        if isinstance(nm, str) and nm:
            return {"type": "tool", "name": nm}
    return {"type": "auto"}


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
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
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
            retry_after = _parse_retry_after_seconds(str(exc)) or OPENAI_RATELIMIT_RESET_SECONDS
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


@app.exception_handler(HTTPException)
async def _openai_http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
    if LOG_REQUESTS:
        print(f"HTTPException {exc.status_code}: {message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    now = int(time.time())
    model_ids = sorted(MODEL_REGISTRY.keys()) if MODEL_REGISTRY else [DEFAULT_MODEL]
    return {
        "object": "list",
        "data": [{"id": mid, "object": "model", "created": now, "owned_by": "router"} for mid in model_ids],
    }


def _extract_genai_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
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


def _extract_genai_function_calls(response: Any, *, provider_to_openai: dict[str, str] | None = None) -> list[dict[str, Any]]:
    provider_to_openai = provider_to_openai or {}

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
            openai_name = provider_to_openai.get(name, name)
            if args is None:
                args = {}
            if not isinstance(args, dict):
                args = {"_raw": args}
            calls.append({"id": f"call_{uuid.uuid4().hex}", "type": "function", "function": {"name": openai_name, "arguments": json.dumps(args)}})
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
        openai_name = provider_to_openai.get(name, name)
        if args is None:
            args = {}
        if not isinstance(args, dict):
            args = {"_raw": args}
        calls.append({"id": f"call_{uuid.uuid4().hex}", "type": "function", "function": {"name": openai_name, "arguments": json.dumps(args)}})
    return calls


def _messages_to_anthropic(
    messages: list[dict[str, Any]],
    *,
    openai_to_provider: dict[str, str] | None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Convert OpenAI chat messages (+ tool results) to Anthropic Messages API format.
    - system: separate string
    - messages: [{"role":"user"/"assistant","content":[...blocks...]}]
    Tool results: OpenAI sends role="tool" with tool_call_id; Anthropic expects "tool_result" blocks.
    """
    openai_to_provider = openai_to_provider or {}

    system_chunks: list[str] = []
    out_msgs: list[dict[str, Any]] = []

    # Map OpenAI tool_call_id -> Anthropic tool_use_id (we generate ids deterministically in our translation)
    # In practice: when we emit tool_calls to Roo, the tool_call_id will be whatever we emitted.
    # For Anthropic, we’ll reuse that same id as tool_use_id so round-trip is clean.
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in {"system", "developer"}:
            system_chunks.append(_message_text(m))
            continue

        if role == "user":
            txt = _message_text(m)
            out_msgs.append({"role": "user", "content": [{"type": "text", "text": txt}]})
            continue

        if role == "assistant":
            txt = _message_text(m)
            tool_calls = m.get("tool_calls")
            blocks: list[dict[str, Any]] = []
            if txt:
                blocks.append({"type": "text", "text": txt})

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
                    provider_name = openai_to_provider.get(openai_name, openai_name)

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

                    tool_use_id = tc.get("id")
                    if not isinstance(tool_use_id, str) or not tool_use_id:
                        tool_use_id = f"call_{uuid.uuid4().hex}"

                    blocks.append({"type": "tool_use", "id": tool_use_id, "name": provider_name, "input": args})

            out_msgs.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            continue

        if role in {"tool", "function"}:
            tool_call_id = m.get("tool_call_id")
            name = m.get("name")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise HTTPException(status_code=400, detail="tool messages must include tool_call_id for Anthropic flow.")
            # content may be JSON string; treat as text for tool_result content
            content = m.get("content")
            if isinstance(content, (dict, list)):
                content_text = json.dumps(content)
            elif content is None:
                content_text = ""
            else:
                content_text = str(content)

            # Anthropic tool_result must be sent as a USER message
            out_msgs.append(
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": content_text}],
                }
            )
            continue

        raise HTTPException(status_code=400, detail=f"Unsupported role: {role!r}")

    system = "\n\n".join([c for c in system_chunks if c.strip()]) or None
    return system, out_msgs


def _extract_anthropic_text_and_tool_calls(resp: Any, *, provider_to_openai: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    """
    Anthropic Messages response typically has .content list with blocks:
      - {"type":"text","text":"..."}
      - {"type":"tool_use","id":"...","name":"...","input":{...}}
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    content = getattr(resp, "content", None)
    if content is None and isinstance(resp, dict):
        content = resp.get("content")
    if content is None:
        return "", []

    try:
        blocks = list(content)
    except Exception:
        blocks = []

    for b in blocks:
        if isinstance(b, dict):
            btype = b.get("type")
            if btype == "text" and isinstance(b.get("text"), str):
                text_parts.append(b["text"])
            elif btype == "tool_use":
                tid = b.get("id")
                nm = b.get("name")
                inp = b.get("input")
                if not isinstance(tid, str) or not tid:
                    tid = f"call_{uuid.uuid4().hex}"
                if not isinstance(nm, str) or not nm:
                    continue
                openai_name = provider_to_openai.get(nm, nm)
                if inp is None:
                    inp = {}
                if not isinstance(inp, dict):
                    inp = {"_raw": inp}
                tool_calls.append(
                    {
                        "id": tid,  # IMPORTANT: keep stable so tool_result can reference it
                        "type": "function",
                        "function": {"name": openai_name, "arguments": json.dumps(inp)},
                    }
                )

    return "".join(text_parts), tool_calls


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

    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(status_code=400, detail="model must be a non-empty string.")
    model = model.strip()

    if model not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Model {model!r} is not configured in {MODEL_REGISTRY_PATH}")

    provider = MODEL_REGISTRY[model].get("provider")
    if provider not in {"genai", "anthropic_vertex"}:
        raise HTTPException(status_code=400, detail=f"Model {model!r} has invalid provider in registry")

    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be an array.")

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
    openai_to_provider, provider_to_openai = _build_tool_name_maps(openai_tools_raw)

    # tool_choice mapping (shared)
    tool_choice = _normalize_tool_choice(body)
    if isinstance(tool_choice, dict):
        # if it names a tool, rename to provider tool name
        if tool_choice.get("type") == "function" and isinstance(tool_choice.get("function"), dict):
            fn = dict(tool_choice["function"])
            if isinstance(fn.get("name"), str):
                fn["name"] = openai_to_provider.get(fn["name"], fn["name"])
            tool_choice = {"type": "function", "function": fn}
        elif isinstance(tool_choice.get("name"), str):
            tool_choice = dict(tool_choice)
            tool_choice["name"] = openai_to_provider.get(tool_choice["name"], tool_choice["name"])

    # Provider execution
    if provider == "genai":
        # Build google-genai tools/config (reuse your existing patterns where possible)
        # For this iteration: minimal, use same approach as before for tools via google-genai types.
        # We keep your previous behavior: if tools + stream => non-stream provider call and then SSE minimal chunks.
        # (To keep this file smaller, this iteration focuses on Anthropic add; GenAI path uses the same core approach.)

        # --- reuse your existing genai conversion helpers via google-genai 'types' if installed ---
        # Minimal inline: convert messages to genai "contents" like you already had.
        # (We keep text-only + tool call support in messages path for Roo loops.)

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
            t = content.strip()
            if not t:
                return {}
            try:
                return json.loads(t)
            except Exception:
                return {"content": content}

        def _messages_to_genai(messages_in: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
            system_chunks: list[str] = []
            contents: list[dict[str, Any]] = []
            tool_call_id_to_name: dict[str, str] = {}

            for message in messages_in:
                if not isinstance(message, dict):
                    raise HTTPException(status_code=400, detail="Each message must be an object.")

                role = message.get("role")
                if role in {"system", "developer"}:
                    system_chunks.append(_message_text(message))
                    continue

                if role == "assistant":
                    tool_calls = message.get("tool_calls")
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
                            name = openai_to_provider.get(openai_name, openai_name)
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
                        raise HTTPException(status_code=400, detail="tool messages must include name or tool_call_id matching a prior tool_call.")
                    provider_name = openai_to_provider.get(name, name)
                    contents.append({"role": "user", "parts": [_part_function_response(provider_name, _parse_tool_result(message.get("content")))]})
                    continue

                if role == "user":
                    contents.append({"role": "user", "parts": [_part_text(_message_text(message))]})
                    continue

                raise HTTPException(status_code=400, detail=f"Unsupported role: {role!r}")

            system_instruction = "\n\n".join([c for c in system_chunks if c.strip()]) or None
            return system_instruction, contents

        def _genai_tools(openai_tools: Any) -> Any:
            """
            Convert OpenAI-style tools to Gemini tool declarations.

            IMPORTANT:
            1) We return ONLY dict-based declarations (no google.genai.types objects).
            2) We recursively STRIP schema keys that Gemini rejects (notably additionalProperties).
            Gemini's error message shows `additional_properties`, but it may come from input
            `additionalProperties` (protobuf field name normalization).
            """
            if openai_tools is None:
                return None
            if not isinstance(openai_tools, list):
                raise HTTPException(status_code=400, detail="tools must be an array.")

            def _scrub(obj: Any) -> Any:
                # Recursively remove keys Gemini rejects.
                # NOTE: Gemini complains about "additional_properties" but the input may be "additionalProperties".
                banned = {"additionalProperties", "additional_properties", "patternProperties", "unevaluatedProperties"}
                if isinstance(obj, dict):
                    out = {}
                    for k, v in obj.items():
                        if k in banned:
                            continue
                        out[k] = _scrub(v)
                    return out
                if isinstance(obj, list):
                    return [_scrub(x) for x in obj]
                return obj

            declarations: list[dict[str, Any]] = []
            for tool in openai_tools:
                if not isinstance(tool, dict) or tool.get("type") != "function":
                    continue
                fn = tool.get("function")
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    continue

                name = fn["name"]
                description = fn.get("description") if isinstance(fn.get("description"), str) else None
                parameters = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object"}

                declarations.append(
                    {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    }
                )

            if not declarations:
                return None

            tools_payload = [{"function_declarations": declarations}]
            return _scrub(tools_payload)



        def _genai_config(
            *,
            system_instruction: str | None,
            temperature: float | None,
            top_p: float | None,
            max_output_tokens: int | None,
            stop_sequences: list[str] | None,
            tools: Any,
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
            return types.GenerateContentConfig(**kwargs)

        system_instruction, contents = _messages_to_genai(messages)
        tools_obj = _genai_tools(openai_tools_raw)
        config = _genai_config(
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            stop_sequences=stop_sequences,
            tools=tools_obj,
        )

        location = MODEL_REGISTRY[model].get("location")
        client = _get_genai_client(location=location)

        if stream and tools_obj:
            # non-stream provider call, SSE minimal
            try:
                response = await _call_with_reauth("genai", lambda: client.models.generate_content(model=model, contents=contents, config=config))
            except Exception as exc:
                err = _maybe_handle_genai_error(exc)
                if err is not None:
                    return err
                raise

            text = _extract_genai_text(response)
            tool_calls = _extract_genai_function_calls(response, provider_to_openai=provider_to_openai)

            async def tool_stream() -> AsyncIterator[str]:
                yield _sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                delta: dict[str, Any] = {}
                if tool_calls:
                    delta["tool_calls"] = tool_calls
                if text:
                    delta["content"] = text
                if delta:
                    yield _sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
                yield _sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]})
                yield "data: [DONE]\n\n"

            return StreamingResponse(tool_stream(), media_type="text/event-stream")

        # non-stream (or stream without tools can be added back later)
        try:
            response = await _call_with_reauth("genai", lambda: client.models.generate_content(model=model, contents=contents, config=config))
        except Exception as exc:
            err = _maybe_handle_genai_error(exc)
            if err is not None:
                return err
            raise

        text = _extract_genai_text(response)
        tool_calls = _extract_genai_function_calls(response, provider_to_openai=provider_to_openai)

    else:
        # Anthropic Vertex provider
        region = MODEL_REGISTRY[model].get("region")
        client = _get_anthropic_vertex_client(region=region)

        # Tools
        openai_tools_mapped: list[dict[str, Any]] | None = None
        if openai_tools_raw:
            # apply name mapping + schema sanitize
            openai_tools_mapped = []
            for t in openai_tools_raw:
                if not isinstance(t, dict) or t.get("type") != "function":
                    continue
                fn = t.get("function")
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    continue
                fn2 = dict(fn)
                fn2["name"] = openai_to_provider.get(fn2["name"], fn2["name"])
                if isinstance(fn2.get("description"), str) and len(fn2["description"]) > 2000:
                    fn2["description"] = fn2["description"][:2000]
                params = fn2.get("parameters")
                fn2["parameters"] = _sanitize_json_schema(params) if params is not None else {"type": "object"}
                openai_tools_mapped.append({"type": "function", "function": fn2})
            if not openai_tools_mapped:
                openai_tools_mapped = None

        anthropic_tools = _openai_tools_to_anthropic(openai_tools_mapped)
        anthropic_tool_choice = _anthropic_tool_choice(tool_choice) if anthropic_tools else {"type": "none"}

        system_instruction, anthropic_messages = _messages_to_anthropic(messages, openai_to_provider=openai_to_provider)

        # Anthropic: stop_sequences supported, temperature/top_p supported in some models; keep simple
        def _anthropic_call():
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens or 1024,
                "messages": anthropic_messages,
            }
            if system_instruction:
                kwargs["system"] = system_instruction
            if isinstance(temperature, (int, float)):
                kwargs["temperature"] = float(temperature)
            if isinstance(top_p, (int, float)):
                kwargs["top_p"] = float(top_p)
            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools
                kwargs["tool_choice"] = anthropic_tool_choice

            return client.messages.create(**kwargs)

        if stream and anthropic_tools:
            # Same Roo-compat approach: do non-stream provider call and stream minimal deltas
            try:
                response = await _call_with_reauth("anthropic_vertex", _anthropic_call)
            except Exception as exc:
                raise

            text, tool_calls = _extract_anthropic_text_and_tool_calls(response, provider_to_openai=provider_to_openai)

            async def tool_stream() -> AsyncIterator[str]:
                yield _sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
                delta: dict[str, Any] = {}
                if tool_calls:
                    delta["tool_calls"] = tool_calls
                if text:
                    delta["content"] = text
                if delta:
                    yield _sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
                yield _sse({"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]})
                yield "data: [DONE]\n\n"

            return StreamingResponse(tool_stream(), media_type="text/event-stream")

        try:
            response = await _call_with_reauth("anthropic_vertex", _anthropic_call)
        except Exception as exc:
            raise

        text, tool_calls = _extract_anthropic_text_and_tool_calls(response, provider_to_openai=provider_to_openai)

    # Common OpenAI-compatible response assembly
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
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text, "tool_calls": tool_calls}, "finish_reason": "tool_calls"}],
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
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": usage,
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
