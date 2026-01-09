import json
import os
import re
import time
import uuid
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
    - Does not override existing environment variables.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or key in os.environ:
                    continue
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                os.environ[key] = value
    except FileNotFoundError:
        return


_load_dotenv()
DEFAULT_MODEL = os.getenv("GENAI_MODEL", "gemini-2.0-flash")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
LOG_REQUESTS = (os.getenv("LOG_REQUESTS") or "").strip().lower() in {"1", "true", "yes", "y", "on"}

app = FastAPI(title=APP_TITLE, version="0.1.0")

_client: Any | None = None


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


def _get_client() -> Any:
    global _client
    if _client is not None:
        return _client
    if genai is None:
        raise HTTPException(
            status_code=500,
            detail="google-genai is not installed. Install with: pip install google-genai",
        )
    _client = genai.Client(api_key=_google_api_key())
    return _client


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
    # OpenAI format:
    # - tool_choice: "none"|"auto"|"required"|{"type":"function","function":{"name":"..."}}
    # Legacy:
    # - function_call: "none"|"auto"|{"name":"..."}
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
        # {"type":"function","function":{"name":"..."}}
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


def _openai_error(status_code: int, message: str, *, error_type: str = "api_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


def _maybe_handle_genai_error(exc: Exception) -> JSONResponse | None:
    if genai_errors is None:
        return None

    if isinstance(exc, getattr(genai_errors, "ClientError", ())):
        status_code = getattr(exc, "status_code", None)
        if not isinstance(status_code, int):
            status_code = 400
        if LOG_REQUESTS:
            print(f"GenAI ClientError {status_code}: {exc}")
        return _openai_error(status_code, str(exc), error_type="rate_limit_error" if status_code == 429 else "api_error")

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
        # Avoid logging secrets by design: we don't expect keys in body.
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

    model = body.get("model") or DEFAULT_MODEL
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be an array.")

    # Be permissive; many clients always send n.
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
                    n = t["function"].get("name")
                    if isinstance(n, str):
                        names.append(n)
            print("REQUEST tool names (first 25):", names)
    openai_to_genai, genai_to_openai = _build_tool_name_maps(openai_tools_raw)
    if LOG_REQUESTS and openai_tools_raw:
        print("TOOL name map size:", len(openai_to_genai))
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

    client = _get_client()

    if stream:
        # Streaming tool calls in OpenAI format requires incremental tool_call deltas.
        # For Roo compatibility, if tools are present, do a single non-stream provider call
        # and stream back a minimal chunk sequence.
        if tools:
            try:
                response = await run_in_threadpool(
                    lambda: client.models.generate_content(model=model, contents=contents, config=config)
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

            stream_fn = getattr(getattr(client, "models", None), "generate_content_stream", None)
            if callable(stream_fn):
                stream_iter: Iterable[Any] = stream_fn(model=model, contents=contents, config=config)
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
                    response = await run_in_threadpool(
                        lambda: client.models.generate_content(model=model, contents=contents, config=config)
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
        response = await run_in_threadpool(
            lambda: client.models.generate_content(model=model, contents=contents, config=config)
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
            # Some clients (incl. agent orchestrators) treat empty content + tool calls as missing output.
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
