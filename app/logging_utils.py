"""Logging helpers with request ID context."""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


# Context variable to carry the request id through async tasks.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestContextFilter(logging.Filter):
    """Injects request_id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        record.request_id = request_id_var.get("-")
        return True


def configure_logging(level: str) -> None:
    """Configure root logging with a consistent format and request id."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s [req=%(request_id)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stdout,
    )
    logging.getLogger().addFilter(RequestContextFilter())


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns a request id (from header or generated) and exposes it to logs."""

    def __init__(self, app, header_name: str = "x-request-id") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable):
        req_id = request.headers.get(self.header_name) or uuid.uuid4().hex
        token = request_id_var.set(req_id)
        request.state.request_id = req_id
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[self.header_name] = req_id
        return response
