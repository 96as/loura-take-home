"""HTTP layer: routes, TicketRow -> TicketOut mapping, uniform error shape."""

from __future__ import annotations

import base64
import json
import logging
import math
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.db import InsertResult, TicketRow
from app.injection import looks_like_injection
from app.schemas import Category, Priority, Status, TicketIn, TicketOut, TicketPage

logger = logging.getLogger(__name__)

router = APIRouter()


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


# ---- errors ---------------------------------------------------------------

_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "too_many_requests",
}


def _error_response(
    status: int, code: str, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}}, status_code=status, headers=headers
    )


def _validation_message(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = [str(p) for p in err["loc"] if p not in ("body", "query")]
        msg = err["msg"].removeprefix("Value error, ")
        parts.append(f"{'.'.join(loc)}: {msg}" if loc else msg)
    return "; ".join(parts) or "invalid request"


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(422, "validation_error", _validation_message(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, f"http_{exc.status_code}")
        message = str(exc.detail) if exc.detail else code.replace("_", " ")
        return _error_response(exc.status_code, code, message, dict(exc.headers or {}))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return _error_response(500, "internal_error", "internal server error")


# ---- mapping and cursor ----------------------------------------------------


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def to_ticket_out(row: TicketRow) -> TicketOut:
    return TicketOut(
        id=row.id,
        subject=row.subject,
        body=row.body,
        status=row.status,  # type: ignore[arg-type]
        category=row.category,  # type: ignore[arg-type]
        priority=row.priority,  # type: ignore[arg-type]
        summary=row.summary,
        attempts=row.attempts,
        last_error=row.last_error,
        suspected_injection=row.suspected_injection,
        created_at=_iso(row.created_at),
        updated_at=_iso(row.updated_at),
    )


def encode_cursor(created_at: float, ticket_id: str) -> str:
    raw = json.dumps([created_at, ticket_id]).encode()
    return base64.urlsafe_b64encode(raw).decode()


def decode_cursor(cursor: str) -> tuple[float, str]:
    try:
        value = json.loads(base64.b64decode(cursor.encode(), altchars=b"-_", validate=True))
    except ValueError:  # bad base64, bad UTF-8 or bad JSON
        raise ApiError(422, "invalid_cursor", "cursor is not valid") from None
    if (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (int, float))
        and not isinstance(value[0], bool)
        and math.isfinite(value[0])
        and isinstance(value[1], str)
    ):
        return float(value[0]), value[1]
    raise ApiError(422, "invalid_cursor", "cursor is not valid")


# ---- routes ----------------------------------------------------------------


@router.post("/tickets", response_model=TicketOut, status_code=202)
async def create_ticket(ticket: TicketIn, request: Request, response: Response) -> TicketOut:
    state = request.app.state
    flagged = looks_like_injection(ticket.subject, ticket.body)
    result, row = await state.db.insert_ticket(
        ticket, suspected_injection=flagged, now=state.clock()
    )
    if result is InsertResult.CONFLICT:
        raise ApiError(
            409, "conflict", f"ticket {ticket.id!r} already exists with different content"
        )
    if result is InsertResult.DUPLICATE:
        response.status_code = 200
    return to_ticket_out(row)


@router.get("/tickets/{ticket_id}", response_model=TicketOut)
async def get_ticket(ticket_id: str, request: Request) -> TicketOut:
    row = await request.app.state.db.get_ticket(ticket_id)
    if row is None:
        raise ApiError(404, "not_found", f"ticket {ticket_id!r} not found")
    return to_ticket_out(row)


@router.get("/tickets", response_model=TicketPage)
async def list_tickets(
    request: Request,
    category: Category | None = None,
    priority: Priority | None = None,
    status: Status | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: str | None = None,
) -> TicketPage:
    after = decode_cursor(cursor) if cursor is not None else None
    rows = await request.app.state.db.list_tickets(
        category=category, priority=priority, status=status, limit=limit + 1, after=after
    )
    page = rows[:limit]
    next_cursor = (
        encode_cursor(page[-1].created_at, page[-1].id) if len(rows) > limit else None
    )
    return TicketPage(items=[to_ticket_out(r) for r in page], next_cursor=next_cursor)
