from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import aiosqlite

from app.schemas import TicketIn

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id                  TEXT PRIMARY KEY,
    subject             TEXT NOT NULL,
    body                TEXT NOT NULL,
    status              TEXT NOT NULL
                        CHECK (status IN ('pending', 'classified', 'failed')),
    category            TEXT,
    priority            TEXT,
    summary             TEXT,
    prompt_version      INTEGER,
    suspected_injection INTEGER NOT NULL,
    attempts            INTEGER NOT NULL DEFAULT 0,
    last_error          TEXT,
    held_by             TEXT,
    held_until          REAL,
    next_attempt_at     REAL,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_claim
    ON tickets (status, created_at, id);
CREATE INDEX IF NOT EXISTS idx_tickets_created
    ON tickets (created_at, id);
CREATE INDEX IF NOT EXISTS idx_tickets_category_priority
    ON tickets (category, priority);
"""

_COLUMNS = (
    "id, subject, body, status, category, priority, summary, prompt_version, "
    "suspected_injection, attempts, last_error, held_by, held_until, "
    "next_attempt_at, created_at, updated_at"
)


@dataclass
class TicketRow:
    id: str
    subject: str
    body: str
    status: str
    category: str | None
    priority: str | None
    summary: str | None
    prompt_version: int | None
    suspected_injection: bool
    attempts: int
    last_error: str | None
    held_by: str | None
    held_until: float | None
    next_attempt_at: float | None
    created_at: float
    updated_at: float


class InsertResult(Enum):
    CREATED = "created"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


def _to_ticket(row: aiosqlite.Row) -> TicketRow:
    d = dict(row)
    d["suspected_injection"] = bool(d["suspected_injection"])
    return TicketRow(**d)


class Database:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def connect(cls, path: str) -> Database:
        conn = await aiosqlite.connect(path, isolation_level=None)
        try:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")
        except BaseException:
            await conn.close()
            raise
        return cls(conn)

    async def close(self) -> None:
        await self._conn.close()

    async def create_schema(self) -> None:
        await self._conn.executescript(_SCHEMA)

    async def insert_ticket(
        self, t: TicketIn, *, suspected_injection: bool, now: float
    ) -> tuple[InsertResult, TicketRow]:
        cur = await self._conn.execute(
            "INSERT INTO tickets (id, subject, body, status, suspected_injection, "
            "attempts, created_at, updated_at) "
            "VALUES (?, ?, ?, 'pending', ?, 0, ?, ?) "
            "ON CONFLICT(id) DO NOTHING",
            (t.id, t.subject, t.body, int(suspected_injection), now, now),
        )
        created = cur.rowcount == 1
        await cur.close()
        row = await self.get_ticket(t.id)
        assert row is not None  # the PK exists either way; rows are never deleted
        if created:
            return InsertResult.CREATED, row
        same = row.subject == t.subject and row.body == t.body
        return (InsertResult.DUPLICATE if same else InsertResult.CONFLICT), row

    async def get_ticket(self, ticket_id: str) -> TicketRow | None:
        async with self._conn.execute(
            f"SELECT {_COLUMNS} FROM tickets WHERE id = ?", (ticket_id,)
        ) as cur:
            row = await cur.fetchone()
        return _to_ticket(row) if row else None

    async def list_tickets(
        self,
        *,
        category: str | None,
        priority: str | None,
        status: str | None,
        limit: int,
        after: tuple[float, str] | None,
    ) -> list[TicketRow]:
        # Column names are fixed literals; every value is a bound parameter.
        clauses: list[str] = []
        params: list[object] = []
        for column, value in (
            ("category", category),
            ("priority", priority),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if after is not None:
            clauses.append("(created_at < ? OR (created_at = ? AND id < ?))")
            params.extend((after[0], after[0], after[1]))
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        params.append(limit)
        async with self._conn.execute(
            f"SELECT {_COLUMNS} FROM tickets {where}"
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            params,
        ) as cur:
            rows = await cur.fetchall()
        return [_to_ticket(r) for r in rows]
