from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import aiosqlite

from app.schemas import Classification, TicketIn

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

    # ---- queue operations -------------------------------------------------
    # Every method taking `token` is fenced on (id, held_by): the token is
    # unique per claim, so a holder whose lease expired and was re-claimed
    # by someone else changes nothing. True only if exactly one row changed.

    async def _fenced(self, sql: str, params: tuple[object, ...]) -> bool:
        cur = await self._conn.execute(sql, params)
        changed = cur.rowcount == 1
        await cur.close()
        return changed

    async def claim_next(
        self, *, token: str, now: float, lease_s: float, max_attempts: int
    ) -> TicketRow | None:
        # One atomic statement: SQLite takes the write lock before running the
        # subquery, so two claimers (even on separate connections) can't both
        # pick the same row.
        async with self._conn.execute(
            "UPDATE tickets SET held_by = :token, held_until = :now + :lease, "
            "attempts = attempts + 1, updated_at = :now "
            "WHERE id = ("
            "  SELECT id FROM tickets"
            "  WHERE status = 'pending'"
            "    AND (held_until IS NULL OR held_until < :now)"
            "    AND (next_attempt_at IS NULL OR next_attempt_at <= :now)"
            "    AND attempts < :max"
            "  ORDER BY created_at, id LIMIT 1"
            ") RETURNING *",
            {"token": token, "now": now, "lease": lease_s, "max": max_attempts},
        ) as cur:
            rows = await cur.fetchall()
        return _to_ticket(rows[0]) if rows else None

    async def renew(
        self, ticket_id: str, token: str, *, now: float, lease_s: float
    ) -> bool:
        return await self._fenced(
            "UPDATE tickets SET held_until = ?, updated_at = ? "
            "WHERE id = ? AND held_by = ?",
            (now + lease_s, now, ticket_id, token),
        )

    async def save_classified(
        self,
        ticket_id: str,
        token: str,
        c: Classification,
        *,
        prompt_version: int,
        now: float,
    ) -> bool:
        return await self._fenced(
            "UPDATE tickets SET status = 'classified', category = ?, "
            "priority = ?, summary = ?, prompt_version = ?, held_by = NULL, "
            "held_until = NULL, next_attempt_at = NULL, last_error = NULL, "
            "updated_at = ? WHERE id = ? AND held_by = ?",
            (
                c.category,
                c.priority,
                c.summary,
                prompt_version,
                now,
                ticket_id,
                token,
            ),
        )

    async def release_for_retry(
        self,
        ticket_id: str,
        token: str,
        *,
        next_attempt_at: float,
        error: str,
        now: float,
    ) -> bool:
        return await self._fenced(
            "UPDATE tickets SET held_by = NULL, held_until = NULL, "
            "next_attempt_at = ?, last_error = ?, updated_at = ? "
            "WHERE id = ? AND held_by = ?",
            (next_attempt_at, error, now, ticket_id, token),
        )

    async def mark_failed(
        self, ticket_id: str, token: str, *, error: str, now: float
    ) -> bool:
        return await self._fenced(
            "UPDATE tickets SET status = 'failed', category = NULL, "
            "priority = NULL, summary = NULL, prompt_version = NULL, "
            "last_error = ?, held_by = NULL, held_until = NULL, "
            "next_attempt_at = NULL, updated_at = ? "
            "WHERE id = ? AND held_by = ?",
            (error, now, ticket_id, token),
        )

    async def release_on_shutdown(
        self, ticket_id: str, token: str, *, now: float
    ) -> bool:
        # The interrupted attempt never finished, so it is given back.
        return await self._fenced(
            "UPDATE tickets SET held_by = NULL, held_until = NULL, "
            "attempts = attempts - 1, updated_at = ? "
            "WHERE id = ? AND held_by = ?",
            (now, ticket_id, token),
        )

    async def sweep_exhausted(self, *, now: float, max_attempts: int) -> int:
        # Not fenced: it targets pending rows at the attempt cap that
        # claim_next will never pick up again: the holder died during the
        # final attempt (expired lease), or the row was released for retry
        # and the cap was lowered by a restart (unheld). A live lease is left alone.
        cur = await self._conn.execute(
            "UPDATE tickets SET status = 'failed', "
            "last_error = CASE WHEN held_until IS NULL "
            "THEN 'attempt limit reached' "
            "ELSE 'worker stopped during final attempt' END, "
            "held_by = NULL, held_until = NULL, updated_at = :now "
            "WHERE status = 'pending' AND attempts >= :max "
            "AND (held_until IS NULL OR held_until < :now)",
            {"now": now, "max": max_attempts},
        )
        count = cur.rowcount
        await cur.close()
        return count

    async def reclassify(
        self,
        target: Literal["failed", "stale_prompt"],
        *,
        current_prompt_version: int,
        now: float,
    ) -> int:
        # Re-queueing is one UPDATE: the workers' normal claim picks the rows
        # up again. Only failed/classified rows can match, so pending rows
        # (held or not) are never touched. `IS NOT` makes a NULL prompt_version
        # count as stale. suspected_injection is kept: it describes the input.
        if target == "failed":
            where = "status = 'failed'"
        else:
            where = "status = 'classified' AND prompt_version IS NOT :v"
        cur = await self._conn.execute(
            "UPDATE tickets SET status = 'pending', attempts = 0, "
            "last_error = NULL, category = NULL, priority = NULL, "
            "summary = NULL, prompt_version = NULL, next_attempt_at = NULL, "
            f"held_by = NULL, held_until = NULL, updated_at = :now WHERE {where}",
            {"now": now, "v": current_prompt_version},
        )
        count = cur.rowcount
        await cur.close()
        return count
