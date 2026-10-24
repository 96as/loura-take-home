import asyncio

from app.db import InsertResult
from app.schemas import TicketIn

NOW = 1_700_000_000.0


def ticket(id_="t1", subject="Hello", body="Body text") -> TicketIn:
    return TicketIn(id=id_, subject=subject, body=body)


async def add(db, id_, now, **kw):
    result, row = await db.insert_ticket(
        ticket(id_), suspected_injection=False, now=now
    )
    assert result is InsertResult.CREATED
    if kw:
        sets = ", ".join(f"{k} = ?" for k in kw)
        await db._conn.execute(
            f"UPDATE tickets SET {sets} WHERE id = ?", (*kw.values(), id_)
        )
    return row


async def test_create_schema_idempotent(db):
    await db.create_schema()
    await db.create_schema()


async def test_insert_new_is_pending(db):
    result, row = await db.insert_ticket(
        ticket(), suspected_injection=True, now=NOW
    )
    assert result is InsertResult.CREATED
    assert row.id == "t1"
    assert row.subject == "Hello"
    assert row.body == "Body text"
    assert row.status == "pending"
    assert row.attempts == 0
    assert row.suspected_injection is True
    assert row.category is None
    assert row.priority is None
    assert row.summary is None
    assert row.prompt_version is None
    assert row.last_error is None
    assert row.held_by is None
    assert row.held_until is None
    assert row.next_attempt_at is None
    assert row.created_at == NOW
    assert row.updated_at == NOW
    assert await db.get_ticket("t1") == row


async def test_insert_same_content_is_duplicate(db):
    _, first = await db.insert_ticket(ticket(), suspected_injection=False, now=NOW)
    result, row = await db.insert_ticket(
        ticket(), suspected_injection=True, now=NOW + 100
    )
    assert result is InsertResult.DUPLICATE
    assert row == first
    assert row.created_at == NOW
    assert await db.get_ticket("t1") == first


async def test_insert_different_content_is_conflict(db):
    _, first = await db.insert_ticket(ticket(), suspected_injection=False, now=NOW)
    for other in (ticket(subject="Other"), ticket(body="Other body")):
        result, row = await db.insert_ticket(
            other, suspected_injection=False, now=NOW + 1
        )
        assert result is InsertResult.CONFLICT
        assert row == first


async def test_concurrent_same_id_one_created(db):
    results = await asyncio.gather(
        *(
            db.insert_ticket(ticket(), suspected_injection=False, now=NOW)
            for _ in range(5)
        )
    )
    outcomes = sorted(r.value for r, _ in results)
    assert outcomes == sorted(
        [InsertResult.CREATED.value] + [InsertResult.DUPLICATE.value] * 4
    )
    assert len({row.created_at for _, row in results}) == 1


async def test_get_missing_returns_none(db):
    assert await db.get_ticket("nope") is None


async def test_list_order_and_filters(db):
    await add(db, "a", NOW + 1, status="classified", category="billing", priority="high")
    await add(db, "b", NOW + 2, status="classified", category="billing", priority="low")
    await add(db, "c", NOW + 3, status="failed", category="technical", priority="high")
    await add(db, "d", NOW + 4, status="classified", category="billing", priority="high")
    await add(db, "e", NOW + 5)  # pending, no category

    async def ids(**f):
        args = dict(category=None, priority=None, status=None, limit=50, after=None)
        args.update(f)
        return [r.id for r in await db.list_tickets(**args)]

    assert await ids() == ["e", "d", "c", "b", "a"]
    assert await ids(category="billing") == ["d", "b", "a"]
    assert await ids(priority="high") == ["d", "c", "a"]
    assert await ids(status="classified") == ["d", "b", "a"]
    assert await ids(status="pending") == ["e"]
    assert await ids(category="billing", priority="high") == ["d", "a"]
    assert await ids(category="billing", priority="high", status="classified") == [
        "d",
        "a",
    ]
    assert await ids(category="technical", status="classified") == []
    assert await ids(limit=2) == ["e", "d"]


async def test_list_after_cursor_no_repeats_when_new_rows_arrive(db):
    for i in range(5):
        await add(db, f"t{i}", NOW + i)

    kw = dict(category=None, priority=None, status=None)
    page1 = await db.list_tickets(**kw, limit=2, after=None)
    assert [r.id for r in page1] == ["t4", "t3"]

    await add(db, "newest", NOW + 100)

    last = page1[-1]
    page2 = await db.list_tickets(**kw, limit=2, after=(last.created_at, last.id))
    assert [r.id for r in page2] == ["t2", "t1"]

    last = page2[-1]
    page3 = await db.list_tickets(**kw, limit=2, after=(last.created_at, last.id))
    assert [r.id for r in page3] == ["t0"]


async def test_same_created_at_tiebreak_by_id(db):
    for id_ in ("a", "c", "b", "d"):
        await add(db, id_, NOW)

    kw = dict(category=None, priority=None, status=None)
    page1 = await db.list_tickets(**kw, limit=2, after=None)
    assert [r.id for r in page1] == ["d", "c"]
    page2 = await db.list_tickets(**kw, limit=2, after=(NOW, "c"))
    assert [r.id for r in page2] == ["b", "a"]
