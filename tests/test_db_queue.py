import asyncio
from dataclasses import replace

from app.db import Database, InsertResult
from app.schemas import Classification, TicketIn

LEASE = 30.0
MAX = 3


async def add(db, id_, now, **kw):
    result, _ = await db.insert_ticket(
        TicketIn(id=id_, subject="Subj", body="Body"),
        suspected_injection=False,
        now=now,
    )
    assert result is InsertResult.CREATED
    if kw:
        sets = ", ".join(f"{k} = ?" for k in kw)
        await db._conn.execute(
            f"UPDATE tickets SET {sets} WHERE id = ?", (*kw.values(), id_)
        )
    return await db.get_ticket(id_)


async def claim(db, clock, token):
    return await db.claim_next(
        token=token, now=clock(), lease_s=LEASE, max_attempts=MAX
    )


def labels() -> Classification:
    return Classification(category="billing", priority="high", summary="Refund ask.")


async def test_claim_takes_oldest_and_increments_attempts(db, clock):
    await add(db, "newer", clock() + 2)
    await add(db, "older", clock() + 1)
    clock.advance(5)

    row = await claim(db, clock, "tok-1")

    assert row is not None
    assert row.id == "older"
    assert row.attempts == 1
    assert row.held_by == "tok-1"
    assert row.held_until == clock() + LEASE
    assert row.updated_at == clock()
    assert row.status == "pending"
    assert await db.get_ticket("older") == row
    untouched = await db.get_ticket("newer")
    assert untouched.attempts == 0
    assert untouched.held_by is None


async def test_claim_same_created_at_tiebreak_by_id(db, clock):
    await add(db, "b", clock())
    await add(db, "a", clock())
    row = await claim(db, clock, "tok")
    assert row.id == "a"


async def test_claim_empty_queue_returns_none(db, clock):
    assert await claim(db, clock, "tok") is None


async def test_held_ticket_not_claimable_until_lease_expires(db, clock):
    await add(db, "t1", clock())
    first = await claim(db, clock, "tok-A")
    assert first is not None

    assert await claim(db, clock, "tok-B") is None
    clock.advance(LEASE)  # held_until == now: still held (strict <)
    assert await claim(db, clock, "tok-B") is None

    clock.advance(1)  # 31 s after the claim
    second = await claim(db, clock, "tok-B")
    assert second is not None
    assert second.id == "t1"
    assert second.attempts == 2
    assert second.held_by == "tok-B"
    assert second.held_until == clock() + LEASE


async def test_next_attempt_at_respected(db, clock):
    await add(db, "t1", clock())
    row = await claim(db, clock, "tok-A")

    ok = await db.release_for_retry(
        row.id, "tok-A", next_attempt_at=clock() + 2, error="boom", now=clock()
    )
    assert ok is True
    released = await db.get_ticket("t1")
    assert released.status == "pending"
    assert released.held_by is None
    assert released.held_until is None
    assert released.next_attempt_at == clock() + 2
    assert released.last_error == "boom"
    assert released.attempts == 1
    assert released.updated_at == clock()

    assert await claim(db, clock, "tok-B") is None
    clock.advance(1.9)
    assert await claim(db, clock, "tok-B") is None
    clock.advance(0.1)  # next_attempt_at == now is claimable (<=)
    again = await claim(db, clock, "tok-B")
    assert again is not None
    assert again.attempts == 2


async def test_attempts_cap(db, clock):
    await add(db, "capped", clock(), attempts=MAX)
    for _ in range(3):
        assert await claim(db, clock, "tok") is None
        clock.advance(1000)
    row = await db.get_ticket("capped")
    assert row.attempts == MAX
    assert row.held_by is None
    assert row.status == "pending"


async def test_non_pending_never_claimed(db, clock):
    await add(db, "done", clock(), status="classified")
    await add(db, "dead", clock(), status="failed")
    assert await claim(db, clock, "tok") is None


async def test_two_concurrent_claims_never_same_ticket(db, clock):
    await add(db, "t1", clock())
    await add(db, "t2", clock() + 1)

    results = await asyncio.gather(
        *(claim(db, clock, f"tok-{i}") for i in range(5))
    )

    got = [r for r in results if r is not None]
    assert len(got) == 2
    assert {r.id for r in got} == {"t1", "t2"}
    for r in got:
        stored = await db.get_ticket(r.id)
        assert stored.held_by == r.held_by
        assert stored.attempts == 1


async def test_claims_from_two_connections_never_same_ticket(db, clock, settings):
    n = 20
    for i in range(n):
        await add(db, f"t{i:02d}", clock() + i)
    clock.advance(n)
    other = await Database.connect(settings.db_path)
    try:
        conns = (db, other)
        results = await asyncio.gather(
            *(claim(conns[i % 2], clock, f"tok-{i}") for i in range(2 * n))
        )
    finally:
        await other.close()

    got = [r for r in results if r is not None]
    assert len(got) == n
    assert len({r.id for r in got}) == n
    assert len({r.held_by for r in got}) == n
    for r in got:
        stored = await db.get_ticket(r.id)
        assert stored.held_by == r.held_by
        assert stored.attempts == 1


async def test_stale_token_writes_change_nothing(db, clock):
    await add(db, "t1", clock())
    assert (await claim(db, clock, "tok-A")) is not None
    clock.advance(LEASE + 1)
    assert (await claim(db, clock, "tok-B")) is not None
    clock.advance(1)
    before = await db.get_ticket("t1")
    assert before.held_by == "tok-B"

    assert await db.save_classified(
        "t1", "tok-A", labels(), prompt_version=1, now=clock()
    ) is False
    assert await db.mark_failed("t1", "tok-A", error="x", now=clock()) is False
    assert await db.release_for_retry(
        "t1", "tok-A", next_attempt_at=clock() + 2, error="x", now=clock()
    ) is False
    assert await db.renew("t1", "tok-A", now=clock(), lease_s=LEASE) is False
    assert await db.release_on_shutdown("t1", "tok-A", now=clock()) is False

    assert await db.get_ticket("t1") == before


async def test_fenced_writes_on_missing_ticket_return_false(db, clock):
    assert await db.renew("nope", "tok", now=clock(), lease_s=LEASE) is False
    assert await db.save_classified(
        "nope", "tok", labels(), prompt_version=1, now=clock()
    ) is False


async def test_renew_extends_lease(db, clock):
    await add(db, "t1", clock())
    await claim(db, clock, "tok-A")
    clock.advance(20)

    assert await db.renew("t1", "tok-A", now=clock(), lease_s=LEASE) is True
    row = await db.get_ticket("t1")
    assert row.held_until == clock() + LEASE
    assert row.updated_at == clock()
    assert row.held_by == "tok-A"
    assert row.attempts == 1

    clock.advance(LEASE - 1)  # past the original lease, inside the renewed one
    assert await claim(db, clock, "tok-B") is None


async def test_save_classified_clears_lease_and_error(db, clock):
    await add(db, "t1", clock())
    await claim(db, clock, "tok-A")
    await db.release_for_retry(
        "t1", "tok-A", next_attempt_at=clock() + 2, error="timeout", now=clock()
    )
    clock.advance(2)
    await claim(db, clock, "tok-B")
    clock.advance(3)

    ok = await db.save_classified(
        "t1", "tok-B", labels(), prompt_version=1, now=clock()
    )

    assert ok is True
    row = await db.get_ticket("t1")
    assert row.status == "classified"
    assert row.category == "billing"
    assert row.priority == "high"
    assert row.summary == "Refund ask."
    assert row.prompt_version == 1
    assert row.held_by is None
    assert row.held_until is None
    assert row.next_attempt_at is None
    assert row.last_error is None
    assert row.attempts == 2
    assert row.updated_at == clock()
    assert await claim(db, clock, "tok-C") is None


async def test_mark_failed_nulls_labels(db, clock):
    await add(db, "t1", clock(), category="other", priority="low",
              summary="stale", prompt_version=1, next_attempt_at=clock() - 1)
    await claim(db, clock, "tok-A")
    clock.advance(1)

    ok = await db.mark_failed("t1", "tok-A", error="bad reply", now=clock())

    assert ok is True
    row = await db.get_ticket("t1")
    assert row.status == "failed"
    assert row.category is None
    assert row.priority is None
    assert row.summary is None
    assert row.prompt_version is None
    assert row.last_error == "bad reply"
    assert row.held_by is None
    assert row.held_until is None
    assert row.next_attempt_at is None
    assert row.updated_at == clock()
    clock.advance(1000)
    assert await claim(db, clock, "tok-B") is None


async def test_sweep_fails_dead_final_attempt(db, clock):
    await add(db, "dead", clock())
    await add(db, "live", clock() + 1)
    await add(db, "early", clock() + 2)  # dead holder but attempts left
    for _ in range(MAX):
        row = await claim(db, clock, "tok")  # holder dies without writing
        assert row.id == "dead"  # oldest, so re-claimed each time it expires
        clock.advance(LEASE + 1)
    # "dead" now has attempts == MAX and an expired lease. Give "live" a fresh
    # final-attempt lease and "early" an expired lease with attempts left.
    await db._conn.execute(
        "UPDATE tickets SET attempts = ?, held_by = 'live-tok', held_until = ? "
        "WHERE id = 'live'",
        (MAX, clock() + LEASE),
    )
    await db._conn.execute(
        "UPDATE tickets SET attempts = 1, held_by = 'gone', held_until = ? "
        "WHERE id = 'early'",
        (clock() - 1,),
    )
    dead_before = await db.get_ticket("dead")
    assert dead_before.attempts == MAX
    assert dead_before.held_until < clock()
    live_before = await db.get_ticket("live")
    early_before = await db.get_ticket("early")
    clock.advance(1)

    count = await db.sweep_exhausted(now=clock(), max_attempts=MAX)

    assert count == 1
    dead = await db.get_ticket("dead")
    assert dead.status == "failed"
    assert dead.last_error == "worker stopped during final attempt"
    assert dead.held_by is None
    assert dead.held_until is None
    assert dead.attempts == MAX
    assert dead.updated_at == clock()
    assert await db.get_ticket("live") == live_before
    assert await db.get_ticket("early") == early_before
    assert await db.sweep_exhausted(now=clock(), max_attempts=MAX) == 0


async def test_sweep_fails_unheld_capped_and_ignores_finished(db, clock):
    # Released for retry at the cap, then MAX_ATTEMPTS lowered by a restart:
    # pending, unheld, capped rows can never be claimed, so the sweep fails them.
    await add(db, "unheld", clock(), attempts=MAX)
    await add(db, "done", clock(), status="classified", attempts=MAX,
              held_until=clock() - 100)
    await add(db, "failed", clock(), status="failed", attempts=MAX)
    before = {i: await db.get_ticket(i) for i in ("done", "failed")}
    clock.advance(1)
    assert await db.sweep_exhausted(now=clock(), max_attempts=MAX) == 1
    unheld = await db.get_ticket("unheld")
    assert unheld.status == "failed"
    assert unheld.last_error == "attempt limit reached"
    assert unheld.held_by is None
    assert unheld.held_until is None
    assert unheld.attempts == MAX
    assert unheld.updated_at == clock()
    assert {i: await db.get_ticket(i) for i in before} == before
    assert await db.sweep_exhausted(now=clock(), max_attempts=MAX) == 0


async def test_release_on_shutdown_gives_attempt_back(db, clock):
    await add(db, "t1", clock())
    row = await claim(db, clock, "tok-A")
    assert row.attempts == 1
    clock.advance(1)

    ok = await db.release_on_shutdown("t1", "tok-A", now=clock())

    assert ok is True
    released = await db.get_ticket("t1")
    assert released == replace(
        row, attempts=0, held_by=None, held_until=None, updated_at=clock()
    )
    again = await claim(db, clock, "tok-B")
    assert again is not None
    assert again.attempts == 1
