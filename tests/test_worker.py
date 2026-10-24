import asyncio
import json
import random
from dataclasses import replace

from app.db import InsertResult
from app.llm.base import LLMPermanentError
from app.llm.fake import ScriptedLLM
from app.schemas import TicketIn
from app.worker import (
    WorkerContext,
    WorkerPool,
    backoff_seconds,
    heartbeat,
    run_once,
)

VALID = json.dumps(
    {"category": "billing", "priority": "high", "summary": "Customer wants a refund."}
)
INVALID = "sorry, I cannot help with that"
OUT_OF_SET = json.dumps(
    {"category": "urgent", "priority": "high", "summary": "Customer wants a refund."}
)


def make_ctx(db, settings, clock, items):
    llm = ScriptedLLM(items)
    ctx = WorkerContext(
        db=db, llm=llm, settings=settings, clock=clock, rng=random.Random(0)
    )
    return ctx, llm


async def add(db, clock, id_="t1", body="I was charged twice, please refund me."):
    result, _ = await db.insert_ticket(
        TicketIn(id=id_, subject="Refund", body=body),
        suspected_injection=False,
        now=clock(),
    )
    assert result is InsertResult.CREATED


async def steal(db, id_="t1"):
    await db._conn.execute("UPDATE tickets SET held_by = 'other' WHERE id = ?", (id_,))


async def test_success_first_try(db, settings, clock):
    await add(db, clock, body="Unique body text 12345")
    ctx, llm = make_ctx(db, settings, clock, [VALID])

    assert await run_once(ctx) is True

    row = await db.get_ticket("t1")
    assert row.status == "classified"
    assert (row.category, row.priority) == ("billing", "high")
    assert row.summary == "Customer wants a refund."
    assert row.prompt_version == 1
    assert row.last_error is None
    assert row.attempts == 1
    assert row.held_by is None
    assert "Unique body text 12345" in llm.prompts[0]


async def test_invalid_then_valid(db, settings, clock):
    await add(db, clock)
    ctx, llm = make_ctx(db, settings, clock, [INVALID, VALID])

    assert await run_once(ctx) is True
    row = await db.get_ticket("t1")
    assert row.status == "pending"
    assert row.attempts == 1
    assert row.last_error.startswith("invalid reply")
    assert row.held_by is None
    assert clock() + 1.6 <= row.next_attempt_at <= clock() + 2.4

    # Not due yet: the backoff holds it back.
    assert await run_once(ctx) is False

    clock.advance(3)
    assert await run_once(ctx) is True
    row = await db.get_ticket("t1")
    assert row.status == "classified"
    assert row.attempts == 2
    assert row.last_error is None


async def test_three_failures_mark_failed(db, settings, clock):
    await add(db, clock)
    ctx, llm = make_ctx(db, settings, clock, [INVALID, INVALID, INVALID])

    for _ in range(3):
        assert await run_once(ctx) is True
        clock.advance(10)

    row = await db.get_ticket("t1")
    assert row.status == "failed"
    assert (row.category, row.priority, row.summary) == (None, None, None)
    assert row.attempts == 3
    assert row.last_error.startswith("invalid reply")
    assert row.held_by is None

    assert await run_once(ctx) is False
    assert len(llm.prompts) == 3


async def test_out_of_set_never_stored(db, settings, clock):
    await add(db, clock)
    ctx, _ = make_ctx(db, settings, clock, [OUT_OF_SET] * 3)

    for _ in range(3):
        assert await run_once(ctx) is True
        clock.advance(10)

    row = await db.get_ticket("t1")
    assert row.status == "failed"
    assert row.category is None
    assert row.priority is None
    assert row.last_error.startswith("invalid reply")


async def test_permanent_error_fails_immediately(db, settings, clock):
    await add(db, clock)
    ctx, llm = make_ctx(db, settings, clock, [LLMPermanentError("rejected")])

    assert await run_once(ctx) is True

    row = await db.get_ticket("t1")
    assert row.status == "failed"
    assert row.attempts == 1
    assert row.last_error == "llm permanent error: rejected"
    assert row.held_by is None
    assert len(llm.prompts) == 1


async def test_timeout_is_retryable(db, settings, clock):
    await add(db, clock)

    async def slow(prompt):
        await asyncio.sleep(5)
        return VALID

    ctx, _ = make_ctx(db, settings, clock, [slow])

    assert await run_once(ctx) is True

    row = await db.get_ticket("t1")
    assert row.status == "pending"
    assert row.attempts == 1
    assert row.last_error == f"timeout after {settings.llm_timeout_s:g}s"
    assert row.held_by is None
    assert row.next_attempt_at > clock()


async def test_lost_lease_discards_result(db, settings, clock):
    await add(db, clock)
    finished = []

    async def stolen_then_valid(prompt):
        await steal(db)
        # Shorter than llm_timeout_s, far longer than heartbeat_s: the heartbeat
        # notices the loss first and cancels this call.
        await asyncio.sleep(0.1)
        finished.append(True)
        return VALID

    ctx, _ = make_ctx(db, settings, clock, [stolen_then_valid])

    assert await run_once(ctx) is True

    row = await db.get_ticket("t1")
    assert row.status == "pending"
    assert row.category is None
    assert row.held_by == "other"
    assert row.last_error is None
    assert row.attempts == 1
    assert finished == []


async def test_heartbeat_renews(db, settings, clock):
    await add(db, clock)
    ctx, _ = make_ctx(db, settings, clock, [])
    row = await db.claim_next(
        token="tok", now=clock(), lease_s=settings.lease_s, max_attempts=3
    )
    assert row.held_until == clock() + settings.lease_s
    clock.advance(25)

    lost = []
    task = asyncio.create_task(heartbeat(ctx, "t1", "tok", lambda: lost.append(1)))
    try:
        for _ in range(100):
            await asyncio.sleep(settings.heartbeat_s)
            row = await db.get_ticket("t1")
            if row.held_until == clock() + settings.lease_s:
                break
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert row.held_until == clock() + settings.lease_s
    assert row.held_by == "tok"
    assert lost == []


async def test_heartbeat_calls_on_lost(db, settings, clock):
    await add(db, clock)
    ctx, _ = make_ctx(db, settings, clock, [])
    await db.claim_next(
        token="tok", now=clock(), lease_s=settings.lease_s, max_attempts=3
    )
    await steal(db)

    lost = []
    await asyncio.wait_for(heartbeat(ctx, "t1", "tok", lambda: lost.append(1)), 1)

    assert lost == [1]
    assert (await db.get_ticket("t1")).held_by == "other"


def test_backoff_values():
    rng = random.Random(0)
    for _ in range(200):
        assert 1.6 <= backoff_seconds(1, rng) <= 2.4
        assert 6.4 <= backoff_seconds(2, rng) <= 9.6


async def test_pool_stop_releases_in_flight(db, settings, clock):
    await add(db, clock)
    started = asyncio.Event()
    never = asyncio.Event()
    cancelled = []

    async def blocks(prompt):
        started.set()
        try:
            await never.wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return VALID

    ctx, _ = make_ctx(db, replace(settings, worker_count=1), clock, [blocks])
    pool = WorkerPool(ctx)
    await pool.start()
    await asyncio.wait_for(started.wait(), 1)
    assert (await db.get_ticket("t1")).held_by is not None

    await pool.stop()

    row = await db.get_ticket("t1")
    assert row.status == "pending"
    assert row.attempts == 0
    assert row.held_by is None
    assert row.held_until is None
    assert cancelled == [True]
