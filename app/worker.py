"""Background workers: claim a ticket, ask the LLM, write exactly one outcome.

Each attempt holds a lease (``held_by`` token + ``held_until``) that a heartbeat
renews. Every write is fenced on the token, so a worker that lost its lease can
never overwrite whoever holds the ticket now.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from app.classifier import PROMPT_VERSION, InvalidReply, build_prompt, validate_reply
from app.config import Settings
from app.db import Database, TicketRow
from app.llm.base import LLMClient, LLMPermanentError, LLMTransientError

logger = logging.getLogger(__name__)


@dataclass
class WorkerContext:
    db: Database
    llm: LLMClient
    settings: Settings
    clock: Callable[[], float]
    rng: random.Random


def backoff_seconds(attempt: int, rng: random.Random) -> float:
    """Delay before attempt ``attempt + 1``: about 2 s, then about 8 s."""
    return 2 * 4 ** (attempt - 1) * rng.uniform(0.8, 1.2)


async def heartbeat(
    ctx: WorkerContext, ticket_id: str, token: str, on_lost: Callable[[], None]
) -> None:
    """Renew the lease every ``heartbeat_s``; on the first failed renew, call on_lost."""
    while True:
        await asyncio.sleep(ctx.settings.heartbeat_s)
        renewed = await ctx.db.renew(
            ticket_id, token, now=ctx.clock(), lease_s=ctx.settings.lease_s
        )
        if not renewed:
            logger.warning("ticket %s: lease lost, abandoning attempt", ticket_id)
            on_lost()
            return


async def _stop_tasks(*tasks: asyncio.Task) -> None:
    """Cancel every task, then wait until all have really finished, without raising
    their outcomes here. Only a cancellation of the caller itself propagates."""
    for task in tasks:
        task.cancel()
    await asyncio.wait(tasks)
    for task in tasks:
        if not task.cancelled() and task.exception() is not None:
            logger.error("background task failed", exc_info=task.exception())


async def _call_llm(ctx: WorkerContext, row: TicketRow) -> str:
    prompt = build_prompt(row.subject, row.body)
    return await asyncio.wait_for(ctx.llm.complete(prompt), ctx.settings.llm_timeout_s)


async def _retry_or_fail(
    ctx: WorkerContext, row: TicketRow, token: str, error: str
) -> bool:
    now = ctx.clock()
    if row.attempts >= ctx.settings.max_attempts:
        # Final attempt: releasing it would leave a capped, unheld pending row
        # that neither the claim nor the sweep would ever touch again.
        logger.warning("ticket %s: failed after %d attempts: %s", row.id, row.attempts, error)
        return await ctx.db.mark_failed(row.id, token, error=error, now=now)
    delay = backoff_seconds(row.attempts, ctx.rng)
    logger.info(
        "ticket %s: attempt %d failed (%s), retry in %.1fs", row.id, row.attempts, error, delay
    )
    return await ctx.db.release_for_retry(
        row.id, token, next_attempt_at=now + delay, error=error, now=now
    )


async def _process(ctx: WorkerContext, row: TicketRow, token: str) -> None:
    lost = False
    llm_task = asyncio.create_task(_call_llm(ctx, row))

    def on_lost() -> None:
        nonlocal lost
        lost = True
        llm_task.cancel()

    hb_task = asyncio.create_task(heartbeat(ctx, row.id, token, on_lost))
    try:
        try:
            text = await llm_task
        finally:
            # Before any write, so the heartbeat never sees our own save as a lost lease.
            await _stop_tasks(llm_task, hb_task)
    except asyncio.CancelledError:
        me = asyncio.current_task()
        if lost and (me is None or me.cancelling() == 0):
            # Only the LLM call was cancelled, by on_lost. Someone else holds the
            # ticket now; any write would be rejected by the fence anyway.
            logger.warning("ticket %s: lease lost during LLM call, result discarded", row.id)
            return
        # The worker itself is being stopped: give the unfinished attempt back.
        await ctx.db.release_on_shutdown(row.id, token, now=ctx.clock())
        logger.info("ticket %s: released on shutdown", row.id)
        raise
    except TimeoutError:
        error = f"timeout after {ctx.settings.llm_timeout_s:g}s"
        written = await _retry_or_fail(ctx, row, token, error)
    except LLMTransientError as e:
        written = await _retry_or_fail(ctx, row, token, f"llm transient error: {e}")
    except LLMPermanentError as e:
        error = f"llm permanent error: {e}"
        logger.warning("ticket %s: %s", row.id, error)
        written = await ctx.db.mark_failed(row.id, token, error=error, now=ctx.clock())
    else:
        try:
            classification = validate_reply(text)
        except InvalidReply as e:
            written = await _retry_or_fail(ctx, row, token, f"invalid reply: {e}")
        else:
            written = await ctx.db.save_classified(
                row.id, token, classification, prompt_version=PROMPT_VERSION, now=ctx.clock()
            )
            if written:
                logger.info("ticket %s: classified", row.id)
    if not written:
        logger.warning("ticket %s: lease lost before write, outcome discarded", row.id)


async def run_once(ctx: WorkerContext) -> bool:
    """Process at most one ticket. False when nothing was claimable."""
    s = ctx.settings
    swept = await ctx.db.sweep_exhausted(now=ctx.clock(), max_attempts=s.max_attempts)
    if swept:
        logger.warning("marked %d abandoned final-attempt ticket(s) failed", swept)
    token = uuid.uuid4().hex
    row = await ctx.db.claim_next(
        token=token, now=ctx.clock(), lease_s=s.lease_s, max_attempts=s.max_attempts
    )
    if row is None:
        return False
    await _process(ctx, row, token)
    return True


class WorkerPool:
    def __init__(self, ctx: WorkerContext) -> None:
        self._ctx = ctx
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        for i in range(self._ctx.settings.worker_count):
            self._tasks.append(asyncio.create_task(self._loop(), name=f"worker-{i}"))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _loop(self) -> None:
        while True:
            try:
                worked = await run_once(self._ctx)
            except Exception:
                # The ticket's lease will expire and it will be retried (or swept).
                logger.exception("worker: unexpected error, continuing")
                worked = False
            if not worked:
                await asyncio.sleep(self._ctx.settings.poll_interval_s)
