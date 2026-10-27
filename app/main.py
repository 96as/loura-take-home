from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import register_error_handlers, router
from app.config import Settings
from app.db import Database
from app.llm.base import LLMClient
from app.llm.fake import FakeLLM
from app.worker import WorkerContext, WorkerPool


def create_app(
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    clock: Callable[[], float] = time.time,
) -> FastAPI:
    settings = settings or Settings.load()
    llm = llm or FakeLLM(settings.fake_llm_failure_rate, settings.fake_llm_seed)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = await Database.connect(settings.db_path)
        pool: WorkerPool | None = None
        try:
            await db.create_schema()
            ctx = WorkerContext(
                db=db,
                llm=llm,
                settings=settings,
                clock=clock,
                rng=random.Random(settings.fake_llm_seed),
            )
            app.state.db = db
            app.state.ctx = ctx
            app.state.clock = clock
            if settings.worker_count > 0:
                pool = WorkerPool(ctx)
                await pool.start()
            yield
        finally:
            try:
                if pool is not None:
                    await pool.stop()
            finally:
                await db.close()

    app = FastAPI(title="Ticket classifier", lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(router)
    return app


# Plain `uvicorn app.main:app` configures no handlers for our loggers, so the
# worker's INFO lines (retries, failures, lost leases) would otherwise vanish.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
app = create_app()
