from dataclasses import replace

import pytest
import pytest_asyncio

from app.config import Settings
from app.db import Database


class FakeClock:
    """Deterministic clock: call it for the current time, advance() to move it."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return replace(
        Settings(),
        db_path=str(tmp_path / "test.db"),
        worker_count=0,
        poll_interval_s=0.01,
        heartbeat_s=0.01,
        llm_timeout_s=0.2,
    )


@pytest_asyncio.fixture
async def db(settings):
    database = await Database.connect(settings.db_path)
    await database.create_schema()
    yield database
    await database.close()
