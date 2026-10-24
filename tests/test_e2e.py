import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path

import httpx

from app.llm.fake import FakeLLM
from app.main import create_app

SAMPLES = Path(__file__).resolve().parent.parent / "samples" / "tickets.json"


def load_samples() -> list[dict]:
    return json.loads(SAMPLES.read_text(encoding="utf-8"))


def client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def all_tickets(c) -> list[dict]:
    resp = await c.get("/tickets", params={"limit": 100})
    assert resp.status_code == 200
    return resp.json()["items"]


async def no_pending(c) -> bool:
    return all(t["status"] != "pending" for t in await all_tickets(c))


async def wait_until(check, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not await check():
        assert time.monotonic() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


async def test_samples_end_to_end(settings):
    s = replace(settings, worker_count=4)
    app = create_app(s, llm=FakeLLM(failure_rate=0, seed=1, hang_s=0, latency=(0, 0)))
    samples = load_samples()
    assert [t["id"] for t in samples] == [f"t-{n}" for n in range(1001, 1011)]
    async with app.router.lifespan_context(app), client(app) as c:
        for t in samples:
            assert (await c.post("/tickets", json=t)).status_code == 202
        await wait_until(lambda: no_pending(c))

        tickets = await all_tickets(c)
        assert len(tickets) == len(samples)
        assert all(t["status"] == "classified" for t in tickets)

        t1005 = (await c.get("/tickets/t-1005")).json()
        assert t1005["suspected_injection"] is True
        # The fake model is gullible: validation cannot catch an obeyed injection.
        assert (t1005["category"], t1005["priority"]) == ("technical", "high")

        t1001 = (await c.get("/tickets/t-1001")).json()
        assert t1001["suspected_injection"] is False

        assert (await c.post("/tickets", json=samples[0])).status_code == 200
