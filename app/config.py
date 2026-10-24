from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_path: str = "tickets.db"
    worker_count: int = 4
    poll_interval_s: float = 0.5
    lease_s: float = 30
    heartbeat_s: float = 10
    llm_timeout_s: float = 20
    max_attempts: int = 3
    fake_llm_failure_rate: float = 0.25
    fake_llm_seed: int | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        d = cls()
        seed = env.get("FAKE_LLM_SEED")
        return cls(
            db_path=env.get("DB_PATH", d.db_path),
            worker_count=int(env.get("WORKER_COUNT", d.worker_count)),
            poll_interval_s=float(env.get("POLL_INTERVAL_S", d.poll_interval_s)),
            lease_s=float(env.get("LEASE_S", d.lease_s)),
            heartbeat_s=float(env.get("HEARTBEAT_S", d.heartbeat_s)),
            llm_timeout_s=float(env.get("LLM_TIMEOUT_S", d.llm_timeout_s)),
            max_attempts=int(env.get("MAX_ATTEMPTS", d.max_attempts)),
            fake_llm_failure_rate=float(
                env.get("FAKE_LLM_FAILURE_RATE", d.fake_llm_failure_rate)
            ),
            fake_llm_seed=int(seed) if seed else None,
        )
