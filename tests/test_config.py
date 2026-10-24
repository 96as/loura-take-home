from app.config import Settings


def test_defaults_match_spec():
    s = Settings.from_env({})
    assert (s.worker_count, s.poll_interval_s, s.lease_s, s.heartbeat_s) == (4, 0.5, 30, 10)
    assert (s.llm_timeout_s, s.max_attempts, s.fake_llm_failure_rate) == (20, 3, 0.25)
    assert s.fake_llm_seed is None and s.db_path == "tickets.db"


def test_env_overrides():
    s = Settings.from_env({"WORKER_COUNT": "2", "FAKE_LLM_SEED": "7", "DB_PATH": "x.db"})
    assert (s.worker_count, s.fake_llm_seed, s.db_path) == (2, 7, "x.db")
