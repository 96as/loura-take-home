from pathlib import Path

import pytest

from app.config import Settings

SETTING_VARS = (
    "DB_PATH", "WORKER_COUNT", "POLL_INTERVAL_S", "LEASE_S", "HEARTBEAT_S",
    "LLM_TIMEOUT_S", "MAX_ATTEMPTS", "FAKE_LLM_FAILURE_RATE", "FAKE_LLM_SEED",
)


@pytest.fixture
def clean_env(monkeypatch):
    # Keep the developer's own shell settings out of these tests.
    for name in SETTING_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_match_spec():
    s = Settings.from_env({})
    assert (s.worker_count, s.poll_interval_s, s.lease_s, s.heartbeat_s) == (4, 0.5, 30, 10)
    assert (s.llm_timeout_s, s.max_attempts, s.fake_llm_failure_rate) == (20, 3, 0.25)
    assert s.fake_llm_seed is None and s.db_path == "tickets.db"


def test_env_overrides():
    s = Settings.from_env({"WORKER_COUNT": "2", "FAKE_LLM_SEED": "7", "DB_PATH": "x.db"})
    assert (s.worker_count, s.fake_llm_seed, s.db_path) == (2, 7, "x.db")


def test_load_reads_dotenv_file(tmp_path, clean_env):
    dotenv = tmp_path / ".env"
    dotenv.write_text("WORKER_COUNT=2\nFAKE_LLM_SEED=9\n")
    s = Settings.load(dotenv)
    assert (s.worker_count, s.fake_llm_seed) == (2, 9)
    assert s.max_attempts == 3  # not in the file, so the default stays


def test_real_env_var_beats_dotenv(tmp_path, clean_env):
    dotenv = tmp_path / ".env"
    dotenv.write_text("WORKER_COUNT=2\n")
    clean_env.setenv("WORKER_COUNT", "5")
    assert Settings.load(dotenv).worker_count == 5


def test_missing_dotenv_gives_defaults(tmp_path, clean_env):
    assert Settings.load(tmp_path / "missing.env") == Settings()


def test_env_example_matches_defaults(clean_env):
    example = Path(__file__).resolve().parent.parent / ".env.example"
    assert Settings.load(example) == Settings()
