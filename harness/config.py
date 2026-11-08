"""Runtime settings and the constants of the competition."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 10 input tokens cost as much as one output token.
INPUT_TOKEN_WEIGHT = 0.1
DEFAULT_BUDGET = 100_000
MAX_WORKERS = 5
# Below the permitted 5: the run takes minutes anyway, and a smaller number stays
# under proxy concurrency limits (the development key allows 3).
DEFAULT_WORKERS = 3
# The run is killed after one hour; stop starting work a little earlier.
DEFAULT_DEADLINE_SECONDS = 55 * 60

MODEL_FAST = "qwen3.8-27b-noreason"
MODEL_FAST_ALT = "qwen3.6-35b-a3b-noreason"
MODEL_REASON = "qwen3.8-27b"

LLM_TIMEOUT_SECONDS = 90.0
# Consecutive proxy failures after which the budget is assumed to be gone.
MAX_CONSECUTIVE_LLM_FAILURES = 5


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    budget: int
    tasks_dir: Path
    deadline_seconds: float
    workers: int
    trajectory_path: Path | None
    out_path: Path | None


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(float(raw)) if raw else default
    except ValueError:
        return default


def load_settings(tasks_dir: str, out: str | None = None) -> Settings:
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not base_url or not api_key:
        raise SystemExit("LLM_BASE_URL and LLM_API_KEY must be set")
    trajectory = os.environ.get("TRAJECTORY_PATH", "").strip()
    out_value = out or os.environ.get("OUT", "").strip()
    return Settings(
        base_url=base_url,
        api_key=api_key,
        budget=_int_env("BUDGET", DEFAULT_BUDGET),
        tasks_dir=Path(tasks_dir),
        deadline_seconds=float(_int_env("DEADLINE_SECONDS", DEFAULT_DEADLINE_SECONDS)),
        workers=max(1, min(MAX_WORKERS, _int_env("WORKERS", DEFAULT_WORKERS))),
        trajectory_path=Path(trajectory) if trajectory else None,
        out_path=Path(out_value) if out_value else None,
    )
