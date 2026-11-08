"""Fit harness/estimator.json on trajectories collected by tools/run_local.py.

For each task type and model tier, cost is modelled as a line in the length of
the question, and the success rate comes from the recorded outcomes. Tiers with
too few samples keep their previous row.

Usage: python tools/fit_estimator.py TRAJECTORY [TRAJECTORY ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TARGET = Path(__file__).resolve().parents[1] / "harness" / "estimator.json"
MIN_SAMPLES = 8
# Predict a little high: running out of budget is worse than a slack estimate.
SAFETY_MARGIN = 1.15


def tier_of(model: str) -> str:
    return "fast" if model.endswith("-noreason") else "reason"


def load_events(paths: list[str]) -> list[dict]:
    events: list[dict] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            events.append({**json.loads(line), "source": path})
    return events


def main() -> None:
    events = load_events(sys.argv[1:])
    # A task can make several calls per attempt, so costs are summed per task.
    # Task ids repeat across runs, hence the source in the key.
    per_task: dict[tuple[str, str, str], list[float]] = {}
    for event in events:
        if event["event"] == "llm":
            key = (f"{event['task_type']}/{tier_of(event['model'])}", event["source"], event["task_id"])
            per_task.setdefault(key, [event["question_chars"], 0.0])[1] += event["cost"]
    samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (key, _, _), (chars, cost) in per_task.items():
        samples[key].append((chars, cost))

    solved: dict[str, int] = defaultdict(int)
    attempted: dict[str, int] = defaultdict(int)
    for event in events:
        if event["event"] != "outcome" or event["status"] == "model_impossible":
            continue
        attempted[event["task_type"]] += 1
        solved[event["task_type"]] += event["status"] == "solved"

    table = json.loads(TARGET.read_text(encoding="utf-8"))
    for key, points in sorted(samples.items()):
        if len(points) < MIN_SAMPLES:
            continue
        x = np.array([point[0] for point in points], dtype=float)
        y = np.array([point[1] for point in points], dtype=float)
        slope, intercept = np.polyfit(x, y, 1) if np.ptp(x) > 0 else (0.0, float(y.mean()))
        slope = max(0.0, float(slope))
        intercept = max(1.0, float(y.mean() - slope * x.mean()))
        task_type = key.split("/")[0]
        rate = solved[task_type] / attempted[task_type] if attempted[task_type] else table[key]["p_solve"]
        table[key] = {
            "base": round(intercept * SAFETY_MARGIN, 2),
            "per_char": round(slope * SAFETY_MARGIN, 4),
            "p_solve": round(min(0.99, rate), 3),
        }
        print(key, table[key], f"n={len(points)}")
    TARGET.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
