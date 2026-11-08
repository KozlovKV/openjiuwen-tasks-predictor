"""Grade a finished run. The only part of the project that reads `expected`.

Usage: python tools/score.py TASKS_DIR [--trajectory PATH]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


def is_correct(root: Path, spec: dict, answer: str | None) -> bool:
    if answer is None:
        return False
    check = spec.get("check")
    if check == "unittest":
        workspace = root / "workspace"
        script = next(iter(sorted(workspace.glob("test_*.py"))), None)
        if script is None or answer != "DONE":
            return False
        result = subprocess.run(
            [sys.executable, script.name], cwd=workspace, capture_output=True, timeout=60
        )
        return result.returncode == 0
    if check == "exact_int":
        try:
            return int(answer) == int(spec["expected"])
        except (TypeError, ValueError):
            return False
    return answer == str(spec.get("expected"))


def trajectory_totals(path: Path | None) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    if not path or not path.is_file():
        return totals
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event") == "llm":
            totals["calls"] += 1
            totals["cost"] += record["cost"]
            totals[f"cost:{record['task_type']}"] += record["cost"]
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks_dir")
    parser.add_argument("--trajectory")
    args = parser.parse_args()

    root = Path(args.tasks_dir)
    earned = maximum = 0.0
    per_type: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    wrong: list[str] = []
    missed: list[str] = []
    infeasible_answered: Counter[str] = Counter()
    infeasible_skipped: Counter[str] = Counter()

    for spec_path in sorted(root.glob("*/task.json")):
        task_root = spec_path.parent
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        answer_path = task_root / "answer.txt"
        answer = answer_path.read_text(encoding="utf-8").strip() if answer_path.is_file() else None
        reward = float(spec["reward"])
        task_type = spec["task_type"]
        if spec.get("feasible", True) is False:
            (infeasible_answered if answer is not None else infeasible_skipped)[spec.get("kind", "?")] += 1
            continue
        maximum += reward
        per_type[task_type][1] += reward
        if is_correct(task_root, spec, answer):
            earned += reward
            per_type[task_type][0] += reward
        elif answer is None:
            missed.append(f"{spec['task_id']} (no answer)")
        else:
            wrong.append(f"{spec['task_id']}: got {answer!r}, want {spec['expected']!r}")

    print(f"score: {earned:.0f} / {maximum:.0f} ({100 * earned / maximum if maximum else 0:.1f}%)")
    for task_type, (got, possible) in sorted(per_type.items()):
        print(f"  {task_type:15s} {got:.0f} / {possible:.0f}")
    print(f"unsolvable skipped:  {sum(infeasible_skipped.values())} {dict(infeasible_skipped)}")
    print(f"unsolvable answered: {sum(infeasible_answered.values())} {dict(infeasible_answered)}")
    print(f"solvable, no answer: {len(missed)}")
    for item in missed[:30]:
        print("   ", item)
    print(f"solvable, wrong:     {len(wrong)}")
    for item in wrong[:30]:
        print("   ", item)
    totals = trajectory_totals(Path(args.trajectory) if args.trajectory else None)
    if totals:
        print(f"llm calls: {totals['calls']:.0f}, cost: {totals['cost']:.0f} output-token equivalents")
        for key, value in sorted(totals.items()):
            if key.startswith("cost:"):
                print(f"  {key[5:]:15s} {value:.0f}")


if __name__ == "__main__":
    main()
