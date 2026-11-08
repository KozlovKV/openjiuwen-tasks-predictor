"""Loading tasks and saving answers."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("harness.tasks")

ANSWER_FILE = "answer.txt"

# Sentences every instruction repeats. The harness does this work itself, so
# sending them to the model would only cost tokens.
_BOILERPLATE = [
    r"Work only inside the current directory\.",
    r"When you are done, write ONLY .*?answer\.txt\)?\.",
    r"Compute it with the bash tool \(python3 -c\)\.",
    r"Use bash with python3 to compute it\.",
    r"Use the file tools or bash \(ls, grep, cat\) to find it\.",
    r"The directory \S+ contains a small knowledge base of markdown files\.",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE), re.DOTALL)


@dataclass(frozen=True)
class Task:
    task_id: str
    task_type: str
    instruction: str
    reward: float
    root: Path

    @property
    def workdir(self) -> Path:
        workspace = self.root / "workspace"
        return workspace if workspace.is_dir() else self.root

    @property
    def question(self) -> str:
        """The instruction without the boilerplate about answer files."""
        return " ".join(_BOILERPLATE_RE.sub(" ", self.instruction).split())


def _reward(value: object) -> float:
    try:
        return max(0.0, float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0


def load_tasks(tasks_dir: Path) -> list[Task]:
    tasks: list[Task] = []
    for child in sorted(tasks_dir.iterdir()):
        spec = child / "task.json"
        if not spec.is_file():
            continue
        try:
            raw = json.loads(spec.read_text(encoding="utf-8"))
            tasks.append(
                Task(
                    task_id=str(raw.get("task_id") or child.name),
                    task_type=str(raw.get("task_type") or "unknown"),
                    instruction=str(raw["instruction"]),
                    reward=_reward(raw.get("reward", 1)),
                    root=child,
                )
            )
        except (OSError, ValueError, KeyError) as exc:
            log.warning("skipping %s: %s", child.name, exc)
    return tasks


def write_answer(task: Task, answer: str) -> None:
    target = task.root / ANSWER_FILE
    temporary = task.root / f".{ANSWER_FILE}.tmp"
    temporary.write_text(answer.strip() + "\n", encoding="utf-8")
    os.replace(temporary, target)
