"""Shared types and parsing helpers of the solvers."""
from __future__ import annotations

import enum
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..ledger import TaskMeter
from ..tasks import Task

if TYPE_CHECKING:
    from ..llm import LLM

IMPOSSIBLE = "IMPOSSIBLE"
_FENCE_RE = re.compile(r"```[ \t]*[\w+-]*[ \t]*\r?\n(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class Status(enum.Enum):
    SOLVED = "solved"
    IMPOSSIBLE = "impossible"
    RETRY = "retry"


@dataclass(frozen=True)
class Attempt:
    number: int
    model: str
    max_tokens: int
    feedback: str = ""


@dataclass(frozen=True)
class SolveResult:
    status: Status
    answer: str | None = None
    feedback: str = ""


@dataclass(frozen=True)
class SolveContext:
    task: Task
    llm: "LLM"
    meter: TaskMeter
    attempt: Attempt
    # Tokens not needed by queued work; optional extras may spend only these.
    surplus: Callable[[], float]

    async def ask(
        self,
        system: str,
        user: str,
        label: str,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        reply = await self.llm.complete(
            system=system,
            user=user,
            model=model or self.attempt.model,
            max_tokens=max_tokens or self.attempt.max_tokens,
            meter=self.meter,
            task_id=self.task.task_id,
            task_type=self.task.task_type,
            label=label,
            question_chars=len(self.task.question),
        )
        return clean_reply(reply)


def clean_reply(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def extract_code(text: str) -> str:
    match = _FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


def says_impossible(text: str) -> bool:
    return text.strip().strip("`*. ").upper().startswith(IMPOSSIBLE)


def retry_note(attempt: Attempt) -> str:
    if not attempt.feedback:
        return ""
    return f"\nYour previous try failed: {attempt.feedback}"


def last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_number(text: str) -> str:
    """Render 389.0 as 389; leave everything else untouched."""
    candidate = text.replace(",", "").strip()
    try:
        value = float(candidate)
    except ValueError:
        return text.strip()
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return candidate


def answer_from_output(stdout: str, stderr: str, ok: bool) -> SolveResult:
    """Turn the output of a model-written program into a result."""
    if not ok:
        return SolveResult(Status.RETRY, feedback=last_line(stderr)[:200] or "program failed")
    answer = last_line(stdout)
    if not answer:
        return SolveResult(Status.RETRY, feedback="the program printed nothing")
    if says_impossible(answer):
        return SolveResult(Status.IMPOSSIBLE)
    return SolveResult(Status.SOLVED, answer=normalize_number(answer))
