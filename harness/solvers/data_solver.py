"""Questions over data files: the model sees the schema, not the rows."""
from __future__ import annotations

import re
from pathlib import Path

from .. import sandbox
from .common import (
    SolveContext,
    SolveResult,
    Status,
    answer_from_output,
    extract_code,
    retry_note,
    says_impossible,
)

SYSTEM = (
    "Reply with only a Python 3 program (standard library) that reads the file from "
    "the current directory and prints the final answer and nothing else. "
    "If the columns cannot answer the question, reply only IMPOSSIBLE."
)
_DATA_FILE_RE = re.compile(r"[\w./-]+\.(?:csv|tsv|json|jsonl)\b")
_DATA_SUFFIXES = (".csv", ".tsv", ".json", ".jsonl")
_HEAD_LINES = 3
_HEAD_LINE_CHARS = 200


def data_files(question: str, workdir: Path) -> list[Path]:
    named = [workdir / name for name in _DATA_FILE_RE.findall(question)]
    existing = [path for path in named if path.is_file()]
    if existing:
        return existing
    return [path for path in sorted(workdir.iterdir()) if path.suffix in _DATA_SUFFIXES]


def _head(path: Path) -> str:
    with path.open(encoding="utf-8", errors="replace") as handle:
        lines = [handle.readline().rstrip("\r\n")[:_HEAD_LINE_CHARS] for _ in range(_HEAD_LINES)]
    return "\n".join(line for line in lines if line)


async def solve(ctx: SolveContext) -> SolveResult:
    question = ctx.task.question
    workdir = ctx.task.workdir
    previews = "\n\n".join(
        f"{path.relative_to(workdir).as_posix()}, first lines:\n{_head(path)}"
        for path in data_files(question, workdir)
    )
    user = f"{question}\n\n{previews}{retry_note(ctx.attempt)}"
    reply = await ctx.ask(SYSTEM, user, "data")
    if says_impossible(reply):
        return SolveResult(Status.IMPOSSIBLE)
    result = await sandbox.run_python(extract_code(reply), workdir)
    return answer_from_output(result.stdout, result.stderr, result.ok)
