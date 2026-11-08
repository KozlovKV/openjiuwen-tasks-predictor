"""Bug fixing: the local test suite gives free verification after every edit."""
from __future__ import annotations

from pathlib import Path

from .. import sandbox
from . import pycode
from .common import (
    SolveContext,
    SolveResult,
    Status,
    extract_code_blocks,
    retry_note,
    says_impossible,
)

SYSTEM = (
    "Fix the bugs in the Python code so that the failing tests pass. Fix the real logic; "
    "never special-case test inputs and never mock anything. Reply with one python code "
    "block holding only what you change: whole functions, methods inside their class, or "
    "module-level constants. If the tests cannot be satisfied, reply only IMPOSSIBLE."
)
DONE = "DONE"
_TEST_TIMEOUT_SECONDS = 30.0
_MAX_FULL_SOURCE_CHARS = 6_000


def _prompt(sources: dict[str, str], failures: str, note: str) -> str:
    total = sum(len(text) for text in sources.values())
    shown = sources if total <= _MAX_FULL_SOURCE_CHARS else pycode.excerpt(sources, failures) or sources
    code = "\n\n".join(f"{name}:\n```python\n{text}\n```" for name, text in shown.items())
    broken = [name for name, text in sources.items() if not pycode.parses(text)]
    hint = f"\n{broken[0]} has a syntax error, so reply with that whole file." if broken else ""
    return f"{code}\n\nFailures:\n{failures}{hint}{note}"


def _write(workdir: Path, modules: dict[str, str]) -> None:
    for name, text in modules.items():
        (workdir / name).write_text(text, encoding="utf-8")


async def solve(ctx: SolveContext) -> SolveResult:
    workdir = ctx.task.workdir
    command = pycode.find_test_command(ctx.task.instruction, workdir)
    if command is None:
        return SolveResult(Status.IMPOSSIBLE)
    tests = pycode.test_files(command, workdir)
    modules = pycode.source_modules(ctx.task.instruction, workdir, tests)
    if not modules:
        return SolveResult(Status.IMPOSSIBLE)

    before = await sandbox.run_interpreter(command, workdir, _TEST_TIMEOUT_SECONDS)
    if before.ok:
        return SolveResult(Status.SOLVED, answer=DONE)
    report = before.stderr + before.stdout
    failures = pycode.summarize_failures(report) or report[-800:]

    sources = {
        path.relative_to(workdir).as_posix(): path.read_text(encoding="utf-8") for path in modules
    }
    reply = await ctx.ask(SYSTEM, _prompt(sources, failures, retry_note(ctx.attempt)), "coding")
    if says_impossible(reply):
        return SolveResult(Status.IMPOSSIBLE)
    try:
        patched = pycode.apply_reply(sources, extract_code_blocks(reply))
    except SyntaxError as exc:
        return SolveResult(Status.RETRY, feedback=f"your code had a syntax error: {exc.msg}")
    changed = {name: text for name, text in patched.items() if text != sources[name]}
    if not changed:
        return SolveResult(Status.RETRY, feedback="your reply changed nothing")

    _write(workdir, changed)
    after = await sandbox.run_interpreter(command, workdir, _TEST_TIMEOUT_SECONDS)
    if after.ok:
        return SolveResult(Status.SOLVED, answer=DONE)
    if pycode.made_worse(report, after.stderr + after.stdout):
        # The next attempt starts from the old code rather than from a regression.
        _write(workdir, {name: sources[name] for name in changed})
        return SolveResult(Status.RETRY, feedback="that edit broke more tests")
    return SolveResult(Status.RETRY)
