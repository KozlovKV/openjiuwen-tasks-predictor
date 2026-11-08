"""Bug fixing: the local test suite gives free verification after every edit."""
from __future__ import annotations

from .. import sandbox
from . import pycode
from .common import SolveContext, SolveResult, Status, extract_code, retry_note, says_impossible

SYSTEM = (
    "Fix the bugs in the Python module so that the failing tests pass. Fix the real "
    "logic; never special-case test inputs. Reply with one python code block holding "
    "only the complete new definitions of the functions or classes you change. "
    "If the tests cannot be satisfied, reply only IMPOSSIBLE."
)
DONE = "DONE"
_TEST_TIMEOUT_SECONDS = 30.0
_MAX_FULL_SOURCE_CHARS = 6_000


async def _run_tests(ctx: SolveContext, script: str) -> sandbox.ExecResult:
    return await sandbox.run_script(script, ctx.task.workdir, _TEST_TIMEOUT_SECONDS)


def _module_excerpt(source: str, failures: str) -> str:
    if len(source) <= _MAX_FULL_SOURCE_CHARS:
        return source
    return pycode.definitions_mentioned(source, failures) or source[:_MAX_FULL_SOURCE_CHARS]


async def solve(ctx: SolveContext) -> SolveResult:
    workdir = ctx.task.workdir
    test_path = pycode.find_test_script(ctx.task.instruction, workdir)
    if test_path is None:
        return SolveResult(Status.IMPOSSIBLE)
    target = pycode.find_target_module(ctx.task.instruction, workdir, test_path)
    if target is None:
        return SolveResult(Status.IMPOSSIBLE)
    script = test_path.relative_to(workdir).as_posix()

    before = await _run_tests(ctx, script)
    if before.ok:
        return SolveResult(Status.SOLVED, answer=DONE)
    report = before.stderr + before.stdout
    failures = pycode.summarize_failures(report) or report[-800:]

    source = target.read_text(encoding="utf-8")
    user = (
        f"{target.name}:\n```python\n{_module_excerpt(source, failures)}\n```\n\n"
        f"Failures:\n{failures}{retry_note(ctx.attempt)}"
    )
    reply = await ctx.ask(SYSTEM, user, "coding")
    if says_impossible(reply):
        return SolveResult(Status.IMPOSSIBLE)
    try:
        patched = pycode.apply_definitions(source, extract_code(reply))
    except SyntaxError as exc:
        return SolveResult(Status.RETRY, feedback=f"your code had a syntax error: {exc.msg}")
    if patched == source:
        return SolveResult(Status.RETRY, feedback="your reply changed no definitions")

    target.write_text(patched, encoding="utf-8")
    after = await _run_tests(ctx, script)
    if after.ok:
        return SolveResult(Status.SOLVED, answer=DONE)
    after_report = after.stderr + after.stdout
    if pycode.failure_count(after_report) >= max(1, pycode.failure_count(report)) + 1:
        # The edit made things worse, so the next attempt starts from the old code.
        target.write_text(source, encoding="utf-8")
        return SolveResult(Status.RETRY, feedback="that edit broke more tests")
    return SolveResult(Status.RETRY)
