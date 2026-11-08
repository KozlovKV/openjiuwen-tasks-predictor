"""Running model-written code inside a task directory."""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 20.0
_MAX_CAPTURED_CHARS = 20_000
# Model-written code must never see the proxy key, so the environment is an
# allowlist rather than a copy of ours.
_ALLOWED_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "TEMP", "TMP")


@dataclass(frozen=True)
class ExecResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _environment() -> dict[str, str]:
    env = {key: os.environ[key] for key in _ALLOWED_ENV if key in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _kill(process: asyncio.subprocess.Process) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


async def run_argv(
    argv: list[str], cwd: Path, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> ExecResult:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=_environment(),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except asyncio.TimeoutError:
        _kill(process)
        await process.wait()
        return ExecResult(returncode=-1, stdout="", stderr="timed out", timed_out=True)
    return ExecResult(
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=stdout.decode("utf-8", "replace")[:_MAX_CAPTURED_CHARS],
        stderr=stderr.decode("utf-8", "replace")[:_MAX_CAPTURED_CHARS],
    )


async def run_python(
    code: str, cwd: Path, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> ExecResult:
    return await run_argv([sys.executable, "-c", code], cwd, timeout)


async def run_interpreter(
    arguments: list[str], cwd: Path, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> ExecResult:
    return await run_argv([sys.executable, *arguments], cwd, timeout)


async def run_shell(
    command: str, cwd: Path, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> ExecResult:
    return await run_argv(["bash", "-c", command], cwd, timeout)
