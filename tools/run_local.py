"""Run the harness on a private copy of a task set, then grade it.

The harness edits task directories, so it always works on a copy under ../runs.
Credentials come from ../.env (OPENAI_BASE_URL and OPENAI_API_KEY).

Usage: python tools/run_local.py TASKS_DIR [--name NAME] [--budget N] [--docker]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SOLUTION = Path(__file__).resolve().parents[1]
PROJECT = SOLUTION.parent
IMAGE = "hack:latest"


def read_env_file() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (PROJECT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"')
    return values


def docker_command(tasks: Path, budget: int, name: str) -> list[str]:
    """The command from the rules, with BUDGET and a trajectory path added."""
    return [
        "docker", "run", "--rm", "--name", f"run-{name}",
        "-e", "LLM_BASE_URL", "-e", "LLM_API_KEY",
        "-e", f"BUDGET={budget}",
        "-e", "TRAJECTORY_PATH=/data/tasks/_trajectory.jsonl",
        "-v", f"{tasks}:/data/tasks:rw",
        "--read-only", "--tmpfs", "/tmp:rw,size=2g", "-e", "HOME=/tmp",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--cpus", "8", "--memory", "16g", "--pids-limit", "512",
        "--workdir", "/app", "--entrypoint", "python",
        IMAGE, "/app/run_agent.py", "--tasks-dir", "/data/tasks",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks_dir")
    parser.add_argument("--name", default="dev")
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--docker", action="store_true")
    args = parser.parse_args()

    run_dir = PROJECT / "runs" / args.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    tasks = run_dir / "tasks"
    shutil.copytree(Path(args.tasks_dir), tasks, ignore=shutil.ignore_patterns("answer.txt", ".DS_Store"))

    credentials = read_env_file()
    env = {
        **os.environ,
        "LLM_BASE_URL": credentials["OPENAI_BASE_URL"],
        "LLM_API_KEY": credentials["OPENAI_API_KEY"],
    }
    trajectory = tasks / "_trajectory.jsonl"
    if args.docker:
        command = docker_command(tasks, args.budget, args.name)
        subprocess.run(command, env=env, check=False)
    else:
        env["BUDGET"] = str(args.budget)
        env["TRAJECTORY_PATH"] = str(trajectory)
        command = [sys.executable, str(SOLUTION / "run_agent.py"), "--tasks-dir", str(tasks)]
        subprocess.run(command, env=env, cwd=run_dir, check=False)

    grade = [sys.executable, str(SOLUTION / "tools" / "score.py"), str(tasks), "--trajectory", str(trajectory)]
    subprocess.run(grade, check=False)


if __name__ == "__main__":
    main()
