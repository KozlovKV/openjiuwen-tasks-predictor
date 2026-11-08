"""The main loop: free checks, then workers pulling from the planner."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Awaitable, Callable

from . import feasibility
from .config import Settings
from .estimator import Estimator
from .ledger import BudgetExhausted, CallUnaffordable, Ledger, TaskCapExceeded, TaskMeter
from .llm import LLM, LLMError, Trajectory
from .planner import GENERIC_MODE, AttemptSpec, Planner, WorkItem
from .solvers import coding_solver, data_solver, generic_solver, math_solver, search_solver
from .solvers.common import Attempt, SolveContext, SolveResult, Status
from .tasks import Task, load_tasks, write_answer

log = logging.getLogger("harness.runner")

Solver = Callable[[SolveContext], Awaitable[SolveResult]]
_SOLVERS: dict[str, Solver] = {
    "math": math_solver.solve,
    "search": search_solver.solve,
    "data_analysis": data_solver.solve,
    "coding": coding_solver.solve,
}
_ATTEMPT_TIMEOUT_SECONDS = 240.0
_MAX_TRANSPORT_RETRIES = 2
_RECONSIDER = "you said IMPOSSIBLE; re-read everything and answer unless the data is truly absent"


def select_solver(task: Task, spec: AttemptSpec) -> Solver:
    if spec.mode == GENERIC_MODE:
        return generic_solver.solve
    return _SOLVERS.get(task.task_type, generic_solver.solve)


class Run:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ledger = Ledger(settings.budget)
        self.trajectory = Trajectory(settings.trajectory_path)
        self.llm = LLM(settings, self.ledger, self.trajectory)
        self.estimator = Estimator()
        self.started = time.monotonic()
        self.stop = False
        self.inflight = 0
        self.wake = asyncio.Event()
        self.meters: dict[str, TaskMeter] = {}
        self.transport_failures: dict[str, int] = {}
        self.answers: dict[str, str] = {}
        self.planner: Planner

    def _past_deadline(self) -> bool:
        return time.monotonic() - self.started > self.settings.deadline_seconds

    def _save_answer(self, task: Task, answer: str) -> None:
        try:
            write_answer(task, answer)
        except OSError:
            # Retrying could not help, so this must not reach the retry logic.
            log.exception("%s: cannot write the answer", task.task_id)
            return
        self.answers[task.task_id] = answer
        out = self.settings.out_path
        if out is None:
            return
        try:
            temporary = out.with_name(out.name + ".tmp")
            temporary.write_text(json.dumps(self.answers), encoding="utf-8")
            os.replace(temporary, out)
        except OSError as exc:
            log.debug("cannot update %s: %s", out, exc)

    def _finish(self, item: WorkItem, status: str) -> None:
        meter = self.meters[item.task.task_id]
        log.info("%s: %s after attempt %d, %.0f tokens, %.0f left",
                 item.task.task_id, status, item.attempt_index + 1, meter.spent, self.ledger.available)
        self.trajectory.write(
            event="outcome",
            task_id=item.task.task_id,
            task_type=item.task.task_type,
            status=status,
            attempts=item.attempt_index + 1,
            spent=meter.spent,
        )

    def _on_transport_error(self, item: WorkItem, exc: Exception) -> None:
        count = self.transport_failures.get(item.task.task_id, 0) + 1
        self.transport_failures[item.task.task_id] = count
        log.warning("%s: transport failure %d: %s", item.task.task_id, count, exc)
        if count <= _MAX_TRANSPORT_RETRIES:
            self.planner.requeue(item)
        else:
            self._finish(item, "transport_failed")

    async def _attempt(self, item: WorkItem) -> None:
        task, spec = item.task, item.spec
        meter = self.meters[task.task_id]
        attempt = Attempt(item.attempt_index + 1, spec.model, spec.max_tokens, item.feedback)
        ctx = SolveContext(task, self.llm, meter, attempt, self.planner.surplus)
        spent_before = meter.spent
        try:
            result = await asyncio.wait_for(
                select_solver(task, spec)(ctx), _ATTEMPT_TIMEOUT_SECONDS
            )
        except TaskCapExceeded:
            self._finish(item, "task_cap")
            return
        except CallUnaffordable:
            # Other calls hold reservations that will come back; wait for them.
            if self.ledger.reserved > 0:
                self.planner.requeue(item)
                self.wake.clear()
                await self.wake.wait()
            else:
                self._finish(item, "unaffordable")
            return
        except (LLMError, asyncio.TimeoutError) as exc:
            self._on_transport_error(item, exc)
            return

        if meter.spent > spent_before:
            self.estimator.observe(task, spec.tier, meter.spent - spent_before)
        if result.status is Status.SOLVED and result.answer is not None:
            self._save_answer(task, result.answer)
            self._finish(item, "solved")
        elif result.status is Status.IMPOSSIBLE:
            # The model sometimes gives up on solvable tasks, and a second look is cheap.
            if item.attempt_index > 0 or not self.planner.retry(item, _RECONSIDER):
                self._finish(item, "model_impossible")
        elif not self.planner.retry(item, result.feedback):
            self._finish(item, "gave_up")

    async def _worker(self) -> None:
        while not self.stop:
            if self._past_deadline():
                self.stop = True
                break
            item = self.planner.next_item()
            if item is None:
                if self.inflight == 0:
                    break
                self.wake.clear()
                await self.wake.wait()
                continue
            self.inflight += 1
            try:
                await self._attempt(item)
            except BudgetExhausted as exc:
                log.warning("stopping: %s", exc)
                self.stop = True
            except Exception:  # one broken task must not end the whole run
                log.exception("%s: solver crashed", item.task.task_id)
                if not self.planner.retry(item, ""):
                    self._finish(item, "crashed")
            finally:
                self.inflight -= 1
                self.wake.set()
        self.wake.set()

    async def execute(self) -> None:
        tasks = load_tasks(self.settings.tasks_dir)
        accepted: list[Task] = []
        for task in tasks:
            verdict = feasibility.check(task)
            if verdict.feasible:
                accepted.append(task)
            else:
                log.info("%s: skipped, %s", task.task_id, verdict.reason)
                self.trajectory.write(
                    event="skip", task_id=task.task_id, task_type=task.task_type, reason=verdict.reason
                )
        self.planner = Planner(accepted, self.ledger, self.estimator)
        self.meters = {task.task_id: TaskMeter(self.planner.task_cap(task)) for task in accepted}
        log.info("%d tasks, %d accepted, budget %d", len(tasks), len(accepted), self.settings.budget)

        workers = [asyncio.create_task(self._worker()) for _ in range(self.settings.workers)]
        grace = self.settings.deadline_seconds + _ATTEMPT_TIMEOUT_SECONDS
        remaining = max(1.0, grace - (time.monotonic() - self.started))
        _, unfinished = await asyncio.wait(workers, timeout=remaining)
        for worker in unfinished:
            worker.cancel()
        log.info(
            "done: %d answered, spent %.0f of %d in %d calls, %.0fs",
            len(self.answers), self.ledger.spent, self.settings.budget, self.ledger.calls,
            time.monotonic() - self.started,
        )


async def run(settings: Settings) -> None:
    await Run(settings).execute()
