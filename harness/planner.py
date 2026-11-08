"""Choosing what to work on next: a knapsack that is re-solved as facts arrive."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import MODEL_FAST, MODEL_REASON
from .estimator import P_FEASIBLE_AFTER_CHECKS, Estimator
from .ledger import Ledger
from .tasks import Task

BUCKET_TOKENS = 25
# Per-task ceiling in output-token equivalents.
_TYPE_CAP = {"math": 2500.0, "search": 2000.0, "data_analysis": 3000.0, "coding": 8000.0}
_DEFAULT_CAP = 5000.0
# How many times its fair per-point share of the budget a task may use.
_CAP_FAIR_SHARE_MULTIPLE = 40.0

GENERIC_MODE = "generic"
FAST_TIER = "fast"
REASON_TIER = "reason"


@dataclass(frozen=True)
class AttemptSpec:
    tier: str
    model: str
    max_tokens: int
    mode: str | None = None


def _fast(max_tokens: int, mode: str | None = None) -> AttemptSpec:
    return AttemptSpec(FAST_TIER, MODEL_FAST, max_tokens, mode)


def _reason(max_tokens: int, mode: str | None = None) -> AttemptSpec:
    return AttemptSpec(REASON_TIER, MODEL_REASON, max_tokens, mode)


_ATTEMPTS: dict[str, list[AttemptSpec]] = {
    "math": [_fast(200), _fast(200), _reason(2000)],
    "search": [_fast(40), _fast(40), _reason(1500)],
    "data_analysis": [_fast(300), _fast(300), _reason(2500)],
    "coding": [_fast(600), _fast(600), _reason(4000), _reason(1500, GENERIC_MODE)],
}
_GENERIC_ATTEMPTS = [_fast(600, GENERIC_MODE), _reason(1500, GENERIC_MODE)]


def attempts_for(task: Task) -> list[AttemptSpec]:
    return _ATTEMPTS.get(task.task_type, _GENERIC_ATTEMPTS)


@dataclass
class WorkItem:
    task: Task
    attempt_index: int = 0
    feedback: str = ""

    @property
    def spec(self) -> AttemptSpec:
        return attempts_for(self.task)[self.attempt_index]


def knapsack(values: Sequence[float], weights: Sequence[int], capacity: int) -> list[int]:
    """Indices of the most valuable subset with total weight within capacity."""
    if capacity <= 0 or not values:
        return []
    if sum(weights) <= capacity:
        return list(range(len(values)))
    best = np.zeros(capacity + 1)
    taken = np.zeros((len(values), capacity + 1), dtype=bool)
    for index, (value, weight) in enumerate(zip(values, weights)):
        if weight > capacity:
            continue
        candidate = best[: capacity + 1 - weight] + value
        improved = candidate > best[weight:]
        best[weight:][improved] = candidate[improved]
        taken[index, weight:] = improved
    chosen: list[int] = []
    remaining = capacity
    for index in range(len(values) - 1, -1, -1):
        if taken[index, remaining]:
            chosen.append(index)
            remaining -= weights[index]
    return sorted(chosen)


class Planner:
    def __init__(self, tasks: list[Task], ledger: Ledger, estimator: Estimator) -> None:
        self._ledger = ledger
        self._estimator = estimator
        self._pending: dict[str, WorkItem] = {t.task_id: WorkItem(t) for t in tasks}
        total_reward = sum(task.reward for task in tasks) or 1.0
        self._per_point = ledger.budget / total_reward

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def surplus(self) -> float:
        """Budget left over after setting aside the predicted cost of queued work."""
        return self._ledger.available - sum(self._cost(item) for item in self._pending.values())

    def task_cap(self, task: Task) -> float:
        by_type = _TYPE_CAP.get(task.task_type, _DEFAULT_CAP)
        return min(by_type, _CAP_FAIR_SHARE_MULTIPLE * self._per_point * max(task.reward, 0.5))

    def _value(self, item: WorkItem) -> float:
        prediction = self._estimator.predict(item.task, item.spec.tier)
        return item.task.reward * P_FEASIBLE_AFTER_CHECKS * prediction.p_solve

    def _cost(self, item: WorkItem) -> float:
        return self._estimator.predict(item.task, item.spec.tier).cost

    def next_item(self) -> WorkItem | None:
        """Pop the densest item of the best affordable set, or None."""
        items = list(self._pending.values())
        if not items:
            return None
        values = [self._value(item) for item in items]
        weights = [max(1, math.ceil(self._cost(item) / BUCKET_TOKENS)) for item in items]
        capacity = int(self._ledger.available // BUCKET_TOKENS)
        chosen = knapsack(values, weights, capacity)
        if not chosen:
            return None
        best = max(chosen, key=lambda index: values[index] / weights[index])
        item = items[best]
        del self._pending[item.task.task_id]
        return item

    def requeue(self, item: WorkItem) -> None:
        """Put an attempt back unchanged after a network failure."""
        self._pending[item.task.task_id] = item

    def retry(self, item: WorkItem, feedback: str) -> bool:
        """Queue the next attempt of a task. False when none are left."""
        next_index = item.attempt_index + 1
        if next_index >= len(attempts_for(item.task)):
            return False
        self._pending[item.task.task_id] = WorkItem(item.task, next_index, feedback)
        return True
