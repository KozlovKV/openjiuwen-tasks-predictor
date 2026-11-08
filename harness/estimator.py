"""Predicting what an attempt costs and how likely it is to succeed.

Coefficients come from `tools/fit_estimator.py`, which fits them on collected
trajectories. During a run the prediction for each task type is corrected by
the ratio of observed to predicted cost.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .tasks import Task

COEFFICIENTS_PATH = Path(__file__).with_name("estimator.json")
# Share of tasks that pass the free checks yet still cannot be solved.
P_FEASIBLE_AFTER_CHECKS = 0.9
_RATIO_SMOOTHING = 0.3
_DEFAULT = {"base": 300.0, "per_char": 0.1, "p_solve": 0.5}


@dataclass(frozen=True)
class Prediction:
    cost: float
    p_solve: float


class Estimator:
    def __init__(self, coefficients: dict[str, dict[str, float]] | None = None) -> None:
        if coefficients is None:
            coefficients = json.loads(COEFFICIENTS_PATH.read_text(encoding="utf-8"))
        self._coefficients = coefficients
        self._ratio: dict[str, float] = {}

    def _row(self, task: Task, tier: str) -> dict[str, float]:
        return (
            self._coefficients.get(f"{task.task_type}/{tier}")
            or self._coefficients.get(f"unknown/{tier}")
            or _DEFAULT
        )

    def _raw_cost(self, task: Task, tier: str) -> float:
        row = self._row(task, tier)
        return max(1.0, row["base"] + row["per_char"] * len(task.question))

    def predict(self, task: Task, tier: str) -> Prediction:
        cost = self._raw_cost(task, tier) * self._ratio.get(task.task_type, 1.0)
        return Prediction(cost=cost, p_solve=self._row(task, tier)["p_solve"])

    def observe(self, task: Task, tier: str, actual_cost: float) -> None:
        ratio = actual_cost / self._raw_cost(task, tier)
        previous = self._ratio.get(task.task_type, 1.0)
        self._ratio[task.task_type] = (
            (1 - _RATIO_SMOOTHING) * previous + _RATIO_SMOOTHING * ratio
        )
