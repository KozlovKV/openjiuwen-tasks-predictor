"""Budget accounting in output-token equivalents."""
from __future__ import annotations

from .config import INPUT_TOKEN_WEIGHT


class BudgetExhausted(Exception):
    """The proxy stopped answering, so the whole run has to end."""


class CallUnaffordable(Exception):
    """The budget cannot cover this call, though a cheaper one may still fit."""


class TaskCapExceeded(Exception):
    """This task has used up its own share of the budget."""


def call_cost(input_tokens: int, output_tokens: int) -> float:
    return output_tokens + input_tokens * INPUT_TOKEN_WEIGHT


class Ledger:
    """Tracks spending and reserves the worst case before every call.

    Reservations keep parallel workers from collectively overrunning the
    budget. The harness runs on one event loop, so no locking is needed.
    """

    def __init__(self, budget: float, safety: float | None = None) -> None:
        self.budget = float(budget)
        self.safety = max(10.0, 0.01 * budget) if safety is None else safety
        self.spent = 0.0
        self.reserved = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    @property
    def available(self) -> float:
        return self.budget - self.safety - self.spent - self.reserved

    def reserve(self, amount: float) -> float:
        if amount > self.available:
            raise CallUnaffordable(f"need {amount:.0f}, have {self.available:.0f}")
        self.reserved += amount
        return amount

    def release(self, reservation: float) -> None:
        self.reserved = max(0.0, self.reserved - reservation)

    def settle(self, reservation: float, input_tokens: int, output_tokens: int) -> float:
        self.release(reservation)
        cost = call_cost(input_tokens, output_tokens)
        self.spent += cost
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1
        return cost


class TaskMeter:
    """Per-task spending limit, so that one task cannot drain the budget."""

    def __init__(self, cap: float) -> None:
        self.cap = cap
        self.spent = 0.0

    def check(self, worst_case: float) -> None:
        if self.spent + worst_case > self.cap:
            raise TaskCapExceeded(f"{self.spent:.0f}+{worst_case:.0f} > {self.cap:.0f}")

    def add(self, cost: float) -> None:
        self.spent += cost
