"""Arithmetic word problems: the model writes a program, the harness runs it.

Fast models go wrong in a recognisable way: they squeeze the problem into one
expression, then lose a step or get the operator precedence wrong. Programs with
one statement per step were reliable, so that form is enforced for free. When
tokens are to spare, a second model also has to agree.
"""
from __future__ import annotations

import ast
import re

from .. import sandbox
from ..config import MODEL_FAST_ALT, MODEL_REASON
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
    "Reply with only a Python 3 program that prints the final answer and nothing else. "
    "Write one statement for every step of the text, in order, and print only at the end. "
    "Use // where the text says rounded down. "
    "If required data is missing or contradictory, reply only IMPOSSIBLE."
)
_NUMBER_RE = re.compile(r"\d+")
_SENTENCE_END_RE = re.compile(r"[.?!](?:\s|$)")
_VERIFY_MAX_TOKENS = 200
_REFEREE_MAX_TOKENS = 2000
# Verification is optional, so it may spend only tokens no queued task needs.
_MIN_SURPLUS_TO_VERIFY = 200.0
_MIN_SURPLUS_FOR_REFEREE = 2500.0
_FORM_FEEDBACK = "write one separate statement for every step and use every number"


def ignored_numbers(question: str, code: str) -> list[str]:
    """Numbers of the problem that never occur in the program."""
    used = set(_NUMBER_RE.findall(code))
    return sorted(set(_NUMBER_RE.findall(question)) - used, key=question.find)


def is_stepwise(question: str, code: str) -> bool:
    """True if the program has about one statement per sentence of the problem."""
    try:
        statements = len(ast.parse(code).body)
    except SyntaxError:
        return False
    sentences = len([part for part in _SENTENCE_END_RE.split(question) if part.strip()])
    return statements >= sentences - 1


def well_formed(question: str, code: str) -> bool:
    return is_stepwise(question, code) and not ignored_numbers(question, code)


def decide(first: SolveResult, second: SolveResult, referee: SolveResult) -> SolveResult:
    """Majority of three. With no majority, the step-by-step first program is safest."""
    if referee.status is Status.SOLVED and referee.answer == second.answer:
        return referee
    return first


async def _sample(ctx: SolveContext, model: str | None, max_tokens: int | None) -> tuple[SolveResult, str]:
    question = ctx.task.question
    reply = await ctx.ask(
        SYSTEM, question + retry_note(ctx.attempt), "math", model=model, max_tokens=max_tokens
    )
    if says_impossible(reply):
        return SolveResult(Status.IMPOSSIBLE), ""
    code = extract_code(reply)
    result = await sandbox.run_python(code, ctx.task.workdir)
    return answer_from_output(result.stdout, result.stderr, result.ok), code


async def solve(ctx: SolveContext) -> SolveResult:
    question = ctx.task.question
    first, code = await _sample(ctx, None, None)
    if first.status is not Status.SOLVED:
        return first
    # Only on the first try: a problem may legitimately mention an unneeded number.
    if ctx.attempt.number == 1 and not well_formed(question, code):
        return SolveResult(Status.RETRY, feedback=_FORM_FEEDBACK)
    if ctx.surplus() < _MIN_SURPLUS_TO_VERIFY:
        return first

    second, second_code = await _sample(ctx, MODEL_FAST_ALT, _VERIFY_MAX_TOKENS)
    # A broken or one-line program is not a vote: it would only cause false alarms.
    if (
        second.status is not Status.SOLVED
        or second.answer == first.answer
        or not well_formed(question, second_code)
        or ctx.surplus() < _MIN_SURPLUS_FOR_REFEREE
    ):
        return first
    referee, _ = await _sample(ctx, MODEL_REASON, _REFEREE_MAX_TOKENS)
    return decide(first, second, referee)
