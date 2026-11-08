"""All model calls go through here, so that every token reaches the ledger."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig

from .config import (
    INPUT_TOKEN_WEIGHT,
    LLM_TIMEOUT_SECONDS,
    MAX_CONSECUTIVE_LLM_FAILURES,
    MODEL_FAST,
    Settings,
)
from .ledger import BudgetExhausted, CallUnaffordable, Ledger, TaskMeter

log = logging.getLogger("harness.llm")

# A rough upper bound of tokens per character, used only for reservations.
_TOKENS_PER_CHAR = 0.5
_MAX_RATE_LIMIT_WAIT_SECONDS = 150.0
_MAX_RATE_LIMIT_PAUSE_SECONDS = 65.0
_DEFAULT_RATE_LIMIT_PAUSE_SECONDS = 2.0
_TRAJECTORY_REPLY_CHARS = 600
_MIN_USEFUL_OUTPUT_TOKENS = 15
_RESETS_AT_RE = re.compile(r"resets at: (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC", re.IGNORECASE)


class LLMError(Exception):
    """A call failed for a reason other than the budget."""


class Trajectory:
    """Append-only log of calls and outcomes, the training data of the estimator."""

    def __init__(self, path: Path | None) -> None:
        self._handle = path.open("a", encoding="utf-8") if path else None

    def write(self, **record: Any) -> None:
        if self._handle:
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part if isinstance(part, str) else str(part.get("text", "")) for part in content)
    return ""


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RateLimitError" in text


def _rate_limit_pause(exc: Exception) -> float:
    match = _RESETS_AT_RE.search(str(exc))
    if not match:
        return _DEFAULT_RATE_LIMIT_PAUSE_SECONDS
    resets = dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    seconds = (resets - dt.datetime.now(dt.timezone.utc)).total_seconds()
    return min(_MAX_RATE_LIMIT_PAUSE_SECONDS, max(_DEFAULT_RATE_LIMIT_PAUSE_SECONDS, seconds))


class LLM:
    def __init__(self, settings: Settings, ledger: Ledger, trajectory: Trajectory) -> None:
        self.ledger = ledger
        self.trajectory = trajectory
        self._consecutive_failures = 0
        # Retries are ours: a retried request may be billed twice by the proxy.
        client_config = ModelClientConfig(
            client_provider="OpenAI",
            api_key=settings.api_key,
            api_base=settings.base_url,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )
        self.model = Model(client_config, ModelRequestConfig(model=MODEL_FAST, temperature=0.0))

    def worst_case(self, input_chars: int, max_tokens: int) -> float:
        return input_chars * _TOKENS_PER_CHAR * INPUT_TOKEN_WEIGHT + max_tokens

    async def guarded(self, invoke: Callable[[], Awaitable[Any]]) -> Any:
        """Run one model call, waiting out rate limits and counting real failures."""
        waited = 0.0
        while True:
            try:
                reply = await invoke()
            except Exception as exc:  # the SDK raises many types for network errors
                if _is_rate_limit(exc) and waited < _MAX_RATE_LIMIT_WAIT_SECONDS:
                    pause = _rate_limit_pause(exc)
                    log.info("rate limited, waiting %.0fs", pause)
                    waited += pause
                    await asyncio.sleep(pause)
                    continue
                self._consecutive_failures += 1
                log.warning("model call failed (%d in a row): %s", self._consecutive_failures, exc)
                if self._consecutive_failures >= MAX_CONSECUTIVE_LLM_FAILURES:
                    # The proxy stops answering once the budget is spent.
                    raise BudgetExhausted("the proxy stopped answering") from exc
                raise LLMError(str(exc)) from exc
            self._consecutive_failures = 0
            return reply

    def record(
        self,
        *,
        reservation: float,
        input_tokens: int,
        output_tokens: int,
        meter: TaskMeter,
        task_id: str,
        task_type: str,
        label: str,
        model: str,
        question_chars: int,
        text: str = "",
    ) -> float:
        cost = self.ledger.settle(reservation, input_tokens, output_tokens)
        meter.add(cost)
        self.trajectory.write(
            event="llm",
            task_id=task_id,
            task_type=task_type,
            label=label,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            question_chars=question_chars,
            reply=text[:_TRAJECTORY_REPLY_CHARS],
        )
        return cost

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        meter: TaskMeter,
        task_id: str,
        task_type: str,
        label: str,
        question_chars: int,
    ) -> str:
        input_chars = len(system) + len(user)
        # On a tight budget, a shorter reply limit keeps the call possible.
        max_tokens = min(max_tokens, int(self.ledger.available - self.worst_case(input_chars, 0)))
        if max_tokens < _MIN_USEFUL_OUTPUT_TOKENS:
            raise CallUnaffordable(f"only {max_tokens} tokens left for the reply")
        worst = self.worst_case(input_chars, max_tokens)
        meter.check(worst)
        reservation = self.ledger.reserve(worst)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            reply = await self.guarded(
                lambda: self.model.invoke(
                    messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    timeout=LLM_TIMEOUT_SECONDS,
                )
            )
        except BaseException:  # includes cancellation by the attempt timeout
            self.ledger.release(reservation)
            raise

        usage = reply.usage_metadata
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        text = _text_of(reply.content)
        if not usage or (input_tokens == 0 and output_tokens == 0):
            # Without usage data, assume the worst rather than undercount.
            input_tokens = int(input_chars * _TOKENS_PER_CHAR)
            output_tokens = max_tokens
        self.record(
            reservation=reservation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            meter=meter,
            task_id=task_id,
            task_type=task_type,
            label=label,
            model=model,
            question_chars=question_chars,
            text=text,
        )
        return text
