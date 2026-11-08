"""Fallback for everything else: a small ReActAgent with one bash tool."""
from __future__ import annotations

import json
import re
from typing import Any

from openjiuwen.core.foundation.llm import Model, ModelRequestConfig
from openjiuwen.core.foundation.tool import LocalFunction, ToolCard
from openjiuwen.core.runner import Runner
from openjiuwen.core.single_agent import AgentCard
from openjiuwen.core.single_agent.agents.react_agent import ReActAgent, ReActAgentConfig

from .. import sandbox
from . import coding_solver, pycode
from .common import SolveContext, SolveResult, Status, clean_reply, last_line, says_impossible

SYSTEM = (
    "Solve the task in the current directory with the bash tool, in as few steps as "
    "possible. When finished, reply with only the final answer. "
    "If it cannot be done, reply only IMPOSSIBLE."
)
_MAX_ITERATIONS = 4
_MAX_OBSERVATION_CHARS = 1_500
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_]")


class MeteredModel(Model):
    """A Model that charges every call the agent makes to the harness ledger."""

    def __init__(self, ctx: SolveContext) -> None:
        super().__init__(
            ctx.llm.model.model_client_config,
            ModelRequestConfig(
                model=ctx.attempt.model, temperature=0.0, max_tokens=ctx.attempt.max_tokens
            ),
        )
        self._ctx = ctx

    async def invoke(self, messages: Any, **kwargs: Any) -> Any:
        ctx, llm = self._ctx, self._ctx.llm
        chars = len(json.dumps(messages, default=str)) + len(json.dumps(kwargs.get("tools"), default=str))
        worst = llm.worst_case(chars, ctx.attempt.max_tokens)
        ctx.meter.check(worst)
        reservation = llm.ledger.reserve(worst)
        try:
            reply = await llm.guarded(lambda: super(MeteredModel, self).invoke(messages, **kwargs))
        except BaseException:  # includes cancellation by the attempt timeout
            llm.ledger.release(reservation)
            raise
        usage = reply.usage_metadata
        llm.record(
            reservation=reservation,
            input_tokens=usage.input_tokens if usage else int(chars * 0.5),
            output_tokens=usage.output_tokens if usage else ctx.attempt.max_tokens,
            meter=ctx.meter,
            task_id=ctx.task.task_id,
            task_type=ctx.task.task_type,
            label="generic",
            model=ctx.attempt.model,
            question_chars=len(ctx.task.question),
        )
        return reply


def _bash_tool(ctx: SolveContext) -> LocalFunction:
    async def bash(command: str) -> str:
        result = await sandbox.run_shell(command, ctx.task.workdir)
        return (result.stdout + result.stderr)[-_MAX_OBSERVATION_CHARS:] or "(no output)"

    tool_id = "bash_" + _SAFE_ID_RE.sub("_", ctx.task.task_id) + f"_{ctx.attempt.number}"
    card = ToolCard(
        id=tool_id,
        name="bash",
        description="Run a bash command.",
        input_params={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )
    return LocalFunction(card=card, func=bash)


def _build_agent(ctx: SolveContext, tool: LocalFunction) -> ReActAgent:
    agent_id = "agent_" + tool.card.id
    agent = ReActAgent(card=AgentCard(id=agent_id, name=agent_id, description="solver"))
    config = ReActAgentConfig()
    config.configure_prompt_template([{"role": "system", "content": SYSTEM}])
    config.configure_max_iterations(_MAX_ITERATIONS)
    agent.configure(config)
    agent.set_llm(MeteredModel(ctx))
    agent.ability_manager.add(tool.card)
    return agent


async def _verified_coding_result(ctx: SolveContext) -> SolveResult:
    command = pycode.find_test_command(ctx.task.instruction, ctx.task.workdir)
    if command is None:
        return SolveResult(Status.RETRY)
    result = await sandbox.run_interpreter(command, ctx.task.workdir)
    return (
        SolveResult(Status.SOLVED, answer=coding_solver.DONE)
        if result.ok
        else SolveResult(Status.RETRY, feedback=last_line(result.stderr)[:200])
    )


async def solve(ctx: SolveContext) -> SolveResult:
    tool = _bash_tool(ctx)
    Runner.resource_mgr.add_tool(tool)
    try:
        agent = _build_agent(ctx, tool)
        outcome = await agent.invoke(
            {"query": ctx.task.question, "conversation_id": tool.card.id}
        )
    finally:
        Runner.resource_mgr.remove_tool(tool.card.id)

    if ctx.task.task_type == "coding":
        return await _verified_coding_result(ctx)
    text = clean_reply(str(outcome.get("output", ""))) if isinstance(outcome, dict) else ""
    if says_impossible(text):
        return SolveResult(Status.IMPOSSIBLE)
    answer = last_line(text).strip("`*\"'").rstrip(".")
    if not answer:
        return SolveResult(Status.RETRY, feedback="no final answer was given")
    return SolveResult(Status.SOLVED, answer=answer)
