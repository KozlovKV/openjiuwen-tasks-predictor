"""Questions over a knowledge base: only the relevant notes reach the model."""
from __future__ import annotations

from . import kb
from .common import SolveContext, SolveResult, Status, retry_note, says_impossible

SYSTEM = (
    "Answer the question using only the notes. Follow references between notes. "
    "Reply with only the answer, no sentence. "
    "If the notes do not contain it, reply only IMPOSSIBLE."
)
# Without a recognisable name the whole base is sent, unless it is unusually big.
_MAX_WHOLE_BASE_CHARS = 12_000
_BASE_LINK_DEPTH = 1


async def solve(ctx: SolveContext) -> SolveResult:
    question = ctx.task.question
    notes = kb.load_notes(ctx.task.workdir)
    # Few notes first: distractors hurt. Each retry looks one link further.
    depth = _BASE_LINK_DEPTH + ctx.attempt.number
    selected = kb.relevant_notes(question, notes, depth) or list(notes.values())
    context = "\n\n".join(note.text for note in selected)[:_MAX_WHOLE_BASE_CHARS]
    user = f"Notes:\n{context}\n\nQuestion: {question}{retry_note(ctx.attempt)}"
    reply = await ctx.ask(SYSTEM, user, "search")
    if says_impossible(reply):
        return SolveResult(Status.IMPOSSIBLE)
    answer = reply.strip().strip("`*\"'").rstrip(".").strip()
    if not answer or "\n" in answer:
        return SolveResult(Status.RETRY, feedback="reply with the answer only, on one line")
    return SolveResult(Status.SOLVED, answer=answer)
