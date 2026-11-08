"""Reading a markdown knowledge base and selecting the relevant notes."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")
_FIELD_RE = re.compile(r"^[^:#]+:\s*(.+)$")


@dataclass(frozen=True)
class Note:
    title: str
    text: str


def load_notes(workdir: Path) -> dict[str, Note]:
    notes: dict[str, Note] = {}
    for path in sorted(workdir.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        first = text.splitlines()[0] if text else ""
        title = first.lstrip("# ").strip() if first.startswith("#") else path.stem
        notes[title] = Note(title=title, text=text)
    return notes


def mentioned_entities(question: str) -> list[str]:
    """Capitalised multi-word names, which is how the base names people and firms."""
    return _ENTITY_RE.findall(question)


def seed_titles(question: str, notes: dict[str, Note]) -> list[str]:
    lowered = question.lower()
    return [title for title in notes if title.lower() in lowered]


def _linked_titles(note: Note, notes: dict[str, Note]) -> list[str]:
    linked: list[str] = []
    for line in note.text.splitlines():
        match = _FIELD_RE.match(line.strip())
        if not match:
            continue
        for value in match.group(1).split(","):
            if value.strip() in notes:
                linked.append(value.strip())
    return linked


def relevant_notes(question: str, notes: dict[str, Note], depth: int) -> list[Note]:
    """Notes named in the question plus those within `depth` links of them."""
    frontier = seed_titles(question, notes)
    seen = list(frontier)
    for _ in range(depth):
        following: list[str] = []
        for title in frontier:
            for linked in _linked_titles(notes[title], notes):
                if linked not in seen:
                    seen.append(linked)
                    following.append(linked)
        frontier = following
    return [notes[title] for title in seen]
