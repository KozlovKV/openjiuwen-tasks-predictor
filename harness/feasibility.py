"""Token-free checks that catch tasks which cannot be solved.

A wrong answer costs only tokens, while a wrongly skipped task loses its reward,
so a task is rejected only on hard evidence. Everything else gets attempted.
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .solvers import kb, pycode
from .tasks import ANSWER_FILE, Task

_REFERENCED_FILE_RE = re.compile(
    r"(?<![\w/.-])([\w./-]+\.(?:csv|tsv|jsonl?|txt|md|py|xlsx|parquet|sql|db|log))\b"
)
_REFERENCED_DIR_RE = re.compile(r"\b(?:directory|folder)\s+([\w.-]+)/")
_OUTPUT_VERB_RE = re.compile(r"\b(?:create|write|save|generate|produce|output|make)\b", re.IGNORECASE)
_NETWORK_RE = re.compile(
    r"(?:https?://\S+.*\b(?:download|fetch|curl|wget|scrape|retrieve|open)\b)"
    r"|(?:\b(?:download|fetch|curl|wget|scrape|retrieve|open)\b.*https?://)",
    re.IGNORECASE | re.DOTALL,
)
_STATED_COLUMNS_RE = re.compile(r"\bcolumns\s+([\w]+(?:\s*,\s*\w+)+)")


@dataclass(frozen=True)
class Verdict:
    feasible: bool
    reason: str = ""


_OK = Verdict(True)


def _reject(reason: str) -> Verdict:
    return Verdict(False, reason)


def _is_output_file(text: str, position: int) -> bool:
    """True if the sentence asks to produce this file rather than to read it."""
    sentence_start = text.rfind(". ", 0, position) + 1
    return bool(_OUTPUT_VERB_RE.search(text[sentence_start:position]))


def _check_references(task: Task) -> Verdict:
    text = task.instruction
    if _NETWORK_RE.search(text):
        return _reject("needs network access")
    workdir = task.workdir
    for match in _REFERENCED_FILE_RE.finditer(text):
        name = match.group(1)
        if Path(name).name == ANSWER_FILE or _is_output_file(text, match.start()):
            continue
        if not (workdir / name).exists() and not any(workdir.rglob(Path(name).name)):
            return _reject(f"missing file {name}")
    for name in _REFERENCED_DIR_RE.findall(text):
        if not (workdir / name).is_dir():
            return _reject(f"missing directory {name}")
    return _OK


def _check_search(task: Task) -> Verdict:
    notes = kb.load_notes(task.workdir)
    if not notes:
        return _reject("empty knowledge base")
    corpus = "\n".join(note.text for note in notes.values()).lower()
    for entity in kb.mentioned_entities(task.question):
        if entity.lower() not in corpus:
            return _reject(f"unknown entity {entity}")
    return _OK


def _check_data(task: Task) -> Verdict:
    match = _STATED_COLUMNS_RE.search(task.question)
    if not match:
        return _OK
    stated = [column.strip() for column in match.group(1).split(",")]
    for path in sorted(task.workdir.glob("*.csv")):
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            header = next(csv.reader(handle), [])
        absent = [column for column in stated if column not in header]
        if absent:
            return _reject(f"{path.name} lacks columns {absent}")
    return _OK


def _is_literal(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)  # type: ignore[arg-type]
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return True


def _pure_call_of_module(node: ast.AST, imported: set[str]) -> bool:
    """True for calls like `m.f(1, 2)`, whose result depends only on the code."""
    if not isinstance(node, ast.Call) or not node.args:
        return False
    names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    return names <= imported


def _test_contradiction(tree: ast.Module) -> str:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}

    expected: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        method = node.func.attr
        if method == "fail":
            return "a test always fails"
        if method != "assertEqual" or len(node.args) < 2:
            continue
        left, right = node.args[:2]
        if _is_literal(left) and _is_literal(right):
            if ast.literal_eval(left) != ast.literal_eval(right):
                return "a test compares unequal constants"
            continue
        if _pure_call_of_module(left, imported) and _is_literal(right):
            key = ast.unparse(left)
            expected.setdefault(key, set()).add(repr(ast.literal_eval(right)))
            if len(expected[key]) > 1:
                return f"tests expect different results from {key}"
    return ""


def _module_available(name: str, workdir: Path) -> bool:
    if name in sys.stdlib_module_names or (workdir / f"{name}.py").exists():
        return True
    if (workdir / name).is_dir():
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _check_coding(task: Task) -> Verdict:
    workdir = task.workdir
    test_path = pycode.find_test_script(task.instruction, workdir)
    if test_path is None:
        return _reject("no test script")
    try:
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return _reject("the test script does not parse")
    for module in pycode.imported_modules(test_path):
        if not _module_available(module, workdir):
            return _reject(f"tests import missing module {module}")
    if pycode.find_target_module(task.instruction, workdir, test_path) is None:
        return _reject("no module to fix")
    contradiction = _test_contradiction(tree)
    return _reject(contradiction) if contradiction else _OK


_TYPE_CHECKS = {
    "search": _check_search,
    "data_analysis": _check_data,
    "coding": _check_coding,
}


def check(task: Task) -> Verdict:
    """Never raises: an unexpected layout simply means the task gets attempted."""
    try:
        verdict = _check_references(task)
        if verdict.feasible and task.task_type in _TYPE_CHECKS:
            verdict = _TYPE_CHECKS[task.task_type](task)
        return verdict
    except (OSError, ValueError, SyntaxError, RecursionError):
        return _OK
