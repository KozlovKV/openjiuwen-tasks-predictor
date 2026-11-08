"""Static helpers for Python sources: patching definitions and reading tests."""
from __future__ import annotations

import ast
import re
from pathlib import Path

_BACKTICK_SCRIPT_RE = re.compile(r"`python3?\s+([\w./-]+\.py)`")
_PY_FILE_RE = re.compile(r"\b[\w./-]+\.py\b")
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_FAILURE_RE = re.compile(r"^(?:FAIL|ERROR): (\S+)", re.MULTILINE)


def find_test_script(instruction: str, workdir: Path) -> Path | None:
    match = _BACKTICK_SCRIPT_RE.search(instruction)
    if match and (workdir / match.group(1)).is_file():
        return workdir / match.group(1)
    candidates = sorted(workdir.glob("test_*.py")) + sorted(workdir.glob("*_test.py"))
    return candidates[0] if candidates else None


def find_target_module(instruction: str, workdir: Path, test_script: Path) -> Path | None:
    for name in _PY_FILE_RE.findall(instruction):
        path = workdir / name
        if path.is_file() and path.resolve() != test_script.resolve():
            return path
    for name in imported_modules(test_script):
        path = workdir / f"{name}.py"
        if path.is_file():
            return path
    return None


def imported_modules(script: Path) -> list[str]:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.append(node.module.split(".")[0])
    return modules


def _start_line(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return decorators[0].lineno if decorators else node.lineno  # type: ignore[attr-defined]


def apply_definitions(source: str, patch: str) -> str:
    """Replace top-level definitions of `source` with those in `patch`.

    New definitions are appended and new imports are added at the top. Anything
    else in the patch is ignored, so a chatty reply cannot break the module.
    Raises SyntaxError if the patch is not valid Python.
    """
    patch_tree = ast.parse(patch)
    patch_lines = patch.splitlines()
    tree = ast.parse(source)
    lines = source.splitlines()
    existing = {node.name: node for node in tree.body if isinstance(node, _DEFINITIONS)}

    replacements: list[tuple[int, int, list[str]]] = []
    appended: list[list[str]] = []
    imports: list[str] = []
    for node in patch_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = "\n".join(patch_lines[node.lineno - 1 : node.end_lineno])
            if text not in source:
                imports.append(text)
        elif isinstance(node, _DEFINITIONS):
            body = patch_lines[_start_line(node) - 1 : node.end_lineno]
            target = existing.get(node.name)
            if target is None:
                appended.append(body)
            else:
                replacements.append((_start_line(target) - 1, target.end_lineno, body))

    for start, end, body in sorted(replacements, key=lambda item: item[0], reverse=True):
        lines[start:end] = body
    for body in appended:
        lines += ["", "", *body]
    result = "\n".join([*imports, *lines]) + "\n"
    ast.parse(result)
    return result


def definitions_mentioned(source: str, text: str) -> str:
    """Imports plus the top-level definitions whose names occur in `text`."""
    tree = ast.parse(source)
    lines = source.splitlines()
    kept: list[str] = []
    for node in tree.body:
        keep = isinstance(node, (ast.Import, ast.ImportFrom))
        if isinstance(node, _DEFINITIONS) and re.search(rf"\b{re.escape(node.name)}\b", text):
            keep = True
        if keep:
            kept.append("\n".join(lines[_start_line(node) - 1 : node.end_lineno]))
    return "\n\n".join(kept)


def failure_count(output: str) -> int:
    return len(_FAILURE_RE.findall(output))


def summarize_failures(output: str, limit: int = 12) -> str:
    """Compress a unittest report to the assertion and error of each failure."""
    blocks = re.split(r"^={40,}\s*$", output, flags=re.MULTILINE)[1:]
    summaries: list[str] = []
    for block in blocks[:limit]:
        header = block.strip().splitlines()[0]
        body = block.split("-" * 40, 1)[-1]
        body = re.sub(r"^-{40,}.*\Z", "", body, flags=re.MULTILINE | re.DOTALL)
        kept = [
            line.strip()
            for line in body.splitlines()
            if line.strip()
            and not line.lstrip().startswith(("Traceback", 'File "', "^", "~"))
        ]
        summaries.append(f"{header}\n  " + "\n  ".join(kept[-5:]))
    return "\n".join(summaries)
