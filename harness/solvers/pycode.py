"""Static helpers for Python sources: finding tests and applying model edits."""
from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path

_BACKTICK_COMMAND_RE = re.compile(r"`python3?\s+([^`]+)`")
_PY_FILE_RE = re.compile(r"\b[\w./-]+\.py\b")
_FAILURE_RE = re.compile(r"^(?:FAIL|ERROR): (\S+)", re.MULTILINE)
_RAN_TESTS_RE = re.compile(r"^Ran \d+ tests? in", re.MULTILINE)
_TOP_LEVEL_NAME_RE = re.compile(r"^(?:async\s+def|def|class)\s+(\w+)", re.MULTILINE)
_WORD_RE = re.compile(r"\w+")

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEFINITIONS = (*_FUNCTIONS, ast.ClassDef)
_IMPORTS = (ast.Import, ast.ImportFrom)
_SKIPPED_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules"}
_TEST_DIRS = {"test", "tests"}
_MAX_EXCERPT_CONSTANT_CHARS = 400


def _python_files(workdir: Path) -> list[Path]:
    return sorted(
        path
        for path in workdir.rglob("*.py")
        if not set(path.relative_to(workdir).parts[:-1]) & _SKIPPED_DIRS
    )


def _looks_like_test(path: Path, workdir: Path) -> bool:
    directories = set(path.relative_to(workdir).parts[:-1])
    return path.name.startswith("test_") or path.name.endswith("_test.py") or bool(directories & _TEST_DIRS)


def find_test_command(instruction: str, workdir: Path) -> list[str] | None:
    """Interpreter arguments that run the tests of a task, or None if it has none."""
    match = _BACKTICK_COMMAND_RE.search(instruction)
    if match:
        try:
            arguments = shlex.split(match.group(1))
        except ValueError:
            arguments = []
        if arguments and (arguments[0].startswith("-") or (workdir / arguments[0]).is_file()):
            return arguments
    tests = [path for path in _python_files(workdir) if _looks_like_test(path, workdir)]
    if not tests:
        return None
    in_root = [path for path in tests if path.parent == workdir]
    if in_root:
        return [in_root[0].name]
    return ["-m", "unittest", "discover", "-s", tests[0].parent.relative_to(workdir).as_posix()]


def test_files(command: list[str], workdir: Path) -> list[Path]:
    """The files a test command runs; these must never be edited."""
    if command[0].endswith(".py"):
        return [workdir / command[0]]
    return [path for path in _python_files(workdir) if _looks_like_test(path, workdir)]


def imported_modules(script: Path) -> list[str]:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.append(node.module.split(".")[0])
    return modules


def source_modules(instruction: str, workdir: Path, tests: list[Path]) -> list[Path]:
    """Editable modules, likeliest first: named in the task, then imported by tests."""
    excluded = {path.resolve() for path in tests}
    named = {Path(name).name for name in _PY_FILE_RE.findall(instruction)}
    imported: set[str] = set()
    for path in tests:
        try:
            imported |= set(imported_modules(path))
        except SyntaxError:
            continue

    def rank(path: Path) -> tuple[int, str]:
        priority = 0 if path.name in named else 1 if path.stem in imported else 2
        return priority, path.as_posix()

    candidates = [
        path
        for path in _python_files(workdir)
        if path.resolve() not in excluded and not _looks_like_test(path, workdir)
    ]
    return sorted(candidates, key=rank)


def parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def _start_line(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return decorators[0].lineno if decorators else node.lineno  # type: ignore[attr-defined]


def _named(nodes: list[ast.stmt], name: str, kinds: tuple[type, ...]) -> ast.AST | None:
    return next((n for n in nodes if isinstance(n, kinds) and n.name == name), None)  # type: ignore[attr-defined]


def _assignment_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _referenced_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
    return names


def _is_placeholder(node: ast.AST) -> bool:
    """A body of just `...` means "unchanged" and must not overwrite real code."""
    body = node.body  # type: ignore[attr-defined]
    last = body[-1]
    return (
        len(body) <= 2
        and isinstance(last, ast.Expr)
        and isinstance(last.value, ast.Constant)
        and last.value.value is Ellipsis
    )


def _block_of(lines: list[str], node: ast.AST) -> list[str]:
    """Source lines of a node, shifted so that the node starts in column 0."""
    column = node.col_offset  # type: ignore[attr-defined]
    block = lines[_start_line(node) - 1 : node.end_lineno]  # type: ignore[attr-defined]
    return [line[min(column, len(line) - len(line.lstrip())) :] for line in block]


def _indented(block: list[str], column: int) -> list[str]:
    return [" " * column + line if line.strip() else line for line in block]


def _splice(text: str, node: ast.AST, block: list[str]) -> str:
    lines = text.splitlines()
    lines[_start_line(node) - 1 : node.end_lineno] = _indented(block, node.col_offset)  # type: ignore[attr-defined]
    return "\n".join(lines) + "\n"


def _parsed(modules: dict[str, str]) -> list[tuple[str, str, ast.Module]]:
    return [(name, text, ast.parse(text)) for name, text in modules.items() if parses(text)]


def _merge_member(text: str, class_name: str, member: ast.stmt, patch_lines: list[str]) -> str:
    owner = _named(ast.parse(text).body, class_name, (ast.ClassDef,))
    assert isinstance(owner, ast.ClassDef)
    if isinstance(member, _FUNCTIONS):
        if _is_placeholder(member):
            return text
        existing = _named(owner.body, member.name, _FUNCTIONS)
    elif _assignment_name(member):
        name = _assignment_name(member)
        existing = next((node for node in owner.body if _assignment_name(node) == name), None)
    else:
        return text
    block = _block_of(patch_lines, member)
    if existing is not None:
        return _splice(text, existing, block)
    lines = text.splitlines()
    assert owner.end_lineno is not None
    lines[owner.end_lineno : owner.end_lineno] = ["", *_indented(block, owner.body[0].col_offset)]
    return "\n".join(lines) + "\n"


def _merge_class(modules: dict[str, str], node: ast.ClassDef, patch_lines: list[str]) -> str | None:
    """Update only the members a reply mentions, so a partial class loses nothing."""
    for name, text, tree in _parsed(modules):
        if _named(tree.body, node.name, (ast.ClassDef,)) is not None:
            for member in node.body:
                text = _merge_member(text, node.name, member, patch_lines)
            modules[name] = text
            return name
    return None


def _replace_function(modules: dict[str, str], node: ast.AST, patch_lines: list[str]) -> str | None:
    block = _block_of(patch_lines, node)
    parsed = _parsed(modules)
    for name, text, tree in parsed:
        existing = _named(tree.body, node.name, _FUNCTIONS)  # type: ignore[attr-defined]
        if existing is not None:
            modules[name] = _splice(text, existing, block)
            return name
    arguments = [*node.args.posonlyargs, *node.args.args]  # type: ignore[attr-defined]
    if not arguments or arguments[0].arg not in ("self", "cls"):
        return None
    # A method sent without its class: accept it when the owner is unambiguous.
    owners = [
        (name, text, method)
        for name, text, tree in parsed
        for owner in tree.body
        if isinstance(owner, ast.ClassDef)
        for method in [_named(owner.body, node.name, _FUNCTIONS)]  # type: ignore[attr-defined]
        if method is not None
    ]
    if len(owners) != 1:
        return None
    name, text, method = owners[0]
    modules[name] = _splice(text, method, block)
    return name


def _replace_assignment(modules: dict[str, str], node: ast.AST, patch_lines: list[str]) -> str | None:
    target = _assignment_name(node)
    for name, text, tree in _parsed(modules):
        existing = next((item for item in tree.body if _assignment_name(item) == target), None)
        if existing is not None:
            modules[name] = _splice(text, existing, _block_of(patch_lines, node))
            return name
    return None


def _apply_node(modules: dict[str, str], node: ast.AST, patch_lines: list[str]) -> str | None:
    if isinstance(node, ast.ClassDef):
        return _merge_class(modules, node, patch_lines)
    if isinstance(node, _FUNCTIONS):
        return _replace_function(modules, node, patch_lines)
    if _assignment_name(node):
        return _replace_assignment(modules, node, patch_lines)
    return None


def _add_new_code(
    modules: dict[str, str], home: str, nodes: list[ast.stmt], tree: ast.Module, patch_lines: list[str]
) -> None:
    """Add what the reply introduced. Stray statements such as demo calls are dropped."""
    text = modules[home]
    if not nodes or not parses(text):
        return
    used: set[str] = set()
    for node in tree.body:
        if isinstance(node, _DEFINITIONS):
            used |= _referenced_names(node)
    imports: list[str] = []
    constants: list[str] = []
    definitions: list[str] = []
    for node in nodes:
        block = _block_of(patch_lines, node)
        if isinstance(node, _IMPORTS):
            if "\n".join(block) not in text:
                imports += block
        elif isinstance(node, _DEFINITIONS):
            definitions += ["", "", *block]
        elif _assignment_name(node) in used:
            constants += block
    lines = text.splitlines()
    first_definition = next(
        (_start_line(node) - 1 for node in ast.parse(text).body if isinstance(node, _DEFINITIONS)),
        len(lines),
    )
    lines[first_definition:first_definition] = [*constants, *([""] if constants else [])]
    modules[home] = "\n".join([*imports, *lines, *definitions]) + "\n"


def _replace_broken_file(modules: dict[str, str], tree: ast.Module, block: str) -> bool:
    """A module with a syntax error can only be replaced as a whole file."""
    offered = {node.name for node in tree.body if isinstance(node, _DEFINITIONS)}
    for name, text in modules.items():
        if parses(text):
            continue
        required = set(_TOP_LEVEL_NAME_RE.findall(text))
        if offered and required <= offered:
            modules[name] = block.rstrip() + "\n"
            return True
    return False


def _apply_block(modules: dict[str, str], block: str) -> None:
    tree = ast.parse(block)
    if _replace_broken_file(modules, tree, block):
        return
    patch_lines = block.splitlines()
    home: str | None = None
    new_code: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, _FUNCTIONS) and _is_placeholder(node):
            continue
        touched = _apply_node(modules, node, patch_lines)
        if touched is None:
            new_code.append(node)
        else:
            home = home or touched
    _add_new_code(modules, home or next(iter(modules)), new_code, tree, patch_lines)


def apply_reply(sources: dict[str, str], blocks: list[str]) -> dict[str, str]:
    """New text of the modules after applying a reply's code blocks.

    Definitions and constants are matched by name across all modules, so the
    model need not say which file it edits. Raises SyntaxError on invalid code.
    """
    modules = dict(sources)
    for block in blocks:
        _apply_block(modules, block)
    for name, text in modules.items():
        if text != sources[name]:
            ast.parse(text)
    return modules


def excerpt(sources: dict[str, str], failures: str, depth: int = 2) -> dict[str, str]:
    """The parts of big modules that matter: what failures mention and what that calls."""
    parsed = _parsed(sources)
    index: dict[str, ast.AST] = {}
    for _, _, tree in parsed:
        for node in tree.body:
            if isinstance(node, _DEFINITIONS):
                index.setdefault(node.name, node)
            if isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, _FUNCTIONS):
                        index.setdefault(member.name, node)

    selected = {id(index[word]): index[word] for word in _WORD_RE.findall(failures) if word in index}
    frontier = list(selected.values())
    for _ in range(depth):
        following: list[ast.AST] = []
        for node in frontier:
            for name in _referenced_names(node):
                target = index.get(name)
                if target is not None and id(target) not in selected:
                    selected[id(target)] = target
                    following.append(target)
        frontier = following

    result = {name: text for name, text in sources.items() if not parses(text)}
    for name, text, tree in parsed:
        lines = text.splitlines()
        if not any(id(node) in selected for node in tree.body):
            continue
        kept: list[str] = []
        for node in tree.body:
            snippet = "\n".join(lines[_start_line(node) - 1 : node.end_lineno])  # type: ignore[attr-defined]
            short_constant = _assignment_name(node) and len(snippet) <= _MAX_EXCERPT_CONSTANT_CHARS
            if id(node) in selected or isinstance(node, _IMPORTS) or short_constant:
                kept.append(snippet)
        result[name] = "\n\n".join(kept)
    return result


def failure_count(output: str) -> int:
    return len(_FAILURE_RE.findall(output))


def made_worse(before: str, after: str) -> bool:
    """More tests fail now. Comparable only if both runs got as far as running tests."""
    if not (_RAN_TESTS_RE.search(before) and _RAN_TESTS_RE.search(after)):
        return False
    return failure_count(after) > failure_count(before)


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
