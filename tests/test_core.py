"""Unit tests for the parts that need no model. Run: python -m unittest discover tests"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import feasibility  # noqa: E402
from harness.ledger import CallUnaffordable, Ledger, TaskCapExceeded, TaskMeter, call_cost  # noqa: E402
from harness.planner import knapsack  # noqa: E402
from harness.solvers import kb, math_solver, pycode  # noqa: E402
from harness.solvers.common import (  # noqa: E402
    answer_from_output,
    extract_code,
    normalize_number,
    SolveResult,
    Status,
)
from harness.tasks import Task, load_tasks, write_answer  # noqa: E402

SUFFIX = (
    " Work only inside the current directory. When you are done, write ONLY the final answer "
    "to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
)


def make_task(root: Path, task_type: str, instruction: str, files: dict[str, str]) -> Task:
    for name, text in files.items():
        path = root / "workspace" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "workspace").mkdir(exist_ok=True)
    return Task("t", task_type, instruction, 1.0, root)


class TaskLoadingTest(unittest.TestCase):
    def test_works_without_expected_field(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "a").mkdir()
            (root / "a" / "task.json").write_text(
                json.dumps({"task_id": "a", "task_type": "math", "instruction": "x", "reward": 2})
            )
            (root / "stray.txt").write_text("not a task")
            tasks = load_tasks(root)
            self.assertEqual([t.task_id for t in tasks], ["a"])
            self.assertFalse(hasattr(tasks[0], "expected"))

    def test_answer_goes_to_task_root_even_with_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "workspace").mkdir()
            task = Task("t", "coding", "x", 1.0, root)
            write_answer(task, "DONE")
            self.assertEqual((root / "answer.txt").read_text(), "DONE\n")
            self.assertFalse((root / "workspace" / "answer.txt").exists())

    def test_question_drops_boilerplate(self) -> None:
        task = Task("t", "math", "Add 1 and 2. Compute it with the bash tool (python3 -c)." + SUFFIX, 1.0, Path("."))
        self.assertEqual(task.question, "Add 1 and 2.")


class LedgerTest(unittest.TestCase):
    def test_cost_weights_input_at_one_tenth(self) -> None:
        self.assertEqual(call_cost(100, 7), 17)

    def test_reservation_blocks_overrun_and_settle_releases(self) -> None:
        ledger = Ledger(100, safety=0)
        held = ledger.reserve(80)
        with self.assertRaises(CallUnaffordable):
            ledger.reserve(30)
        ledger.settle(held, 100, 10)
        self.assertEqual(ledger.available, 80)

    def test_task_meter(self) -> None:
        meter = TaskMeter(10)
        meter.add(8)
        with self.assertRaises(TaskCapExceeded):
            meter.check(3)


class KnapsackTest(unittest.TestCase):
    def test_takes_everything_when_it_fits(self) -> None:
        self.assertEqual(knapsack([1, 2], [1, 1], 5), [0, 1])

    def test_prefers_best_combination_over_greedy_density(self) -> None:
        self.assertEqual(knapsack([6, 5, 5], [3, 2, 2], 4), [1, 2])

    def test_zero_capacity(self) -> None:
        self.assertEqual(knapsack([1], [1], 0), [])


class ParsingTest(unittest.TestCase):
    def test_extract_code_with_and_without_fence(self) -> None:
        self.assertEqual(extract_code("```python\nprint(1)\n```"), "print(1)")
        self.assertEqual(extract_code("print(2)"), "print(2)")

    def test_normalize_number(self) -> None:
        self.assertEqual(normalize_number("389.0"), "389")
        self.assertEqual(normalize_number("1,735"), "1735")
        self.assertEqual(normalize_number("Osaka"), "Osaka")

    def test_program_failure_asks_for_retry(self) -> None:
        self.assertIs(answer_from_output("", "ZeroDivisionError: x", False).status, Status.RETRY)
        self.assertIs(answer_from_output("IMPOSSIBLE\n", "", True).status, Status.IMPOSSIBLE)
        self.assertEqual(answer_from_output("debug\n42\n", "", True).answer, "42")


class MathDecisionTest(unittest.TestCase):
    @staticmethod
    def solved(answer: str) -> SolveResult:
        return SolveResult(Status.SOLVED, answer=answer)

    def test_referee_siding_with_second_overrides_first(self) -> None:
        result = math_solver.decide(self.solved("1"), self.solved("2"), self.solved("2"))
        self.assertEqual(result.answer, "2")

    def test_no_majority_keeps_first(self) -> None:
        result = math_solver.decide(self.solved("249"), self.solved("126"), self.solved("1013"))
        self.assertEqual(result.answer, "249")

    def test_failed_referee_keeps_first(self) -> None:
        result = math_solver.decide(self.solved("1"), self.solved("2"), SolveResult(Status.RETRY))
        self.assertEqual(result.answer, "1")

    def test_one_expression_is_not_stepwise(self) -> None:
        question = "Start with 5 boxes. Add 3 boxes. Double the stock. How many are there?"
        self.assertFalse(math_solver.is_stepwise(question, "print((5 + 3) * 2)"))
        self.assertTrue(math_solver.is_stepwise(question, "b = 5\nb += 3\nb *= 2\nprint(b)"))
        self.assertFalse(math_solver.is_stepwise(question, "def broken(:"))

    def test_ignored_numbers(self) -> None:
        self.assertEqual(math_solver.ignored_numbers("Start 10, add 5.", "print(10 + 5)"), [])
        self.assertEqual(math_solver.ignored_numbers("Start 10, add 5.", "print(126)"), ["10", "5"])


class PatchingTest(unittest.TestCase):
    SOURCE = "import os\n\n\ndef a():\n    return 1\n\n\ndef b():\n    return 2\n"

    def test_replaces_only_named_definition(self) -> None:
        patched = pycode.apply_definitions(self.SOURCE, "def b():\n    return 3\n")
        self.assertIn("return 1", patched)
        self.assertIn("return 3", patched)
        self.assertNotIn("return 2", patched)

    def test_adds_new_definition_and_import(self) -> None:
        patched = pycode.apply_definitions(self.SOURCE, "import re\n\ndef c():\n    return re\n")
        self.assertTrue(patched.startswith("import re"))
        self.assertIn("def c():", patched)

    def test_rejects_bad_syntax(self) -> None:
        with self.assertRaises(SyntaxError):
            pycode.apply_definitions(self.SOURCE, "def b(:\n")

    def test_summarize_keeps_assertion(self) -> None:
        report = (
            "F\n" + "=" * 70 + "\nFAIL: test_total (__main__.T.test_total)\n" + "-" * 70 + "\n"
            'Traceback (most recent call last):\n  File "t.py", line 11, in test_total\n'
            "    self.assertEqual(m.total([1, 2, 3]), 6)\nAssertionError: -6 != 6\n\n"
            + "-" * 70 + "\nRan 1 test in 0.000s\n\nFAILED (failures=1)\n"
        )
        summary = pycode.summarize_failures(report)
        self.assertIn("FAIL: test_total", summary)
        self.assertIn("m.total([1, 2, 3])", summary)
        self.assertIn("-6 != 6", summary)
        self.assertNotIn("Ran 1 test", summary)
        self.assertEqual(pycode.failure_count(report), 1)


class KnowledgeBaseTest(unittest.TestCase):
    def test_follows_links_from_named_person(self) -> None:
        notes = {
            "Ann Lee": kb.Note("Ann Lee", "# Ann Lee\n\nBorn: 1980\nEmployer: Acme Corp"),
            "Acme Corp": kb.Note("Acme Corp", "# Acme Corp\n\nHeadquarters: Oslo"),
            "Bob Ray": kb.Note("Bob Ray", "# Bob Ray\n\nBorn: 1970"),
        }
        question = "Where is Ann Lee's employer based?"
        self.assertEqual([n.title for n in kb.relevant_notes(question, notes, 2)], ["Ann Lee", "Acme Corp"])
        self.assertEqual([n.title for n in kb.relevant_notes(question, notes, 0)], ["Ann Lee"])


class FeasibilityTest(unittest.TestCase):
    TESTS = (
        "import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n"
        "    def test_a(self):\n        self.assertEqual(m.f(1), 2)\n"
    )

    def _check(self, task_type: str, instruction: str, files: dict[str, str]) -> feasibility.Verdict:
        with tempfile.TemporaryDirectory() as raw:
            return feasibility.check(make_task(Path(raw), task_type, instruction, files))

    def test_valid_coding_task_accepted(self) -> None:
        verdict = self._check("coding", "Run `python3 test_mylib.py`. Fix mylib.py.",
                              {"mylib.py": "def f(x):\n    return x\n", "test_mylib.py": self.TESTS})
        self.assertTrue(verdict.feasible, verdict.reason)

    def test_contradictory_tests_rejected(self) -> None:
        tests = self.TESTS + "    def test_b(self):\n        self.assertEqual(m.f(1), 3)\n"
        verdict = self._check("coding", "Run `python3 test_mylib.py`. Fix mylib.py.",
                              {"mylib.py": "def f(x):\n    return x\n", "test_mylib.py": tests})
        self.assertFalse(verdict.feasible)

    def test_missing_test_module_rejected(self) -> None:
        tests = "import no_such_sdk_here\n" + self.TESTS
        verdict = self._check("coding", "Run `python3 test_mylib.py`. Fix mylib.py.",
                              {"mylib.py": "def f(x):\n    return x\n", "test_mylib.py": tests})
        self.assertFalse(verdict.feasible)

    def test_missing_data_file_rejected(self) -> None:
        self.assertFalse(self._check("data_analysis", "The file sales.csv has data. Sum it.", {}).feasible)

    def test_stated_column_absent_rejected(self) -> None:
        verdict = self._check("data_analysis", "The file s.csv has columns a,b,c (2 rows). Sum c.",
                              {"s.csv": "a,b\n1,2\n"})
        self.assertFalse(verdict.feasible)

    def test_unknown_person_rejected_known_accepted(self) -> None:
        note = {"kb/person_ann_lee.md": "# Ann Lee\n\nBorn: 1980\n"}
        question = "The directory kb/ contains notes. In which year was {} born?"
        self.assertFalse(self._check("search", question.format("Zed Quill"), note).feasible)
        self.assertTrue(self._check("search", question.format("Ann Lee"), note).feasible)

    def test_answer_file_is_not_a_missing_reference(self) -> None:
        self.assertTrue(self._check("math", "Add 1 and 2, save to answer.txt.", {}).feasible)

    def test_file_to_create_is_not_missing(self) -> None:
        self.assertTrue(self._check("misc", "Create a file called out.txt with the word hi.", {}).feasible)

    def test_network_task_rejected(self) -> None:
        self.assertFalse(self._check("math", "Download https://example.com/x and count lines.", {}).feasible)


if __name__ == "__main__":
    unittest.main()
