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
    extract_code_blocks,
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
    CLASS = (
        "class C:\n    limit = 1\n\n    def x(self):\n        return 1\n\n"
        "    def y(self):\n        return 2\n"
    )

    @staticmethod
    def apply(source: str, patch: str) -> str:
        return pycode.apply_reply({"m.py": source}, [patch])["m.py"]

    def test_replaces_only_named_definition(self) -> None:
        patched = self.apply(self.SOURCE, "def b():\n    return 3\n")
        self.assertIn("return 1", patched)
        self.assertIn("return 3", patched)
        self.assertNotIn("return 2", patched)

    def test_adds_new_definition_and_import(self) -> None:
        patched = self.apply(self.SOURCE, "import re\n\ndef c():\n    return re\n")
        self.assertTrue(patched.startswith("import re"))
        self.assertIn("def c():", patched)

    def test_rejects_bad_syntax(self) -> None:
        with self.assertRaises(SyntaxError):
            self.apply(self.SOURCE, "def b(:\n")

    def test_replaces_module_constant(self) -> None:
        patched = self.apply("RATE = 1\n\n\ndef f():\n    return RATE\n", "RATE = 2\n")
        self.assertIn("RATE = 2", patched)
        self.assertNotIn("RATE = 1", patched)

    def test_partial_class_keeps_other_members(self) -> None:
        patched = self.apply(self.CLASS, "class C:\n    def x(self):\n        return 9\n")
        self.assertIn("return 9", patched)
        self.assertIn("return 2", patched)
        self.assertIn("limit = 1", patched)

    def test_partial_class_updates_attribute_and_adds_method(self) -> None:
        patched = self.apply(self.CLASS, "class C:\n    limit = 5\n\n    def z(self):\n        return 3\n")
        self.assertIn("limit = 5", patched)
        self.assertIn("    def z(self):", patched)

    def test_method_without_its_class(self) -> None:
        patched = self.apply(self.CLASS, "def y(self):\n    return 7\n")
        self.assertIn("        return 7", patched)
        self.assertNotIn("return 2", patched)

    def test_indented_methods_in_a_fence(self) -> None:
        reply = "```python\n    def x(self):\n        return 8\n\n    def y(self):\n        return 9\n```"
        patched = pycode.apply_reply({"m.py": self.CLASS}, extract_code_blocks(reply))["m.py"]
        self.assertIn("        return 8", patched)
        self.assertIn("        return 9", patched)
        self.assertIn("limit = 1", patched)

    def test_ellipsis_placeholder_changes_nothing(self) -> None:
        patched = self.apply(self.CLASS, "class C:\n    def y(self): ...\n")
        self.assertIn("return 2", patched)

    def test_stray_statements_are_dropped(self) -> None:
        patched = self.apply(self.SOURCE, "print(a())\nvalue = a()\n\ndef b():\n    return 3\n")
        self.assertNotIn("print(", patched)
        self.assertNotIn("value =", patched)

    def test_new_constant_used_by_patch_is_inserted_before_code(self) -> None:
        patched = self.apply(self.SOURCE, "LIMIT = 3\n\ndef b():\n    return LIMIT\n")
        self.assertLess(patched.index("LIMIT = 3"), patched.index("def a"))

    def test_edits_are_routed_across_modules(self) -> None:
        sources = {"a.py": "def f():\n    return 1\n", "b.py": "def g():\n    return 2\n"}
        patched = pycode.apply_reply(sources, ["def g():\n    return 3\n"])
        self.assertEqual(patched["a.py"], sources["a.py"])
        self.assertIn("return 3", patched["b.py"])

    def test_broken_file_is_replaced_whole(self) -> None:
        broken = "def f()\n    return 1\n\n\ndef g():\n    return 2\n"
        fixed = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
        self.assertEqual(self.apply(broken, fixed), fixed)

    def test_broken_file_is_kept_when_reply_is_partial(self) -> None:
        broken = "def f()\n    return 1\n\n\ndef g():\n    return 2\n"
        self.assertEqual(self.apply(broken, "def f():\n    return 1\n"), broken)

    def test_regression_needs_two_real_test_runs(self) -> None:
        ran = "F\nFAIL: test_a (x)\nRan 2 tests in 0.0s\n"
        worse = "F\nFAIL: test_a (x)\nFAIL: test_b (x)\nRan 2 tests in 0.0s\n"
        self.assertTrue(pycode.made_worse(ran, worse))
        self.assertFalse(pycode.made_worse(worse, ran))
        self.assertFalse(pycode.made_worse("SyntaxError: bad", worse))

    def test_excerpt_follows_calls_into_other_modules(self) -> None:
        sources = {
            "a.py": "from b import helper\n\n\ndef top():\n    return helper()\n\n\ndef other():\n    return 1\n",
            "b.py": "RATE = 2\n\n\ndef helper():\n    return RATE\n\n\ndef unused():\n    return 3\n",
        }
        shown = pycode.excerpt(sources, "FAIL: test_top\n  m.top() != 5")
        self.assertIn("def helper", shown["b.py"])
        self.assertIn("RATE = 2", shown["b.py"])
        self.assertNotIn("def unused", shown["b.py"])
        self.assertNotIn("def other", shown["a.py"])

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


class TestDiscoveryTest(unittest.TestCase):
    @staticmethod
    def layout(root: Path, *names: str) -> Path:
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        return root

    def test_script_command_from_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self.layout(Path(raw), "check_lib.py", "mylib.py")
            command = pycode.find_test_command("Run `python3 check_lib.py` now.", root)
            self.assertEqual(command, ["check_lib.py"])
            self.assertEqual(pycode.test_files(command, root), [root / "check_lib.py"])

    def test_module_command_from_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self.layout(Path(raw), "tests/test_c.py", "c.py")
            command = pycode.find_test_command("Run `python3 -m unittest discover -s tests` now.", root)
            self.assertEqual(command, ["-m", "unittest", "discover", "-s", "tests"])
            self.assertEqual(pycode.test_files(command, root), [root / "tests" / "test_c.py"])

    def test_falls_back_to_tests_in_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self.layout(Path(raw), "unit/test_c.py", "c.py")
            self.assertEqual(
                pycode.find_test_command("Fix the code.", root),
                ["-m", "unittest", "discover", "-s", "unit"],
            )

    def test_no_tests_at_all(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            self.assertIsNone(pycode.find_test_command("Fix it.", self.layout(Path(raw), "c.py")))

    def test_modules_ranked_and_tests_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = self.layout(Path(raw), "a.py", "b.py", "z.py", "tests/helper.py")
            test_path = root / "test_x.py"
            test_path.write_text("import b\n", encoding="utf-8")
            modules = pycode.source_modules("Fix the bugs in z.py.", root, [test_path])
            self.assertEqual([path.name for path in modules], ["z.py", "b.py", "a.py"])


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

    def test_tests_in_subdirectory_accepted(self) -> None:
        verdict = self._check("coding", "Run `python3 -m unittest discover -s tests`. Fix calc.py.",
                              {"calc.py": "def f(x):\n    return x\n",
                               "tests/test_calc.py": self.TESTS.replace("mylib", "calc")})
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
