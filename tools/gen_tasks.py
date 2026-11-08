"""Generate a training set in the format of the published examples.

The published examples are all solvable, so this generator also produces
unsolvable variants. Each task.json gets extra `feasible` and `kind` fields that
only tools/score.py and tools/fit_estimator.py read.

Usage: python tools/gen_tasks.py OUTPUT_DIR [--count 200] [--seed 7]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

ANSWER_SUFFIX = (
    " Work only inside the current directory. When you are done, write ONLY the final "
    "answer to the file answer.txt in the current directory (for example with bash: "
    "echo 42 > answer.txt)."
)
DONE_SUFFIX = (
    " Work only inside the current directory. When you are done, write ONLY the word DONE "
    "to the file answer.txt in the current directory (for example with bash: "
    "echo 42 > answer.txt)."
)
FIRST = "Alice Kira Rosa Chen Tanya Sergei Irina Jonas Farid Grace Quinn Hiro Dmitri Olga".split()
LAST = "Novak Kowalski Wang Garcia Muller Haddad Sato Smith Petrov Ivanova".split()
COMPANIES = (
    "Granite AI|Arktika Labs|Fjord Data|Bluefin Systems|Juniper Foods|Helix Motors|"
    "Ember Bio|Dune Robotics|Cobalt Works|Iris Optics"
).split("|")
CITIES = "Osaka Lisbon Oslo Lagos Lima Seoul Dublin Quito Hanoi Tunis".split()
PRODUCTS = "widget gadget sprocket gizmo".split()
REGIONS = "north south east west".split()


@dataclass
class Generated:
    instruction: str
    expected: str | None
    check: str
    kind: str = "ok"
    files: dict[str, str] = field(default_factory=dict)


def _math_steps(rng: random.Random, count: int) -> tuple[list[str], int, int]:
    start = rng.randint(500, 2500)
    value = start
    parts: list[str] = []
    for _ in range(count):
        roll = rng.randrange(4)
        if roll == 0:
            per_day, days = rng.randint(5, 40), rng.randint(2, 5)
            value -= per_day * days
            parts.append(f"It ships {per_day} boxes per day for {days} days.")
        elif roll == 1:
            times, size = rng.randint(2, 4), rng.randint(20, 90)
            value += times * size
            parts.append(f"It receives {times} deliveries of {size} boxes each.")
        elif roll == 2:
            shares = rng.randint(2, 3)
            value //= shares
            parts.append(
                f"Then the stock is split equally into {shares} warehouses and it keeps "
                "one share (rounded down)."
            )
        else:
            value *= 2
            parts.append("Then the stock is doubled by merging with a sister warehouse.")
    return parts, start, value


def gen_math(rng: random.Random, feasible: bool) -> Generated:
    parts, start, value = _math_steps(rng, rng.randint(2, 6))
    opening = f"A warehouse starts with {start} boxes."
    kind = "ok"
    if not feasible:
        kind = rng.choice(["unknown_start", "unknown_quantity", "zero_split"])
        if kind == "unknown_start":
            opening = "A warehouse starts with an undisclosed number of boxes."
        elif kind == "unknown_quantity":
            parts.insert(rng.randrange(len(parts) + 1), "It ships an unspecified number of boxes.")
        else:
            parts.insert(
                rng.randrange(len(parts) + 1),
                "Then the stock is split equally into 0 warehouses and it keeps one share.",
            )
    text = " ".join([opening, *parts]) + (
        " How many boxes does it have at the end? Compute it with the bash tool (python3 -c)."
    )
    return Generated(text + ANSWER_SUFFIX, str(value) if feasible else None, "exact_int", kind)


def _build_kb(rng: random.Random, people_count: int) -> tuple[dict, dict]:
    names = sorted({f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(people_count * 2)})
    rng.shuffle(names)
    names = names[:people_count]
    companies = {
        name: {"Headquarters": rng.choice(CITIES), "Founded": rng.randint(1950, 2015)}
        for name in COMPANIES
    }
    people = {
        name: {
            "Born": rng.randint(1950, 2000),
            "Employer": rng.choice(COMPANIES),
            "Mentor": rng.choice([other for other in names if other != name]),
        }
        for name in names
    }
    return people, companies


def _kb_files(people: dict, companies: dict) -> dict[str, str]:
    files: dict[str, str] = {}
    for prefix, table in (("person", people), ("company", companies)):
        for name, fields in table.items():
            body = "".join(f"{key}: {value}\n" for key, value in fields.items())
            files[f"kb/{prefix}_{name.lower().replace(' ', '_')}.md"] = f"# {name}\n\n{body}"
    return files


def gen_search(rng: random.Random, feasible: bool) -> Generated:
    people, companies = _build_kb(rng, rng.randint(22, 28))
    person = rng.choice(list(people))
    record = people[person]
    employer = companies[record["Employer"]]
    mentor = people[record["Mentor"]]
    options = [
        (f"In which year was {person} born?", record["Born"]),
        (f"In which city is the headquarters of {person}'s employer?", employer["Headquarters"]),
        (f"Who is the mentor of {person}?", record["Mentor"]),
        (f"In which year was {person}'s employer founded?", employer["Founded"]),
        (f"In which year was {person}'s mentor born?", mentor["Born"]),
        (
            f"In which city is the headquarters of the employer of {person}'s mentor?",
            companies[mentor["Employer"]]["Headquarters"],
        ),
    ]
    question, answer = rng.choice(options)
    kind = "ok"
    if not feasible:
        kind = rng.choice(["unknown_person", "missing_company", "missing_field"])
        if kind == "unknown_person":
            stranger = "Zelda Quartermaine"
            question = question.replace(person, stranger)
        elif kind == "missing_company":
            del companies[record["Employer"]]
            question = f"In which city is the headquarters of {person}'s employer?"
        else:
            del employer["Headquarters"]
            question = f"In which city is the headquarters of {person}'s employer?"
    text = (
        "The directory kb/ contains a small knowledge base of markdown files. "
        f"{question} Use the file tools or bash (ls, grep, cat) to find it."
    )
    return Generated(
        text + ANSWER_SUFFIX,
        str(answer) if feasible else None,
        "exact_str",
        kind,
        _kb_files(people, companies),
    )


def _sales_csv(rows: list[dict]) -> str:
    lines = ["order_id,region,product,month,units,price"]
    lines += [
        f"{r['order_id']},{r['region']},{r['product']},{r['month']},{r['units']},{r['price']}"
        for r in rows
    ]
    return "\n".join(lines) + "\n"


def gen_data(rng: random.Random, feasible: bool) -> Generated:
    count = rng.choice([20, 50, 100, 300])
    rows = [
        {
            "order_id": index + 1,
            "region": rng.choice(REGIONS),
            "product": rng.choice(PRODUCTS),
            "month": rng.randint(1, 12),
            "units": rng.randint(1, 50),
            "price": rng.randint(10, 100),
        }
        for index in range(count)
    ]
    product, region = rng.choice(PRODUCTS), rng.choice(REGIONS)
    low = rng.randint(1, 10)
    high = low + rng.randint(1, 2)
    selected = [r for r in rows if r["product"] == product and low <= r["month"] <= high]
    in_region = [r for r in rows if r["region"] == region]
    options = [
        (
            f"How many units of product '{product}' were sold in months {low} through "
            f"{high} inclusive?",
            sum(r["units"] for r in selected),
        ),
        (
            f"What is the total revenue (units times price) of region '{region}'?",
            sum(r["units"] * r["price"] for r in in_region),
        ),
        (f"How many orders are there for region '{region}'?", len(in_region)),
        (f"What is the largest number of units in a single '{product}' order?",
         max((r["units"] for r in rows if r["product"] == product), default=0)),
    ]
    question, answer = rng.choice(options)
    files = {"sales.csv": _sales_csv(rows)}
    columns = "order_id,region,product,month,units,price"
    kind = "ok"
    if not feasible:
        kind = rng.choice(["missing_file", "missing_column", "bad_schema"])
        if kind == "missing_file":
            files = {}
        elif kind == "missing_column":
            question = f"What is the average discount of region '{region}'?"
        else:
            columns = "order_id,region,product,month,quantity,price"
    text = (
        f"The file sales.csv has columns {columns} ({count} rows). {question} "
        "Use bash with python3 to compute it."
    )
    return Generated(text + ANSWER_SUFFIX, str(answer) if feasible else None, "exact_int", kind, files)


# name -> (correct body, buggy body, test call, expected repr, signature)
LIBRARY = {
    "total": ("s = 0\n    for x in xs:\n        s += x\n    return s",
              "s = 0\n    for x in xs:\n        s -= x\n    return s", "[1, 2, 3]", "6", "xs"),
    "maximum": ("best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best",
                "best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best",
                "[3, 9, 2]", "9", "xs"),
    "minimum": ("best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best",
                "best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best",
                "[3, 9, 2]", "2", "xs"),
    "average": ("return sum(xs) / len(xs)", "return sum(xs) / (len(xs) + 1)", "[2, 4, 6]", "4", "xs"),
    "count_even": ("return sum(1 for x in xs if x % 2 == 0)",
                   "return sum(1 for x in xs if x % 2 == 1)", "[1, 2, 4, 6, 7]", "3", "xs"),
    "reverse_words": ("return ' '.join(reversed(s.split()))", "return ' '.join(sorted(s.split()))",
                      "'c a b'", "'b a c'", "s"),
    "first_n": ("return xs[:n]", "return xs[:n + 1]", "[5, 6, 7, 8], 3", "[5, 6, 7]", "xs, n"),
    "last_n": ("return xs[-n:]", "return xs[n:]", "[5, 6, 7, 8], 1", "[8]", "xs, n"),
    "is_palindrome": ("return s == s[::-1]", "return s != s[::-1]", "'abba'", "True", "s"),
    "factorial": ("r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r",
                  "r = 1\n    for i in range(2, n):\n        r *= i\n    return r", "5", "120", "n"),
    "unique": ("return sorted(set(xs))", "return sorted(xs)", "[3, 1, 3, 2]", "[1, 2, 3]", "xs"),
    "clamp": ("return max(lo, min(x, hi))", "return min(lo, max(x, hi))", "15, 0, 10", "10", "x, lo, hi"),
}


def gen_coding(rng: random.Random, feasible: bool) -> Generated:
    names = rng.sample(list(LIBRARY), rng.randint(4, 7))
    buggy = set(rng.sample(names, rng.randint(1, 3)))
    module = ""
    tests = ""
    for name in names:
        correct, wrong, call, expected, signature = LIBRARY[name]
        body = wrong if name in buggy else correct
        module += f"def {name}({signature}):\n    {body}\n\n\n"
        tests += (
            f"    def test_{name}(self):\n"
            f"        self.assertEqual(m.{name}({call}), {expected})\n\n"
        )
    header = "import unittest\n\nimport mylib as m\n"
    kind = "ok"
    if not feasible:
        kind = rng.choice(["contradiction", "missing_module", "always_fails"])
        name = names[0]
        call, expected = LIBRARY[name][2], LIBRARY[name][3]
        if kind == "contradiction":
            tests += (
                f"    def test_{name}_again(self):\n"
                f"        self.assertEqual(m.{name}({call}), {expected} + [0] if False else 'nope')\n\n"
            ).replace(f"{expected} + [0] if False else 'nope'", "-12345")
        elif kind == "missing_module":
            header = "import unittest\n\nimport quantum_flux_sdk\nimport mylib as m\n"
        else:
            tests += "    def test_impossible(self):\n        self.fail('cannot pass')\n\n"
    test_file = (
        f"{header}\n\nclass T(unittest.TestCase):\n{tests.rstrip()}\n\n\n"
        "if __name__ == '__main__':\n    unittest.main()\n"
    )
    text = (
        "Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs "
        "in mylib.py (do not edit the tests) until all tests pass."
    )
    return Generated(
        text + DONE_SUFFIX,
        "tests_pass" if feasible else None,
        "unittest",
        kind,
        {"mylib.py": module.rstrip() + "\n", "test_mylib.py": test_file},
    )


GENERATORS = [("math", gen_math), ("search", gen_search), ("data_analysis", gen_data), ("coding", gen_coding)]


def write_task(root: Path, task_id: str, task_type: str, generated: Generated, reward: int, feasible: bool) -> None:
    directory = root / task_id
    directory.mkdir(parents=True)
    spec = {
        "task_id": task_id,
        "task_type": task_type,
        "instruction": generated.instruction,
        "reward": reward,
        "expected": generated.expected,
        "check": generated.check,
        "feasible": feasible,
        "kind": generated.kind,
    }
    (directory / "task.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    for relative, content in generated.files.items():
        path = directory / "workspace" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    if task_type != "math":
        (directory / "workspace").mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--infeasible", type=float, default=0.2)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    root = Path(args.output)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for index in range(args.count):
        task_type, generator = GENERATORS[index % len(GENERATORS)]
        feasible = rng.random() >= args.infeasible
        reward = rng.randint(1, 5)
        write_task(root, f"{task_type}_{index:04d}", task_type, generator(rng, feasible), reward, feasible)
    print(f"wrote {args.count} tasks to {root}")


if __name__ == "__main__":
    main()
