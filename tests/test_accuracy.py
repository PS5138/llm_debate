"""Regression tests for core.scoring.accuracy.find_answer.

The bug we're guarding against: a verbose judge output like
    "Final answer: Answer: B"
was matched as both A *and* B by a permissive case-insensitive regex
(the literal `A` from the second word "Answer" was being treated as the
chosen letter). That silently inverted scoring for any condition where
the judge wrapped its answer in "Final answer: ...". The cases below pin
the correct behaviour so the regex can't drift back into that shape.

Run:
    python tests/test_accuracy.py
"""

import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.scoring.accuracy import find_answer  # noqa: E402


CASES = [
    # (text, letter, expected, description)
    # The original bug.
    ("Final answer: Answer: B", "A", False, "verbose-final-answer-B not A"),
    ("Final answer: Answer: B", "B", True, "verbose-final-answer-B is B"),
    ("Final answer: Answer: A", "A", True, "verbose-final-answer-A is A"),
    ("Final answer: Answer: A", "B", False, "verbose-final-answer-A not B"),

    # Clean canonical outputs.
    ("Answer: A", "A", True, "clean-A"),
    ("Answer: A", "B", False, "clean-A not B"),
    ("Answer: B", "B", True, "clean-B"),
    ("Answer: B", "A", False, "clean-B not A"),

    # Bracketed forms.
    ("Answer: <A>", "A", True, "bracketed-A"),
    ("Answer: <B>", "B", True, "bracketed-B"),

    # With <thinking> wrapper, answer outside.
    ("<thinking>weighing options</thinking>\nAnswer: B", "B",
     True, "thinking-then-clean-B"),
    ("<thinking>weighing options</thinking>\nAnswer: B", "A",
     False, "thinking-then-clean-B not A"),

    # With <thinking> wrapper, answer inside (still picked).
    ("<thinking>Reasoning... Final answer: Answer: B</thinking>", "B",
     True, "thinking-internal-final-answer-B"),
    ("<thinking>Reasoning... Final answer: Answer: B</thinking>", "A",
     False, "thinking-internal-final-answer-B not A"),

    # Multiple Answer: lines — the LAST one wins (judge changed mind).
    ("Answer: A\nOn reflection: Answer: B", "B",
     True, "reconsidered-to-B"),
    ("Answer: A\nOn reflection: Answer: B", "A",
     False, "reconsidered-not-A"),

    # gpt-5.5 swap-style output from the smoke run.
    (
        "<thinking>\nVerified evidence supports hoarse/soft voice...\n"
        "</thinking>\n\nAnswer: B",
        "B",
        True,
        "smoke-gpt5_5-swap-B",
    ),
    # gpt-4o-mini-style output from the smoke run (the case that exposed the bug).
    (
        "<thinking>\n1. Debater A argues for X...\n2. Debater B argues for Y...\n"
        "...the more likely diagnosis...\n\nFinal answer: Answer: B\n</thinking>",
        "B",
        True,
        "smoke-gpt4o-final-answer-B",
    ),
    (
        "<thinking>\n1. Debater A argues for X...\n2. Debater B argues for Y...\n"
        "...the more likely diagnosis...\n\nFinal answer: Answer: B\n</thinking>",
        "A",
        False,
        "smoke-gpt4o-final-answer-B not A",
    ),
]


def main() -> int:
    failures = []
    for text, letter, expected, desc in CASES:
        got = find_answer(text, letter)
        if got != expected:
            failures.append((desc, letter, expected, got, text))
            print(f"FAIL  [{desc}] letter={letter}  expected={expected}  got={got}")
        else:
            print(f"ok    [{desc}] letter={letter}  -> {got}")

    print()
    if failures:
        print(f"{len(failures)} regression(s) detected.")
        return 1
    print(f"{len(CASES)} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
