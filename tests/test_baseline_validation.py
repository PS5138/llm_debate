"""Focused regression tests for app-side baseline validation.

Run:
    python tests/test_baseline_validation.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import runner  # noqa: E402
from scripts import resolve_baselines  # noqa: E402

JUDGES = ("weaker-test", "frontier-test")
EXPECTED_IDS = ("case-001", "case-002")


def _write_baseline_set(
    root: Path, rows: list[dict], judges: tuple[str, ...] = JUDGES
) -> None:
    for arm in ("baseline_blind", "baseline_oracle"):
        for judge in judges:
            judge_dir = root / arm / judge
            judge_dir.mkdir(parents=True, exist_ok=True)
            for filename in ("data0_judgement.csv", "data0_swap_judgement.csv"):
                pd.DataFrame(rows).to_csv(judge_dir / filename, index=False)


def _valid_rows() -> list[dict]:
    return [
        {"id": "case-001", "complete_judge": True, "answer_judge": "Answer: A"},
        {"id": "case-002", "complete_judge": True, "answer_judge": "Answer: B"},
    ]


def _check(description: str, condition: bool) -> bool:
    print(f"{'ok   ' if condition else 'FAIL '} [{description}]")
    return condition


def main() -> int:
    results: list[bool] = []
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)

        valid_root = temp / "valid"
        _write_baseline_set(valid_root, _valid_rows())
        results.append(
            _check(
                "completed parseable expected IDs pass",
                runner.baselines_dir_is_complete(
                    valid_root,
                    "openai",
                    judges=JUDGES,
                    min_rows=2,
                    expected_ids=EXPECTED_IDS,
                ),
            )
        )

        for description, mutation, expected_fragment in (
            (
                "nonblank answer with incomplete flag fails",
                lambda rows: rows[0].update(complete_judge=False),
                "incomplete",
            ),
            (
                "completed error string fails",
                lambda rows: rows[0].update(
                    answer_judge="Error message: authentication failed"
                ),
                "unparseable/error",
            ),
            (
                "completed unknown answer fails",
                lambda rows: rows[0].update(answer_judge="I cannot decide"),
                "unparseable/error",
            ),
            (
                "duplicate case IDs fail",
                lambda rows: rows[1].update(id="case-001"),
                "duplicate case ID",
            ),
            (
                "wrong cohort cannot satisfy count",
                lambda rows: rows[1].update(id="case-other"),
                "missing expected case ID",
            ),
        ):
            root = temp / description.replace(" ", "_")
            rows = _valid_rows()
            mutation(rows)
            _write_baseline_set(root, rows)
            issues = runner.missing_baseline_files(
                root,
                "openai",
                judges=JUDGES,
                min_rows=2,
                expected_ids=EXPECTED_IDS,
            )
            results.append(
                _check(
                    description,
                    bool(issues) and expected_fragment in issues[0],
                )
            )

        # Exercise canonical selection with its real family-model layout while
        # isolating the test from the committed data bundle.
        canonical_root = temp / "canonical"
        openai_root = canonical_root / "openai"
        pilot_path = temp / "pilot.csv"
        pd.DataFrame({"case_id": ["case-001", "case-002", "case-003"]}).to_csv(
            pilot_path, index=False
        )
        default_judges = (
            runner.FAMILY_MODELS["openai"]["weaker"],
            runner.FAMILY_MODELS["openai"]["frontier"],
        )
        original_root = runner.CANONICAL_BASELINES_ROOT
        original_pilot = runner.PILOT_CSV
        try:
            runner.CANONICAL_BASELINES_ROOT = canonical_root
            runner.PILOT_CSV = pilot_path
            _write_baseline_set(openai_root, _valid_rows(), default_judges)
            (openai_root / "results.csv").write_text("condition,accuracy\n")
            results.append(
                _check(
                    "canonical selection enforces requested count and IDs",
                    runner.canonical_baselines_path(
                        "openai", min_rows=2, expected_ids=EXPECTED_IDS
                    )
                    == openai_root,
                )
            )
            results.append(
                _check(
                    "canonical selection rejects a larger requested count",
                    runner.canonical_baselines_path(
                        "openai",
                        min_rows=3,
                        expected_ids=(*EXPECTED_IDS, "case-003"),
                    )
                    is None,
                )
            )

            wrong_ids_root = temp / "wrong-ids"
            wrong_rows = _valid_rows()
            wrong_rows[1]["id"] = "case-other"
            _write_baseline_set(wrong_ids_root, wrong_rows)
            resolved, _ = resolve_baselines.resolve(
                "openai", wrong_ids_root, judges=JUDGES, min_rows=2
            )
            results.append(
                _check(
                    "CLI resolver requires the requested pilot cohort",
                    resolved is None,
                )
            )

            resume_root = temp / "resume"
            resume_family = resume_root / "openai"
            resume_family.mkdir(parents=True)
            (resume_family / "run_metadata.json").write_text(
                json.dumps({"baselines_dir": str(openai_root)}) + "\n",
                encoding="utf-8",
            )
            canonical_before = {
                path.relative_to(openai_root): path.read_bytes()
                for path in openai_root.rglob("*")
                if path.is_file()
            }
            try:
                runner.DebateRun(
                    "openai",
                    3,
                    "test-openai-key",
                    "",
                    resume_from=resume_root,
                )
                rejected_resume = False
            except ValueError as exc:
                rejected_resume = "canonical baselines recorded" in str(exc)
            canonical_after = {
                path.relative_to(openai_root): path.read_bytes()
                for path in openai_root.rglob("*")
                if path.is_file()
            }
            results.append(
                _check(
                    "resume rejects invalid canonical data without rewriting it",
                    rejected_resume and canonical_after == canonical_before,
                )
            )
        finally:
            runner.CANONICAL_BASELINES_ROOT = original_root
            runner.PILOT_CSV = original_pilot

    failures = len([result for result in results if not result])
    print()
    if failures:
        print(f"{failures} regression(s) detected.")
        return 1
    print(f"{len(results)} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
