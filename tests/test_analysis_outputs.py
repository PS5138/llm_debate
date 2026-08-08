"""Regression tests for analysis cohort and stale-output handling.

Run:
    python tests/test_analysis_outputs.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aggregate_medical_results import main as aggregate_main  # noqa: E402
from scripts.compute_paired_tests import build_tables  # noqa: E402
from scripts.compute_paired_tests import main as paired_tests_main  # noqa: E402


class AnalysisOutputRegressionTests(unittest.TestCase):
    def test_headline_uses_one_three_arm_complete_cohort(self) -> None:
        per_case = pd.DataFrame(
            [
                {
                    "family": "test",
                    "condition": "e1_info_asymmetry",
                    "judge_model": "judge",
                    "case_id": "1",
                    "acc": 1.0,
                },
                {
                    "family": "test",
                    "condition": "e1_info_asymmetry",
                    "judge_model": "judge",
                    "case_id": "2",
                    "acc": 0.0,
                },
                {
                    "family": "test",
                    "condition": "e1_info_asymmetry",
                    "judge_model": "judge",
                    "case_id": "3",
                    "acc": 1.0,
                },
            ]
        )
        baselines_case = pd.DataFrame(
            [
                {
                    "family": "test",
                    "arm": "B1",
                    "judge_model": "judge",
                    "case_id": "1",
                    "acc": 0.0,
                },
                {
                    "family": "test",
                    "arm": "B1",
                    "judge_model": "judge",
                    "case_id": "2",
                    "acc": np.nan,
                },
                {
                    "family": "test",
                    "arm": "B1",
                    "judge_model": "judge",
                    "case_id": "3",
                    "acc": 1.0,
                },
                {
                    "family": "test",
                    "arm": "B2",
                    "judge_model": "judge",
                    "case_id": "1",
                    "acc": 1.0,
                },
                {
                    "family": "test",
                    "arm": "B2",
                    "judge_model": "judge",
                    "case_id": "2",
                    "acc": 1.0,
                },
                {
                    "family": "test",
                    "arm": "B2",
                    "judge_model": "judge",
                    "case_id": "3",
                    "acc": np.nan,
                },
            ]
        )

        headline, paired, skipped = build_tables(per_case, baselines_case)

        self.assertEqual(skipped, [])
        self.assertEqual(len(headline), 1)
        row = headline.iloc[0]
        self.assertEqual(row["n_cases"], 1)
        self.assertEqual(row["blind_acc"], 0.0)
        self.assertEqual(row["debate_acc"], 1.0)
        self.assertEqual(row["oracle_acc"], 1.0)
        self.assertEqual(row["pgr"], 1.0)
        reconstructed_pgr = (row["debate_acc"] - row["blind_acc"]) / (
            row["oracle_acc"] - row["blind_acc"]
        )
        self.assertEqual(row["pgr"], reconstructed_pgr)

        # The two hypothesis tests remain pairwise-complete comparisons; they
        # need not discard a case merely because the unused third arm is NaN.
        pair_counts = paired.set_index("comparison")["n_cases"].to_dict()
        self.assertEqual(pair_counts["debate_vs_blind"], 2)
        self.assertEqual(pair_counts["debate_vs_oracle"], 2)

    def test_skipped_paired_analysis_removes_all_stale_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            out_dir.mkdir()
            family_dir = root / "missing" / "test"
            analysis_dir = family_dir / "analysis"
            analysis_dir.mkdir(parents=True)
            # A resumed directory can retain a perfectly parseable analysis
            # table even though it has no current debate judgements. It must
            # not be allowed to repopulate QC for the empty current run.
            pd.DataFrame([{"side": "A", "words": 10}]).to_csv(
                analysis_dir / "verbosity.csv", index=False
            )
            stale_names = (
                "headline_results.csv",
                "paired_tests.csv",
                "quality_control_summary.csv",
            )
            for name in stale_names:
                (out_dir / name).write_text("stale\n", encoding="utf-8")

            paired_tests_main(
                [family_dir],
                root / "missing-baselines",
                out_dir,
                None,
            )

            for name in stale_names:
                self.assertFalse((out_dir / name).exists(), name)

    def test_empty_aggregation_removes_stale_baseline_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(parents=True)
            stale_paths = [
                out_dir / "baselines_by_arm.csv",
                plots_dir / "01_accuracy_by_condition.png",
                plots_dir / "02_pgr_by_condition.png",
                plots_dir / "03_per_case_lift.png",
            ]
            for path in stale_paths:
                path.write_bytes(b"stale")

            aggregate_main(
                [root / "missing" / "test"],
                root / "missing-baselines",
                out_dir,
                False,
            )

            for path in stale_paths:
                self.assertFalse(path.exists(), path.name)


if __name__ == "__main__":
    unittest.main()
