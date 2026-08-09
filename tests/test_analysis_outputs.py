"""Regression tests for analysis cohort and stale-output handling.

Run:
    python tests/test_analysis_outputs.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from the repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.aggregate_medical_results import (
    _family_label_from_path,
    _lift_contingencies_by_family,
    _lift_contingency,
    load_per_case_scores,
)
from scripts.aggregate_medical_results import main as aggregate_main  # noqa: E402
from scripts.aggregate_medical_results import plot_per_case_lift
from scripts.analyze_medical_debate import (  # noqa: E402
    concession_rows,
    debate_outcome_rows,
    display_path,
    plot_debate_outcomes,
    plot_verbosity,
    write_summary,
)
from scripts.compute_paired_tests import build_tables  # noqa: E402
from scripts.compute_paired_tests import main as paired_tests_main  # noqa: E402

MODEL_TRIOS = {
    "openai": {
        "frontier_model": "gpt-5.5",
        "weaker_model": "gpt-5.4-mini",
        "concession_model": "gpt-4o-mini",
    },
    "anthropic": {
        "frontier_model": "claude-opus-4-6",
        "weaker_model": "claude-sonnet-4-6",
        "concession_model": "gpt-4o-mini",
    },
}


def write_family_metadata(family_dir: Path, family: str) -> None:
    family_dir.mkdir(parents=True, exist_ok=True)
    record = {"family": family, **MODEL_TRIOS[family], "n_cases": 5}
    (family_dir / "run_metadata.json").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )


def write_judgement_pair(directory: Path, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "data0_judgement.csv", index=False)
    swapped = [dict(row) for row in rows]
    for row in swapped:
        if row.get("answer_judge") == "Answer: A":
            row["answer_judge"] = "Answer: B"
        elif row.get("answer_judge") == "Answer: B":
            row["answer_judge"] = "Answer: A"
    pd.DataFrame(swapped).to_csv(directory / "data0_swap_judgement.csv", index=False)


class AnalysisOutputRegressionTests(unittest.TestCase):
    def test_repo_paths_are_portable_in_generated_summaries(self) -> None:
        absolute = REPO_ROOT / "data" / "canonical_baselines" / "openai"
        self.assertEqual(display_path(absolute), "data/canonical_baselines/openai")

    def test_summary_writer_removes_trailing_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_summary(
                out_dir,
                "sample",
                ["Heading  ", "padded index   \nvalue 1\t", ""],
            )

            written = (out_dir / "sample_summary.txt").read_text()
            self.assertEqual(written, "Heading\npadded index\nvalue 1\n")
            self.assertTrue(written.endswith("\n"))

    def test_verbosity_plot_is_byte_identical_across_repeated_generation(
        self,
    ) -> None:
        verbosity = pd.DataFrame(
            [
                {"case_id": f"case-{i}", "side": side, "words": words}
                for i, (side, words) in enumerate(
                    [
                        ("correct", 118),
                        ("incorrect", 121),
                        ("correct", 132),
                        ("incorrect", 129),
                        ("correct", 144),
                        ("incorrect", 147),
                    ]
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first" / "04_verbosity.png"
            second = Path(tmp) / "second" / "04_verbosity.png"
            plot_verbosity(verbosity, "openai", first)
            plot_verbosity(verbosity, "openai", second)

            self.assertGreater(first.stat().st_size, 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_position_sensitive_lift_is_not_counted_as_correct(self) -> None:
        table = _lift_contingency(
            pd.DataFrame(
                [
                    {"blind_acc": 0.0, "debate_acc": 0.5},
                    {"blind_acc": 0.5, "debate_acc": 1.0},
                    {"blind_acc": 0.0, "debate_acc": 1.0},
                ]
            )
        )

        self.assertEqual(table.loc[0.0, 0.5], 1)
        self.assertEqual(table.loc[0.5, 1.0], 1)
        self.assertEqual(table.loc[0.0, 1.0], 1)
        self.assertEqual(table.values.sum(), 3)

    def test_per_case_lift_never_pools_families(self) -> None:
        lift = pd.DataFrame(
            [
                {
                    "family": "openai",
                    "condition": "e1_info_asymmetry",
                    "blind_acc": 0.0,
                    "debate_acc": 1.0,
                },
                {
                    "family": "anthropic",
                    "condition": "e1_info_asymmetry",
                    "blind_acc": 1.0,
                    "debate_acc": 0.0,
                },
                {
                    "family": "anthropic",
                    "condition": "e1_info_asymmetry",
                    "blind_acc": 1.0,
                    "debate_acc": 0.0,
                },
            ]
        )

        tables = _lift_contingencies_by_family(lift)
        self.assertEqual(tables["openai"]["e1_info_asymmetry"].loc[0.0, 1.0], 1)
        self.assertEqual(tables["openai"]["e1_info_asymmetry"].values.sum(), 1)
        self.assertEqual(tables["anthropic"]["e1_info_asymmetry"].loc[1.0, 0.0], 2)
        self.assertEqual(tables["anthropic"]["e1_info_asymmetry"].values.sum(), 2)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "03_per_case_lift.png"
            plot_per_case_lift(lift, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_metadata_family_label_supports_curated_bundle_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            curated = Path(tmp) / "openai_100"
            write_family_metadata(curated, "openai")
            self.assertEqual(_family_label_from_path(curated), "openai")

    def test_metadata_family_label_supports_older_root_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "2026-01-01_00-00-00_results"
            family_dir = run_root / "anthropic"
            family_dir.mkdir(parents=True)
            record = {"family": "anthropic", **MODEL_TRIOS["anthropic"], "n_cases": 5}
            (run_root / "run_metadata.json").write_text(
                json.dumps({"families": {"anthropic": record}}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(_family_label_from_path(family_dir), "anthropic")

    def test_aggregation_ignores_stale_debate_and_baseline_judges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            family = root / "openai"
            write_family_metadata(family, "openai")
            row = [
                {
                    "id": "case-1",
                    "complete_judge": True,
                    "answer_judge": "Answer: A",
                }
            ]
            write_judgement_pair(
                family / "debate_sim" / "e1_info_asymmetry_gpt-5.5", row
            )
            write_judgement_pair(
                family / "debate_sim" / "e1_info_asymmetry_stale-model", row
            )

            baselines = root / "baselines" / "openai"
            for arm in ("baseline_blind", "baseline_oracle"):
                write_judgement_pair(baselines / arm / "gpt-5.5", row)
                write_judgement_pair(baselines / arm / "gpt-5.4-mini", row)
                write_judgement_pair(baselines / arm / "stale-model", row)

            debate, baseline = load_per_case_scores([family], root / "baselines")

            self.assertEqual(set(debate["judge_model"]), {"gpt-5.5"})
            self.assertEqual(set(baseline["judge_model"]), {"gpt-5.5", "gpt-5.4-mini"})
            self.assertNotIn("stale-model", set(debate["judge_model"]))
            self.assertNotIn("stale-model", set(baseline["judge_model"]))

    def test_aggregation_rejects_missing_or_contradictory_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            family = Path(tmp) / "openai"
            family.mkdir()
            with self.assertRaisesRegex(ValueError, "missing run metadata"):
                load_per_case_scores([family], None)

            write_family_metadata(family, "openai")
            metadata_path = family / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["frontier_model"] = "stale-model"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed openai experiment"):
                load_per_case_scores([family], None)

    def test_aggregation_excludes_incomplete_and_unparseable_judgements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            family = root / "openai"
            write_family_metadata(family, "openai")
            debate_dir = family / "debate_sim" / "e1_info_asymmetry_gpt-5.5"
            baseline_dir = root / "baselines" / "openai" / "baseline_blind" / "gpt-5.5"
            debate_dir.mkdir(parents=True)
            baseline_dir.mkdir(parents=True)

            def write_pair(directory: Path) -> None:
                common = [
                    {
                        "id": "kept",
                        "complete_judge": True,
                        "answer_judge": "Answer: A",
                    },
                    {
                        "id": "incomplete",
                        "complete_judge": False,
                        # Answer-like partial output must never be scored.
                        "answer_judge": "Answer: A",
                    },
                    {
                        "id": "unparseable",
                        "complete_judge": True,
                        "answer_judge": "I cannot decide",
                    },
                ]
                pd.DataFrame(common).to_csv(
                    directory / "data0_judgement.csv", index=False
                )
                swapped = [dict(row) for row in common]
                swapped[0]["answer_judge"] = "Answer: B"
                swapped[1]["answer_judge"] = "Answer: B"
                pd.DataFrame(swapped).to_csv(
                    directory / "data0_swap_judgement.csv", index=False
                )

            write_pair(debate_dir)
            write_pair(baseline_dir)

            debate, baselines = load_per_case_scores([family], root / "baselines")

            self.assertEqual(debate["case_id"].tolist(), ["kept"])
            self.assertEqual(baselines["case_id"].tolist(), ["kept"])
            self.assertEqual(debate.iloc[0]["acc"], 1.0)
            self.assertEqual(baselines.iloc[0]["acc"], 1.0)

    def test_concession_qc_requires_complete_parseable_case_matched_passes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            family_dir = Path(tmp) / "openai"
            concession_dir = family_dir / "debate_sim" / "concession_gpt-4o-mini"
            concession_dir.mkdir(parents=True)
            original = pd.DataFrame(
                [
                    {
                        "id": "usable",
                        "complete_concession": True,
                        "answer_concession": "N",
                    },
                    {
                        "id": "conceded",
                        "complete_concession": True,
                        "answer_concession": "Y",
                    },
                    {
                        "id": "partial",
                        "complete_concession": False,
                        "answer_concession": "N",
                    },
                    {
                        "id": "missing",
                        "complete_concession": True,
                        "answer_concession": "N",
                    },
                    {
                        "id": "unparseable",
                        "complete_concession": True,
                        "answer_concession": "?",
                    },
                    # This row is unrelated to the debate cohort and must not leak in.
                    {
                        "id": "stale",
                        "complete_concession": True,
                        "answer_concession": "Y",
                    },
                ]
            )
            swapped = pd.DataFrame(
                [
                    {
                        "id": "usable",
                        "complete_concession": True,
                        "answer_concession": "N",
                    },
                    {
                        "id": "conceded",
                        "complete_concession": True,
                        "answer_concession": "N",
                    },
                    {
                        "id": "partial",
                        "complete_concession": True,
                        "answer_concession": "N",
                    },
                    {
                        "id": "unparseable",
                        "complete_concession": True,
                        "answer_concession": "N",
                    },
                    {
                        "id": "stale",
                        "complete_concession": True,
                        "answer_concession": "N",
                    },
                ]
            )
            original.to_csv(concession_dir / "data0_judgement.csv", index=False)
            swapped.to_csv(concession_dir / "data0_swap_judgement.csv", index=False)

            expected = [
                "usable",
                "conceded",
                "partial",
                "missing",
                "unparseable",
                "debate-incomplete",
                "debate-string-false",
            ]
            concession = pd.DataFrame(concession_rows(family_dir, expected))
            statuses = concession.set_index("case_id")["qc_status"].to_dict()
            self.assertEqual(statuses["usable"], "usable_non_conceded")
            self.assertEqual(statuses["conceded"], "conceded")
            self.assertEqual(statuses["partial"], "partial_concession_judgement")
            self.assertEqual(statuses["missing"], "missing_concession_judgement")
            self.assertEqual(
                statuses["unparseable"], "unparseable_concession_judgement"
            )
            self.assertNotIn("stale", statuses)

            debate = pd.DataFrame(
                [
                    {"id": case_id, "complete": True, "transcript_parsed": True}
                    for case_id in expected[:-2]
                ]
                + [
                    {
                        "id": "debate-incomplete",
                        "complete": False,
                        "transcript_parsed": True,
                    },
                    {
                        "id": "debate-string-false",
                        "complete": "False",
                        "transcript_parsed": "True",
                    },
                ]
            )
            outcomes = pd.DataFrame(debate_outcome_rows(debate, concession))
            counts = outcomes.set_index("status")["count"].to_dict()
            self.assertEqual(counts["usable_non_conceded"], 1)
            self.assertEqual(counts["conceded"], 1)
            self.assertEqual(counts["partial_concession_judgement"], 1)
            self.assertEqual(counts["complete_missing_concession_judgement"], 1)
            self.assertEqual(counts["unparseable_concession_judgement"], 1)
            self.assertEqual(counts["incomplete_or_unparsed"], 2)
            self.assertEqual(sum(counts.values()), len(debate))

            plot_path = Path(tmp) / "debate_outcomes.png"
            plot_debate_outcomes(outcomes, "openai", plot_path)
            self.assertTrue(plot_path.exists())

    def test_debate_outcomes_categorize_blank_and_duplicate_ids(self) -> None:
        debate = pd.DataFrame(
            [
                {"id": "", "complete": True, "transcript_parsed": True},
                {"id": "duplicate", "complete": True, "transcript_parsed": True},
                {"id": "duplicate", "complete": True, "transcript_parsed": True},
            ]
        )
        outcomes = pd.DataFrame(debate_outcome_rows(debate, None))
        counts = outcomes.set_index("status")["count"].to_dict()
        self.assertEqual(counts["malformed_debate_row"], 3)
        self.assertEqual(sum(counts.values()), 3)

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
            family_dir = root / "missing" / "openai"
            write_family_metadata(family_dir, "openai")
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
            family_dir = root / "missing" / "openai"
            write_family_metadata(family_dir, "openai")
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
                [family_dir],
                root / "missing-baselines",
                out_dir,
                False,
            )

            for path in stale_paths:
                self.assertFalse(path.exists(), path.name)


if __name__ == "__main__":
    unittest.main()
