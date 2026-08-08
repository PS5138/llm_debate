"""Build the curated-bundle statistics CSVs from cached judgement output.

Emits the three files that used to be assembled by hand for
`results/<family>_100/`, which meant the headline significance claim in the
README rested on numbers no code in the repo could regenerate:

  headline_results.csv         — one row per condition: same-judge blind /
                                 debate / oracle accuracy with bootstrap CIs,
                                 plus PGR with a case-paired bootstrap CI.
  paired_tests.csv             — paired debate-vs-blind and debate-vs-oracle
                                 tests: mean difference, bootstrap CI, paired
                                 t-test, sign-flip permutation p, and
                                 Holm-adjusted p-values.
  quality_control_summary.csv  — rollup of the `analysis/` QC tables (quote
                                 verification, verbosity, debate outcomes,
                                 position bias) into one table.

Per-case scores come from `aggregate_medical_results.load_per_case_scores`, so
they are the same swap-averaged values behind the headline accuracies. All
statistics live in `medical_stats`, shared with the aggregator. No API calls —
safe to run repeatedly on cached output.

Usage:
    python scripts/compute_paired_tests.py exp/YYYY-MM-DD_HH-MM-SS_results/openai \\
        --baselines-dir data/canonical_baselines \\
        --out-dir exp/YYYY-MM-DD_HH-MM-SS_results/medical_results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import medical_stats
from scripts.aggregate_medical_results import (
    _COND_ORDER,
    _family_label_from_path,
    aligned_case_scores,
    load_per_case_scores,
)

_CONDITION_LABELS = {
    "e1_info_asymmetry": "E1 info asymmetry",
    "e2_double_asymmetry": "E2 double asymmetry",
    "e3_capability_asymmetry": "E3 capability asymmetry",
    "e4_full_symmetry": "E4 full symmetry",
}


# ---------------------------------------------------------------------------
# Headline + paired tests
# ---------------------------------------------------------------------------


def build_tables(
    per_case: pd.DataFrame,
    baselines_case: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (headline_results, paired_tests).

    Holm correction is applied per family over that family's whole set of
    tests, and separately within each comparison type. With the standard four
    conditions those are the 8- and 4-test families the column names describe.
    """
    if per_case.empty or baselines_case.empty:
        return pd.DataFrame(), pd.DataFrame()

    headline_rows: list[dict] = []
    test_rows: list[dict] = []

    for family in sorted(per_case["family"].unique()):
        fam = per_case[per_case["family"] == family]
        conditions = [c for c in _COND_ORDER if c in set(fam["condition"])]
        for condition in conditions:
            judges = sorted(fam[fam["condition"] == condition]["judge_model"].unique())
            for judge in judges:
                arms = aligned_case_scores(
                    per_case, baselines_case, family, condition, judge
                )
                if arms is None:
                    continue
                debate, blind, oracle = arms
                label = _CONDITION_LABELS.get(condition, condition)
                n_cases = int(debate.size)

                d_acc, d_lo, d_hi = medical_stats.mean_ci(debate)
                b_acc, b_lo, b_hi = medical_stats.mean_ci(blind)
                o_acc, o_lo, o_hi = medical_stats.mean_ci(oracle)
                pgr_val, pgr_lo, pgr_hi = medical_stats.pgr_ci(debate, blind, oracle)

                headline_rows.append(
                    {
                        "family": family,
                        "condition": condition,
                        "label": label,
                        "judge_model": judge,
                        "n_cases": n_cases,
                        "blind_acc": b_acc,
                        "blind_ci_lo": b_lo,
                        "blind_ci_hi": b_hi,
                        "debate_acc": d_acc,
                        "debate_ci_lo": d_lo,
                        "debate_ci_hi": d_hi,
                        "oracle_acc": o_acc,
                        "oracle_ci_lo": o_lo,
                        "oracle_ci_hi": o_hi,
                        "pgr": pgr_val,
                        "pgr_ci_lo": pgr_lo,
                        "pgr_ci_hi": pgr_hi,
                    }
                )

                for comparison, reference in (
                    ("debate_vs_blind", blind),
                    ("debate_vs_oracle", oracle),
                ):
                    mean_diff, ci_lo, ci_hi = medical_stats.paired_mean_diff_ci(
                        debate, reference
                    )
                    t_stat, raw_p = medical_stats.paired_t_test(debate, reference)
                    perm_p = medical_stats.paired_permutation_p(debate, reference)
                    diffs = debate - reference
                    test_rows.append(
                        {
                            "family": family,
                            "condition": condition,
                            "label": label,
                            "judge_model": judge,
                            "comparison": comparison,
                            "n_cases": n_cases,
                            "mean_diff": mean_diff,
                            "bootstrap_ci_lo": ci_lo,
                            "bootstrap_ci_hi": ci_hi,
                            "t_statistic": t_stat,
                            "raw_p": raw_p,
                            "paired_permutation_p_two_sided": perm_p,
                            "nonzero_pairs": int((diffs != 0).sum()),
                        }
                    )

    headline = pd.DataFrame(headline_rows)
    tests = pd.DataFrame(test_rows)
    if tests.empty:
        return headline, tests

    # Holm across every test in the family, then within each comparison type.
    tests["holm_p_all_8_tests"] = float("nan")
    tests["holm_p_within_4_tests"] = float("nan")
    for family, grp in tests.groupby("family"):
        idx = grp.index.tolist()
        tests.loc[idx, "holm_p_all_8_tests"] = medical_stats.holm(grp["raw_p"].tolist())
        for _, sub in grp.groupby("comparison"):
            sub_idx = sub.index.tolist()
            tests.loc[sub_idx, "holm_p_within_4_tests"] = medical_stats.holm(
                sub["raw_p"].tolist()
            )

    return headline, tests


# ---------------------------------------------------------------------------
# Quality-control rollup
# ---------------------------------------------------------------------------


def build_quality_control(analysis_dir: Path) -> pd.DataFrame:
    """Consolidate the `analysis/` QC tables into one long table.

    Every number here already exists in `analysis/`; this only gathers the
    ones the write-ups quote so a reader does not have to open four files.
    Missing inputs are skipped rather than fatal — a partial run still gets a
    partial QC table.
    """
    rows: list[dict] = []

    quotes = analysis_dir / "quote_verification.csv"
    if quotes.exists():
        df = pd.read_csv(quotes)
        for side, grp in df.groupby("side"):
            total = int(grp["n_quotes"].sum())
            verified = int(grp["n_verified"].sum())
            rows.append(
                {
                    "metric": f"quote_verification_{side}",
                    "value": (verified / total) if total else float("nan"),
                    "numerator": verified,
                    "denominator": total,
                    "notes": "verified quote tags / total quote tags",
                }
            )

    verbosity = analysis_dir / "verbosity.csv"
    if verbosity.exists():
        df = pd.read_csv(verbosity)
        for side, grp in df.groupby("side"):
            rows.append(
                {
                    "metric": f"avg_words_per_turn_{side}",
                    "value": float(grp["words"].mean()),
                    "numerator": "",
                    "denominator": "",
                    "notes": "mean words per debater turn across 3 rounds",
                }
            )

    outcomes = analysis_dir / "debate_outcomes.csv"
    if outcomes.exists():
        df = pd.read_csv(outcomes)
        for _, r in df.iterrows():
            rows.append(
                {
                    "metric": f"debate_outcome_{r['status']}",
                    "value": float(r["count"]),
                    "numerator": "",
                    "denominator": "",
                    "notes": "count of cases",
                }
            )

    bias = analysis_dir / "position_bias.csv"
    if bias.exists():
        df = pd.read_csv(bias)
        for _, r in df.iterrows():
            rows.append(
                {
                    "metric": f"position_bias_pct_{r['stage']}_{r['arm']}_{r['judge_model']}",
                    "value": float(r["pos_bias_total_pct"]),
                    "numerator": int(r["pos_bias_total"]),
                    "denominator": int(r["n"]),
                    "notes": "position-biased cases / cases",
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(
    family_dirs: list[Path],
    baselines_dir: Optional[Path],
    out_dir: Path,
    analysis_dir: Optional[Path],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    per_case, baselines_case = load_per_case_scores(family_dirs, baselines_dir)
    if per_case.empty:
        raise SystemExit(
            "no debate judgement CSVs found — run the debate + judging stages first"
        )
    if baselines_case.empty:
        raise SystemExit(
            "no baseline judgement CSVs found — pass --baselines-dir pointing at "
            "a baselines folder (or data/canonical_baselines)"
        )

    headline, tests = build_tables(per_case, baselines_case)
    if headline.empty:
        raise SystemExit(
            "no condition had all three of debate / blind / oracle for one judge; "
            "check that the baselines cover the judge models used by the debate arms"
        )

    headline.to_csv(out_dir / "headline_results.csv", index=False)
    tests.to_csv(out_dir / "paired_tests.csv", index=False)
    print(f"wrote {out_dir}/headline_results.csv  ({len(headline)} rows)")
    print(f"wrote {out_dir}/paired_tests.csv      ({len(tests)} rows)")

    resolved_analysis = analysis_dir
    if resolved_analysis is None and len(family_dirs) == 1:
        candidate = family_dirs[0] / "analysis"
        resolved_analysis = candidate if candidate.exists() else None
    if resolved_analysis is not None:
        qc = build_quality_control(resolved_analysis)
        if qc.empty:
            print(f"no QC inputs found under {resolved_analysis}/ — skipped")
        else:
            qc.to_csv(out_dir / "quality_control_summary.csv", index=False)
            print(f"wrote {out_dir}/quality_control_summary.csv ({len(qc)} rows)")
    else:
        print("quality_control_summary.csv skipped — pass --analysis-dir")

    print()
    print("Headline (accuracy / PGR):")
    show = headline[
        ["label", "judge_model", "blind_acc", "debate_acc", "oracle_acc", "pgr"]
    ]
    print(show.to_string(index=False))
    print()
    print("Paired tests:")
    show = tests[
        [
            "label",
            "comparison",
            "mean_diff",
            "paired_permutation_p_two_sided",
            "holm_p_all_8_tests",
        ]
    ]
    print(show.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "family_dirs",
        nargs="+",
        type=Path,
        help="One or more family directories, e.g. exp/YYYY-MM-DD_HH-MM-SS_results/openai",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=None,
        help="Baselines directory (per-judge subdirs), or a parent with family subdirs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("exp/medical_results"),
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Analysis directory for the QC rollup. Defaults to "
        "<family_dir>/analysis when exactly one family dir is given.",
    )
    args = parser.parse_args()
    main(args.family_dirs, args.baselines_dir, args.out_dir, args.analysis_dir)
