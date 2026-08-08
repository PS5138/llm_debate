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

import numpy as np
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
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (headline_results, paired_tests, skipped).

    `skipped` names every (family, condition, judge) that could not be scored
    because it had no same-judge blind and oracle baseline. Callers are
    expected to surface it: a partial run that silently drops half its
    conditions looks indistinguishable from a complete one.

    Holm correction is applied per family over that family's whole set of
    tests, and separately within each comparison type. Both the t-test and the
    permutation p-values are corrected, in separate columns, and the number of
    tests in each correction family is recorded alongside — the counts are
    data rather than something to infer from a column name.
    """
    if per_case.empty or baselines_case.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    headline_rows: list[dict] = []
    test_rows: list[dict] = []
    skipped: list[str] = []

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
                    skipped.append(f"{family}/{condition} (judge {judge})")
                    continue
                debate, blind, oracle = arms
                label = _CONDITION_LABELS.get(condition, condition)
                # Count cases that are actually usable in all three arms. A
                # case whose judging passes did not both parse is NaN and is
                # dropped from every headline statistic below, so reporting
                # the raw array length would overstate what the numbers rest
                # on. Keep the three displayed arm means and PGR on this same
                # cohort; otherwise the displayed means need not reconstruct
                # the displayed PGR.
                usable = ~(np.isnan(debate) | np.isnan(blind) | np.isnan(oracle))
                n_cases = int(usable.sum())
                headline_debate = debate[usable]
                headline_blind = blind[usable]
                headline_oracle = oracle[usable]

                d_acc, d_lo, d_hi = medical_stats.mean_ci(headline_debate)
                b_acc, b_lo, b_hi = medical_stats.mean_ci(headline_blind)
                o_acc, o_lo, o_hi = medical_stats.mean_ci(headline_oracle)
                pgr_val, pgr_lo, pgr_hi = medical_stats.pgr_ci(
                    headline_debate, headline_blind, headline_oracle
                )

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
                    t_stat, p_ttest = medical_stats.paired_t_test(debate, reference)
                    p_perm = medical_stats.paired_permutation_p(debate, reference)
                    diffs = debate - reference
                    # NaN != 0 evaluates True, so unusable pairs would be
                    # counted as genuine disagreements. Restrict to pairs
                    # where both arms scored the case.
                    paired = diffs[~np.isnan(diffs)]
                    test_rows.append(
                        {
                            "family": family,
                            "condition": condition,
                            "label": label,
                            "judge_model": judge,
                            "comparison": comparison,
                            "n_cases": int(paired.size),
                            "mean_diff": mean_diff,
                            "bootstrap_ci_lo": ci_lo,
                            "bootstrap_ci_hi": ci_hi,
                            "nonzero_pairs": int((paired != 0).sum()),
                            "t_statistic": t_stat,
                            "p_ttest": p_ttest,
                            "p_permutation": p_perm,
                        }
                    )

    headline = pd.DataFrame(headline_rows)
    tests = pd.DataFrame(test_rows)
    if tests.empty:
        return headline, tests, skipped

    # Holm across every test in the family, then within each comparison type.
    # Both p-values are corrected: the permutation test is the one the
    # methodology says carries the claim, so leaving it unadjusted while
    # publishing an adjusted t-test p invites the reader to attribute the
    # correction to the wrong test.
    for col in (
        "holm_p_ttest_all",
        "holm_p_ttest_within",
        "holm_p_permutation_all",
        "holm_p_permutation_within",
        "n_tests_all",
        "n_tests_within",
    ):
        tests[col] = float("nan")

    for _, grp in tests.groupby("family"):
        idx = grp.index.tolist()
        tests.loc[idx, "holm_p_ttest_all"] = medical_stats.holm(grp["p_ttest"].tolist())
        tests.loc[idx, "holm_p_permutation_all"] = medical_stats.holm(
            grp["p_permutation"].tolist()
        )
        tests.loc[idx, "n_tests_all"] = len(idx)
        for _, sub in grp.groupby("comparison"):
            sub_idx = sub.index.tolist()
            tests.loc[sub_idx, "holm_p_ttest_within"] = medical_stats.holm(
                sub["p_ttest"].tolist()
            )
            tests.loc[sub_idx, "holm_p_permutation_within"] = medical_stats.holm(
                sub["p_permutation"].tolist()
            )
            tests.loc[sub_idx, "n_tests_within"] = len(sub_idx)

    for col in ("n_tests_all", "n_tests_within"):
        tests[col] = tests[col].astype(int)

    return headline, tests, skipped


# ---------------------------------------------------------------------------
# Quality-control rollup
# ---------------------------------------------------------------------------


def build_quality_control(analysis_dir: Path, family: str) -> pd.DataFrame:
    """Consolidate one family's `analysis/` QC tables into one long table.

    Every number here already exists in `analysis/`; this only gathers the
    ones the write-ups quote so a reader does not have to open four files.
    Missing inputs are skipped rather than fatal — a partial run still gets a
    partial QC table.

    Rows carry `family` because a shared `medical_results/` directory holds
    every family in the run. Without it a second family's pass would overwrite
    the first's QC summary rather than extend it.
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
                    "family": family,
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
                    "family": family,
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
                    "family": family,
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
                    "family": family,
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


def _remove_stale_output(target: Path, reason: str) -> None:
    """Remove one output that this run cannot regenerate."""
    if target.exists():
        target.unlink()
        print(f"removed stale {target} ({reason})", file=sys.stderr)


def _clear_stale_outputs(out_dir: Path, reason: str) -> None:
    """Delete result files this run cannot regenerate.

    This stage is non-fatal, and `--out-dir` is reused across a resume or
    extend. Leaving a previous run's `headline_results.csv` in place while
    reporting success would present stale numbers as current ones, and the
    Streamlit runner only inspects the exit code. Removing them makes the
    failure visible instead of silently authoritative.
    """
    for name in ("headline_results.csv", "paired_tests.csv"):
        _remove_stale_output(out_dir / name, reason)


def _write_quality_control(
    family_dirs: list[Path],
    out_dir: Path,
    analysis_dir: Optional[Path],
) -> None:
    """Build the QC rollup for every family in the run, in one table.

    Each family's analysis directory is derived from its own family dir, so a
    two-family run gets both. `--analysis-dir` overrides that, but only when a
    single family dir was given: pointing every family at one analysis folder
    would silently mislabel the rows.
    """
    frames: list[pd.DataFrame] = []
    for family_dir in family_dirs:
        family = _family_label_from_path(family_dir)
        if analysis_dir is not None and len(family_dirs) == 1:
            candidate = analysis_dir
        else:
            candidate = family_dir / "analysis"
        if not candidate.exists():
            print(f"no analysis dir for {family} at {candidate} — QC skipped")
            continue
        qc = build_quality_control(candidate, family)
        if qc.empty:
            print(f"no QC inputs under {candidate}/ — QC skipped for {family}")
        else:
            frames.append(qc)

    if not frames:
        print("quality_control_summary.csv skipped — no QC inputs found")
        _remove_stale_output(out_dir / "quality_control_summary.csv", "no QC inputs")
        return
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / "quality_control_summary.csv", index=False)
    families = ", ".join(sorted(combined["family"].unique()))
    print(
        f"wrote {out_dir}/quality_control_summary.csv "
        f"({len(combined)} rows; families: {families})"
    )


def main(
    family_dirs: list[Path],
    baselines_dir: Optional[Path],
    out_dir: Path,
    analysis_dir: Optional[Path],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    per_case, baselines_case = load_per_case_scores(family_dirs, baselines_dir)

    # These stages are non-fatal by design. This script runs late in
    # `run_medical_debate.sh`, after the debate and judging stages have already
    # been paid for, and `set -e` means a nonzero exit here would skip the cost
    # summary and the transcript export. The neighbouring aggregation stage
    # tolerates the same missing inputs, so this one matches that contract:
    # explain loudly, write what can be written, and leave the pipeline running.
    if per_case.empty:
        print(
            "WARNING: no debate judgement CSVs found — run the debate and "
            "judging stages first. Skipping paired tests.",
            file=sys.stderr,
        )
        _clear_stale_outputs(out_dir, "no debate judgements")
        _remove_stale_output(
            out_dir / "quality_control_summary.csv", "no debate judgements"
        )
        return
    if baselines_case.empty:
        print(
            "WARNING: no baseline judgement CSVs found under the given "
            "--baselines-dir, so PGR and the paired tests cannot be computed. "
            "Point --baselines-dir at a baselines folder (or "
            "data/canonical_baselines). Skipping paired tests.",
            file=sys.stderr,
        )
        _clear_stale_outputs(out_dir, "no baselines")
        _write_quality_control(family_dirs, out_dir, analysis_dir)
        return

    headline, tests, skipped = build_tables(per_case, baselines_case)

    if skipped:
        print(
            "WARNING: no same-judge blind+oracle baseline for "
            f"{len(skipped)} condition(s), so they are absent from the tables "
            "below. This run is INCOMPLETE:",
            file=sys.stderr,
        )
        for item in skipped:
            print(f"  - {item}", file=sys.stderr)

    if headline.empty:
        print(
            "WARNING: no condition had all three of debate / blind / oracle "
            "for one judge — check that the baselines cover the judge models "
            "used by the debate arms. Skipping paired tests.",
            file=sys.stderr,
        )
        _clear_stale_outputs(out_dir, "no scoreable condition")
        _write_quality_control(family_dirs, out_dir, analysis_dir)
        return

    headline.to_csv(out_dir / "headline_results.csv", index=False)
    tests.to_csv(out_dir / "paired_tests.csv", index=False)
    print(f"wrote {out_dir}/headline_results.csv  ({len(headline)} rows)")
    print(f"wrote {out_dir}/paired_tests.csv      ({len(tests)} rows)")

    _write_quality_control(family_dirs, out_dir, analysis_dir)

    print()
    print("Headline (accuracy / PGR):")
    show = headline[
        ["label", "judge_model", "blind_acc", "debate_acc", "oracle_acc", "pgr"]
    ]
    print(show.to_string(index=False))
    print()
    print("Paired tests (Holm across all tests in the family):")
    show = tests[
        [
            "label",
            "comparison",
            "mean_diff",
            "p_permutation",
            "holm_p_permutation_all",
            "holm_p_ttest_all",
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
