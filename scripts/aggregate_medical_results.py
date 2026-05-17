"""Aggregate medical-debate results into headline CSVs and plots.

Reads the cached judgement CSVs under one or more exp_dirs and emits:

  exp/medical_results/
    accuracy_by_condition.csv     — accuracy + bootstrap 95% CI per (family, condition).
    pgr_by_condition.csv          — Performance Gap Recovered per (family, condition);
                                    NaN until matching baselines (B1, B2) are run.
    per_case_lift.csv             — per-case correctness, swap-averaged, joined across
                                    conditions; for diagnostic heatmaps.
    plots/01_accuracy_by_condition.png
    plots/02_pgr_by_condition.png
    plots/03_per_case_lift.png

This is pure data manipulation — no API calls. Safe to run repeatedly.

Usage:
    python scripts/aggregate_medical_results.py exp/medical_debate_n100/openai
    python scripts/aggregate_medical_results.py exp/medical_debate_n100/openai \\
                                                exp/medical_debate_n100/anthropic
    python scripts/aggregate_medical_results.py --plots-only exp/medical_debate_n100/openai
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.scoring.accuracy import find_answer


# ---------------------------------------------------------------------------
# Pulling per-case correctness out of the judgement CSVs
# ---------------------------------------------------------------------------

# subdirectories matching this pattern hold debate-stage judgements.
# Concession judging lives under `concession_*` and is excluded here.
_CONDITION_RE = re.compile(r"^(e[1-4]_[a-z_]+?)_(.+)$")


def _judge_correctness(answer_text: str, swap: bool) -> Optional[bool]:
    """Returns True/False/None based on the judge's final answer.

    `swap=False`: A is correct, B is the distractor.
    `swap=True`:  letters are inverted in the prompt, so B is correct.
    """
    if not isinstance(answer_text, str) or not answer_text.strip():
        return None
    a = find_answer(answer_text, "A")
    b = find_answer(answer_text, "B")
    if a and not b:
        return not swap  # picked A → correct iff not swapped
    if b and not a:
        return swap  # picked B → correct iff swapped
    return None  # unparseable / ambiguous


def _scan_family_dir(family_dir: Path, family_label: str) -> pd.DataFrame:
    """Walk one family's debate_sim/<condition>_<judge>/ directories and
    produce a long-format DataFrame: one row per (case, condition, swap).
    """
    debate_root = family_dir / "debate_sim"
    if not debate_root.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    for sub in sorted(debate_root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith("concession_"):
            continue
        m = _CONDITION_RE.match(sub.name)
        if not m:
            continue
        condition, judge = m.group(1), m.group(2)
        for fname, swap in (
            ("data0_judgement.csv", False),
            ("data0_swap_judgement.csv", True),
        ):
            p = sub / fname
            if not p.exists():
                continue
            df = pd.read_csv(p, keep_default_na=False)
            for _, r in df.iterrows():
                ans = r.get("answer_judge", "")
                rows.append(
                    {
                        "family": family_label,
                        "condition": condition,
                        "judge_model": judge,
                        "case_id": r.get("id"),
                        "swap": swap,
                        "correct": _judge_correctness(ans, swap),
                    }
                )
    return pd.DataFrame(rows)


def _scan_baselines(baselines_dir: Path, family_label: str) -> pd.DataFrame:
    """Walk a baselines directory and collect per-case baseline correctness.

    Expected layout:
        <baselines_dir>/baseline_blind/<judge>/data0[_swap]_judgement.csv
        <baselines_dir>/baseline_oracle/<judge>/data0[_swap]_judgement.csv

    Returns an empty DataFrame if no baselines have been run yet — the
    rest of the pipeline tolerates this and just leaves PGR as NaN.
    """
    if not baselines_dir.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    for arm in ("baseline_blind", "baseline_oracle"):
        arm_dir = baselines_dir / arm
        if not arm_dir.exists():
            continue
        for judge_dir in sorted(arm_dir.iterdir()):
            if not judge_dir.is_dir():
                continue
            for fname, swap in (
                ("data0_judgement.csv", False),
                ("data0_swap_judgement.csv", True),
            ):
                p = judge_dir / fname
                if not p.exists():
                    continue
                df = pd.read_csv(p, keep_default_na=False)
                for _, r in df.iterrows():
                    ans = r.get("answer_judge", "")
                    rows.append(
                        {
                            "family": family_label,
                            "arm": "B1" if arm == "baseline_blind" else "B2",
                            "judge_model": judge_dir.name,
                            "case_id": r.get("id"),
                            "swap": swap,
                            "correct": _judge_correctness(ans, swap),
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _swap_average(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse swap=False/True pairs into a single per-case accuracy."""
    grouped = (
        df.groupby([c for c in df.columns if c not in {"swap", "correct"}])
        .agg(acc=("correct", "mean"), n_passes=("correct", "size"))
        .reset_index()
    )
    return grouped


def _bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Returns (mean, ci_lo, ci_hi) with 95% percentile bootstrap."""
    values = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    if values.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def accuracy_table(per_case: pd.DataFrame) -> pd.DataFrame:
    """E1 — accuracy + bootstrap CIs per (family, condition, judge_model)."""
    if per_case.empty:
        return pd.DataFrame(columns=[
            "family", "condition", "judge_model", "n_cases",
            "acc_mean", "ci_lo", "ci_hi",
        ])
    out: list[dict] = []
    for (family, condition, judge), grp in per_case.groupby(
        ["family", "condition", "judge_model"]
    ):
        mean, lo, hi = _bootstrap_ci(grp["acc"].to_numpy())
        out.append({
            "family": family,
            "condition": condition,
            "judge_model": judge,
            "n_cases": int(grp["acc"].notna().sum()),
            "acc_mean": mean,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    return pd.DataFrame(out).sort_values(["family", "condition"]).reset_index(drop=True)


def baseline_table(baselines_case: pd.DataFrame) -> pd.DataFrame:
    if baselines_case.empty:
        return pd.DataFrame(columns=[
            "family", "arm", "judge_model", "n_cases",
            "acc_mean", "ci_lo", "ci_hi",
        ])
    out: list[dict] = []
    for (family, arm, judge), grp in baselines_case.groupby(
        ["family", "arm", "judge_model"]
    ):
        mean, lo, hi = _bootstrap_ci(grp["acc"].to_numpy())
        out.append({
            "family": family,
            "arm": arm,
            "judge_model": judge,
            "n_cases": int(grp["acc"].notna().sum()),
            "acc_mean": mean,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    return pd.DataFrame(out).sort_values(["family", "arm"]).reset_index(drop=True)


# Map each debate condition to which judge's baselines should be used for
# its PGR denominator. (Same model as the judge in numerator — Khan
# convention.)
_CONDITION_JUDGE_TYPE = {
    "e1_info_asymmetry": "frontier",
    "e2_double_asymmetry": "weaker",
    "e3_capability_asymmetry": "weaker",
    "e4_full_symmetry": "frontier",
}


def pgr_table(accuracy: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """E2 — PGR per (family, condition).

    PGR(condition) = (acc_condition − acc_B1_same_judge) / (acc_B2_same_judge − acc_B1_same_judge).
    Bootstrap CIs are propagated from the underlying per-case data via the
    swap-averaged means; for a quick first pass we use a normal-approximation
    interval until we have enough cases to wire a paired bootstrap.

    NaN throughout if `baselines` is empty (baselines haven't been run yet).
    """
    rows: list[dict] = []
    if accuracy.empty:
        return pd.DataFrame(rows)

    bl = baselines.set_index(["family", "arm", "judge_model"])["acc_mean"] if not baselines.empty else None

    for _, r in accuracy.iterrows():
        cond = r["condition"]
        judge = r["judge_model"]
        family = r["family"]
        pgr, lo, hi = float("nan"), float("nan"), float("nan")
        if bl is not None:
            try:
                b1 = bl.loc[(family, "B1", judge)]
                b2 = bl.loc[(family, "B2", judge)]
                if b2 - b1 > 0:
                    pgr = (r["acc_mean"] - b1) / (b2 - b1)
                    # Crude CI: propagate the symmetric half-width on acc_mean
                    half = (r["ci_hi"] - r["ci_lo"]) / 2
                    if b2 - b1 > 0:
                        lo = pgr - half / (b2 - b1)
                        hi = pgr + half / (b2 - b1)
            except KeyError:
                pass
        rows.append({
            "family": family,
            "condition": cond,
            "judge_model": judge,
            "pgr": pgr,
            "pgr_ci_lo": lo,
            "pgr_ci_hi": hi,
        })
    return pd.DataFrame(rows).sort_values(["family", "condition"]).reset_index(drop=True)


def per_case_lift_table(per_case: pd.DataFrame, baselines_case: pd.DataFrame) -> pd.DataFrame:
    """E3 — per case, did debate flip the answer vs baseline?

    Joins each per-case (debate) accuracy against the same-judge B1 baseline
    accuracy. Emits one row per (case, condition, family). For McNemar-style
    2x2 contingency tables the consumer can pivot on (debate>=0.5, baseline>=0.5).
    """
    if per_case.empty or baselines_case.empty:
        return pd.DataFrame()
    blind = (
        baselines_case[baselines_case["arm"] == "B1"]
        .groupby(["family", "judge_model", "case_id"])["acc"]
        .mean()
        .rename("blind_acc")
        .reset_index()
    )
    merged = per_case.merge(blind, on=["family", "judge_model", "case_id"], how="left")
    merged = merged.rename(columns={"acc": "debate_acc"})
    return merged[[
        "family", "condition", "judge_model", "case_id", "debate_acc", "blind_acc",
    ]]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


_COND_ORDER = ["e1_info_asymmetry", "e2_double_asymmetry", "e3_capability_asymmetry", "e4_full_symmetry"]
_COND_LABELS = {
    "e1_info_asymmetry": "E1\ninfo asym",
    "e2_double_asymmetry": "E2\ndouble asym",
    "e3_capability_asymmetry": "E3\ncap asym",
    "e4_full_symmetry": "E4\nsymmetry",
}


def plot_accuracy(accuracy: pd.DataFrame, baselines: pd.DataFrame, out_path: Path) -> None:
    if accuracy.empty:
        return
    families = sorted(accuracy["family"].unique())
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.35
    x_labels: list[str] = []
    x_positions = []

    # Build x ordering: B1, B2, E1..E4
    base_x = list(range(6))
    label_keys = ["B1", "B2"] + _COND_ORDER
    label_display = ["B1\nblind", "B2\noracle"] + [_COND_LABELS[c] for c in _COND_ORDER]

    for fi, family in enumerate(families):
        offset = (fi - (len(families) - 1) / 2) * width
        heights, lo, hi = [], [], []
        for k in label_keys:
            if k in ("B1", "B2"):
                rows = baselines[(baselines["family"] == family) & (baselines["arm"] == k)]
            else:
                rows = accuracy[(accuracy["family"] == family) & (accuracy["condition"] == k)]
            if rows.empty:
                heights.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            else:
                heights.append(rows["acc_mean"].mean())
                lo.append(rows["ci_lo"].mean())
                hi.append(rows["ci_hi"].mean())
        xs = [b + offset for b in base_x]
        yerr = [[h - l if not np.isnan(h) and not np.isnan(l) else 0 for h, l in zip(heights, lo)],
                [u - h if not np.isnan(h) and not np.isnan(u) else 0 for h, u in zip(heights, hi)]]
        ax.bar(xs, heights, width=width, label=family, yerr=yerr, capsize=3)

    ax.set_xticks(base_x)
    ax.set_xticklabels(label_display)
    ax.set_ylabel("Judge accuracy")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="chance")
    ax.set_title("Per-condition judge accuracy (bootstrap 95% CI)")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pgr(pgr: pd.DataFrame, out_path: Path) -> None:
    if pgr.empty or pgr["pgr"].isna().all():
        return
    families = sorted(pgr["family"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    base_x = list(range(4))
    for fi, family in enumerate(families):
        offset = (fi - (len(families) - 1) / 2) * width
        ys = []
        for c in _COND_ORDER:
            rows = pgr[(pgr["family"] == family) & (pgr["condition"] == c)]
            ys.append(rows["pgr"].mean() if not rows.empty else np.nan)
        ax.bar([b + offset for b in base_x], ys, width=width, label=family)
    ax.axhline(0, color="grey", linewidth=1)
    ax.axhline(1, color="grey", linestyle=":", alpha=0.6, label="full recovery")
    ax.set_xticks(base_x)
    ax.set_xticklabels([_COND_LABELS[c] for c in _COND_ORDER])
    ax.set_ylabel("Performance Gap Recovered")
    ax.set_title("PGR per condition (relative to same-judge B1/B2 baselines)")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_case_lift(lift: pd.DataFrame, out_path: Path) -> None:
    if lift.empty:
        return
    fig, axes = plt.subplots(
        1, max(1, len(_COND_ORDER)),
        figsize=(3 * len(_COND_ORDER), 3.5),
        squeeze=False,
    )
    for i, cond in enumerate(_COND_ORDER):
        ax = axes[0][i]
        sub = lift[lift["condition"] == cond]
        if sub.empty or sub["blind_acc"].isna().all():
            ax.set_title(_COND_LABELS[cond])
            ax.text(0.5, 0.5, "no baseline data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        # 2x2 contingency on >=0.5 thresholds
        sub = sub.copy()
        sub["blind_correct"] = sub["blind_acc"] >= 0.5
        sub["debate_correct"] = sub["debate_acc"] >= 0.5
        ct = pd.crosstab(sub["blind_correct"], sub["debate_correct"], dropna=False)
        for r in [False, True]:
            for c in [False, True]:
                if r not in ct.index: ct.loc[r] = 0
                if c not in ct.columns: ct[c] = 0
        ct = ct.reindex(index=[False, True], columns=[False, True], fill_value=0)
        im = ax.imshow(ct.values, cmap="Blues")
        for ri in range(2):
            for ci in range(2):
                ax.text(ci, ri, ct.values[ri, ci], ha="center", va="center", color="black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["debate wrong", "debate right"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["blind wrong", "blind right"])
        ax.set_title(_COND_LABELS[cond])
    fig.suptitle("Per-case lift: debate vs blind baseline (2×2 per condition)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _family_label_from_path(family_dir: Path) -> str:
    """Best-effort: use the leaf directory name (e.g. `openai`, `anthropic`)."""
    return family_dir.name


def main(family_dirs: list[Path], baselines_dir: Optional[Path], out_dir: Path, plots_only: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    per_case_all = pd.concat(
        [_scan_family_dir(d, _family_label_from_path(d)) for d in family_dirs],
        ignore_index=True,
    ) if family_dirs else pd.DataFrame()

    baselines_per_case = (
        _scan_baselines(baselines_dir, _family_label_from_path(family_dirs[0]))
        if (baselines_dir and family_dirs)
        else pd.DataFrame()
    )

    per_case_agg = _swap_average(per_case_all) if not per_case_all.empty else pd.DataFrame()
    baselines_case_agg = _swap_average(baselines_per_case) if not baselines_per_case.empty else pd.DataFrame()

    acc = accuracy_table(per_case_agg)
    bls = baseline_table(baselines_case_agg)
    pgr = pgr_table(acc, bls)
    lift = per_case_lift_table(per_case_agg, baselines_case_agg)

    if not plots_only:
        acc.to_csv(out_dir / "accuracy_by_condition.csv", index=False)
        pgr.to_csv(out_dir / "pgr_by_condition.csv", index=False)
        lift.to_csv(out_dir / "per_case_lift.csv", index=False)
        if not bls.empty:
            bls.to_csv(out_dir / "baselines_by_arm.csv", index=False)

    plot_accuracy(acc, bls, out_dir / "plots" / "01_accuracy_by_condition.png")
    plot_pgr(pgr, out_dir / "plots" / "02_pgr_by_condition.png")
    plot_per_case_lift(lift, out_dir / "plots" / "03_per_case_lift.png")

    print(f"wrote results to {out_dir}/")
    print(f"  accuracy_by_condition.csv  ({len(acc)} rows)")
    print(f"  pgr_by_condition.csv       ({len(pgr)} rows)")
    print(f"  per_case_lift.csv          ({len(lift)} rows)")
    if not bls.empty:
        print(f"  baselines_by_arm.csv       ({len(bls)} rows)")
    else:
        print(f"  baselines_by_arm.csv       (skipped — no baseline data found)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "family_dirs",
        nargs="+",
        type=Path,
        help="One or more family directories, e.g. exp/medical_debate_n100/openai",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=None,
        help="Optional baselines directory (per-judge subdirs).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("exp/medical_results"),
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots from cached CSVs without re-walking exp dirs.",
    )
    args = parser.parse_args()
    main(args.family_dirs, args.baselines_dir, args.out_dir, args.plots_only)
