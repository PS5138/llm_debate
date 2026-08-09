"""Aggregate medical-debate results into headline CSVs and plots.

Reads the cached judgement CSVs under one or more exp_dirs and emits:

  exp/medical_results/
    accuracy_by_condition.csv     — accuracy + bootstrap 95% CI per (family, condition).
    (PGR is computed here to draw plot 02, but is written out by
     compute_paired_tests.py as part of headline_results.csv.)
    per_case_lift.csv             — per-case correctness, swap-averaged, joined across
                                    conditions; for diagnostic heatmaps.
    plots/01_accuracy_by_condition.png
    plots/02_pgr_by_condition.png
    plots/03_per_case_lift.png

This is pure data manipulation — no API calls. Safe to run repeatedly.

Usage:
    python scripts/aggregate_medical_results.py exp/YYYY-MM-DD_HH-MM-SS_results/openai
    python scripts/aggregate_medical_results.py exp/YYYY-MM-DD_HH-MM-SS_results/openai \\
                                                exp/YYYY-MM-DD_HH-MM-SS_results/anthropic \\
                                                --baselines-dir exp/YYYY-MM-DD_HH-MM-SS_results/baselines
    python scripts/aggregate_medical_results.py --plots-only exp/YYYY-MM-DD_HH-MM-SS_results/openai
"""

from __future__ import annotations

import argparse
import json
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
from scripts import medical_stats

# ---------------------------------------------------------------------------
# Pulling per-case correctness out of the judgement CSVs
# ---------------------------------------------------------------------------

# Debate-stage judgement subdirectories are named `<condition>_<judge_model>`,
# e.g. `e2_double_asymmetry_claude-sonnet-4-6`. The naive regex
# `^(e[1-4]_[a-z_]+?)_(.+)$` is too greedy/non-greedy and splits in the
# middle of the condition name. Match against the fixed condition prefixes
# instead so `claude-opus-4-6` and `gpt-5.4-mini` round-trip cleanly as the
# judge model.
_KNOWN_CONDITIONS = (
    "e1_info_asymmetry",
    "e2_double_asymmetry",
    "e3_capability_asymmetry",
    "e4_full_symmetry",
)

# This repository intentionally exposes one fixed experiment definition per
# provider family.  Analysis must use the models recorded by the run, but those
# recorded values must themselves match the supported fixed trio.  Otherwise a
# stale judgement directory (for example, one left behind after a local model
# experiment) could silently enter the published tables.
_FIXED_MODEL_TRIOS = {
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

_CONDITION_MODEL_ROLE = {
    "e1_info_asymmetry": "frontier_model",
    "e2_double_asymmetry": "weaker_model",
    "e3_capability_asymmetry": "weaker_model",
    "e4_full_symmetry": "frontier_model",
}


def _load_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid run metadata at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid run metadata at {path}: expected a JSON object")
    return value


def _select_root_family_record(root_doc: dict, family_dir: Path) -> Optional[dict]:
    """Select one family record from app-style root metadata.

    Current runs also write authoritative metadata directly inside the family
    directory.  This root-record path exists for older app runs whose metadata
    only lived at the timestamped run root.
    """
    families = root_doc.get("families")
    if not isinstance(families, dict):
        return None
    leaf = family_dir.name
    candidate = families.get(leaf)
    if isinstance(candidate, dict):
        return candidate
    return None


def _family_run_spec(family_dir: Path) -> dict[str, str]:
    """Return validated, authoritative model metadata for one family dir.

    Supported metadata shapes are:

    * ``<family>/run_metadata.json`` with the family record directly (the
      current app format and the curated ``results/openai_100`` format); or
    * ``<run>/run_metadata.json`` with a ``families`` mapping (older app runs).

    Missing or contradictory metadata fails closed.  Inferring models from
    whatever directory names happen to exist would make stale caches look like
    current experiment output.
    """
    family_meta = family_dir / "run_metadata.json"
    if family_meta.exists():
        record = _load_json_object(family_meta)
        if "families" in record and "family" not in record:
            selected = _select_root_family_record(record, family_dir)
            if selected is None:
                raise ValueError(
                    f"run metadata at {family_meta} has no record for "
                    f"family directory {family_dir.name!r}"
                )
            record = selected
    else:
        root_meta = family_dir.parent / "run_metadata.json"
        if not root_meta.exists():
            raise ValueError(
                f"missing run metadata for {family_dir}; analysis refuses to "
                "infer a model trio from cached directory names"
            )
        root_doc = _load_json_object(root_meta)
        selected = _select_root_family_record(root_doc, family_dir)
        if selected is not None:
            record = selected
        elif root_doc.get("family") == family_dir.name:
            record = root_doc
        else:
            raise ValueError(
                f"run metadata at {root_meta} has no record for "
                f"family directory {family_dir.name!r}"
            )

    family = record.get("family")
    if not isinstance(family, str) or family not in _FIXED_MODEL_TRIOS:
        raise ValueError(
            f"unsupported or missing family in run metadata for {family_dir}: {family!r}"
        )
    expected = _FIXED_MODEL_TRIOS[family]
    validated: dict[str, str] = {"family": family}
    for key, expected_model in expected.items():
        recorded = record.get(key)
        if not isinstance(recorded, str) or recorded.strip() != expected_model:
            raise ValueError(
                f"run metadata for {family_dir} records {key}={recorded!r}; "
                f"the fixed {family} experiment requires {expected_model!r}"
            )
        validated[key] = expected_model
    return validated


def _split_condition_dir(name: str) -> Optional[tuple[str, str]]:
    """Return (condition, judge_model) or None if the dir name isn't an
    E1-E4 final-judge directory.
    """
    for cond in _KNOWN_CONDITIONS:
        prefix = cond + "_"
        if name.startswith(prefix):
            judge = name[len(prefix) :]
            if judge:
                return cond, judge
    return None


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


def _is_explicitly_complete(value: object) -> bool:
    """Return true only for an explicit CSV boolean ``True`` value.

    A partially written judgement may already contain answer-like text while
    ``complete_judge`` is false. Such text is diagnostic state, not a scored
    model decision. Pandas can load mixed boolean columns as strings, so the
    exact case-insensitive string ``"true"`` is accepted as well.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return isinstance(value, str) and value.strip().lower() == "true"


def _scan_family_dir(family_dir: Path, spec: dict[str, str]) -> pd.DataFrame:
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
        parsed = _split_condition_dir(sub.name)
        if parsed is None:
            continue
        condition, judge = parsed
        expected_judge = spec[_CONDITION_MODEL_ROLE[condition]]
        if judge != expected_judge:
            # A model-looking directory is not evidence that it belongs to the
            # recorded run. Ignore stale/local variants deterministically.
            continue
        for fname, swap in (
            ("data0_judgement.csv", False),
            ("data0_swap_judgement.csv", True),
        ):
            p = sub / fname
            if not p.exists():
                continue
            df = pd.read_csv(p, keep_default_na=False)
            for _, r in df.iterrows():
                if not _is_explicitly_complete(r.get("complete_judge")):
                    continue
                ans = r.get("answer_judge", "")
                correct = _judge_correctness(ans, swap)
                if correct is None:
                    continue
                rows.append(
                    {
                        "family": spec["family"],
                        "condition": condition,
                        "judge_model": judge,
                        "case_id": r.get("id"),
                        "swap": swap,
                        "correct": correct,
                    }
                )
    return pd.DataFrame(rows)


def _scan_baselines(baselines_dir: Path, spec: dict[str, str]) -> pd.DataFrame:
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
            if judge_dir.name not in {
                spec["weaker_model"],
                spec["frontier_model"],
            }:
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
                    if not _is_explicitly_complete(r.get("complete_judge")):
                        continue
                    ans = r.get("answer_judge", "")
                    correct = _judge_correctness(ans, swap)
                    if correct is None:
                        continue
                    rows.append(
                        {
                            "family": spec["family"],
                            "arm": "B1" if arm == "baseline_blind" else "B2",
                            "judge_model": judge_dir.name,
                            "case_id": r.get("id"),
                            "swap": swap,
                            "correct": correct,
                        }
                    )
    return pd.DataFrame(rows)


def _scan_baselines_for_families(
    baselines_dir: Optional[Path],
    family_specs: list[tuple[Path, dict[str, str]]],
) -> pd.DataFrame:
    """Collect baselines for each family.

    `baselines_dir` can be either:
      - a family-specific directory, e.g. run/baselines/openai
      - a parent directory with family subdirs, e.g. run/baselines
    """
    if not baselines_dir or not family_specs:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for family_dir, spec in family_specs:
        family_label = spec["family"]
        family_baselines = baselines_dir / family_label
        target = family_baselines if family_baselines.exists() else baselines_dir
        df = _scan_baselines(target, spec)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def load_per_case_scores(
    family_dirs: list[Path],
    baselines_dir: Optional[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the cached judgement CSVs into swap-averaged per-case scores.

    Returns `(debate_per_case, baselines_per_case)`. Public so
    `compute_paired_tests.py` derives its per-case vectors from exactly the
    same parsing and swap-averaging as the tables and plots here — the paired
    tests must not disagree with the headline accuracies about what a case
    scored.
    """
    family_specs = [
        (directory, _family_run_spec(directory)) for directory in family_dirs
    ]
    per_case_all = (
        pd.concat(
            [_scan_family_dir(directory, spec) for directory, spec in family_specs],
            ignore_index=True,
        )
        if family_specs
        else pd.DataFrame()
    )
    baselines_per_case = _scan_baselines_for_families(baselines_dir, family_specs)
    return (
        _swap_average(per_case_all) if not per_case_all.empty else pd.DataFrame(),
        _swap_average(baselines_per_case)
        if not baselines_per_case.empty
        else pd.DataFrame(),
    )


def aligned_case_scores(
    per_case: pd.DataFrame,
    baselines_case: pd.DataFrame,
    family: str,
    condition: str,
    judge: str,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (debate, blind, oracle) score vectors aligned by case id.

    Only cases present in all three arms for this (family, judge) are kept, so
    index i refers to the same patient case in every returned array. Returns
    None when any arm is missing entirely — e.g. baselines for this judge have
    not been run.
    """
    debate = per_case[
        (per_case["family"] == family)
        & (per_case["condition"] == condition)
        & (per_case["judge_model"] == judge)
    ].set_index("case_id")["acc"]

    def _baseline(arm: str) -> pd.Series:
        return baselines_case[
            (baselines_case["family"] == family)
            & (baselines_case["arm"] == arm)
            & (baselines_case["judge_model"] == judge)
        ].set_index("case_id")["acc"]

    blind, oracle = _baseline("B1"), _baseline("B2")
    if debate.empty or blind.empty or oracle.empty:
        return None

    common = sorted(set(debate.index) & set(blind.index) & set(oracle.index))
    if not common:
        return None
    return (
        debate.reindex(common).to_numpy(dtype=float),
        blind.reindex(common).to_numpy(dtype=float),
        oracle.reindex(common).to_numpy(dtype=float),
    )


def _swap_average(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse swap=False/True pairs into a single per-case accuracy.

    A case only gets a score when BOTH judging passes parsed. `mean()` skips
    NaN, so a case whose swapped pass was unparseable would otherwise be
    scored off the surviving pass alone — a single correct pass would read
    1.0, indistinguishable from a case the judge got right in both orders.
    That silently bypasses the position-bias control that swap judging exists
    to provide, and it inflates accuracy.

    Such cases are scored NaN so they drop out of the means and paired tests
    rather than counting as clean wins. `n_parsed` records how many passes
    actually parsed, so the shortfall is visible in the CSV instead of being
    hidden behind `n_passes`, which counts rows present rather than usable.
    """
    group_cols = [c for c in df.columns if c not in {"swap", "correct"}]
    grouped = (
        df.groupby(group_cols)
        .agg(
            acc=("correct", "mean"),
            n_passes=("correct", "size"),
            n_parsed=("correct", "count"),
        )
        .reset_index()
    )
    grouped.loc[grouped["n_parsed"] < 2, "acc"] = float("nan")
    return grouped


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float, float]:
    """Returns (mean, ci_lo, ci_hi) with 95% percentile bootstrap.

    Delegates to `medical_stats.mean_ci` so this script and
    `compute_paired_tests.py` share one bootstrap implementation, resample
    count, and seed. They used to differ, which is how the repo ended up
    shipping two different CIs for the same PGR values.
    """
    return medical_stats.mean_ci(values)


def accuracy_table(per_case: pd.DataFrame) -> pd.DataFrame:
    """E1 — accuracy + bootstrap CIs per (family, condition, judge_model)."""
    if per_case.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "condition",
                "judge_model",
                "n_cases",
                "acc_mean",
                "ci_lo",
                "ci_hi",
            ]
        )
    out: list[dict] = []
    for (family, condition, judge), grp in per_case.groupby(
        ["family", "condition", "judge_model"]
    ):
        mean, lo, hi = _bootstrap_ci(grp["acc"].to_numpy())
        out.append(
            {
                "family": family,
                "condition": condition,
                "judge_model": judge,
                "n_cases": int(grp["acc"].notna().sum()),
                "acc_mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(out).sort_values(["family", "condition"]).reset_index(drop=True)


def baseline_table(baselines_case: pd.DataFrame) -> pd.DataFrame:
    if baselines_case.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "arm",
                "judge_model",
                "n_cases",
                "acc_mean",
                "ci_lo",
                "ci_hi",
            ]
        )
    out: list[dict] = []
    for (family, arm, judge), grp in baselines_case.groupby(
        ["family", "arm", "judge_model"]
    ):
        mean, lo, hi = _bootstrap_ci(grp["acc"].to_numpy())
        out.append(
            {
                "family": family,
                "arm": arm,
                "judge_model": judge,
                "n_cases": int(grp["acc"].notna().sum()),
                "acc_mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
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


def pgr_table(
    accuracy: pd.DataFrame,
    baselines: pd.DataFrame,
    per_case: pd.DataFrame,
    baselines_case: pd.DataFrame,
) -> pd.DataFrame:
    """E2 — PGR per (family, condition).

    PGR(condition) = (acc_condition − acc_B1_same_judge) / (acc_B2_same_judge − acc_B1_same_judge).

    The confidence interval is a **case-paired bootstrap** over the underlying
    per-case scores (`medical_stats.pgr_ci`): each replicate resamples cases
    once and reads debate, blind, and oracle at those same positions. PGR's
    numerator and denominator share the blind term, so the arms have to be
    resampled together.

    NaN throughout if `baselines` is empty (baselines haven't been run yet).
    """
    rows: list[dict] = []
    if accuracy.empty:
        return pd.DataFrame(rows)

    for _, r in accuracy.iterrows():
        cond, judge, family = r["condition"], r["judge_model"], r["family"]
        pgr_val, lo, hi = float("nan"), float("nan"), float("nan")
        if not baselines_case.empty and not per_case.empty:
            arms = aligned_case_scores(per_case, baselines_case, family, cond, judge)
            if arms is not None:
                debate, blind, oracle = arms
                pgr_val, lo, hi = medical_stats.pgr_ci(debate, blind, oracle)
        rows.append(
            {
                "family": family,
                "condition": cond,
                "judge_model": judge,
                "pgr": pgr_val,
                "pgr_ci_lo": lo,
                "pgr_ci_hi": hi,
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["family", "condition"]).reset_index(drop=True)
    )


def per_case_lift_table(
    per_case: pd.DataFrame, baselines_case: pd.DataFrame
) -> pd.DataFrame:
    """E3 — per case, did debate flip the answer vs baseline?

    Joins each per-case (debate) accuracy against the same-judge B1 baseline
    accuracy. Emits one row per (case, condition, family). The diagnostic plot
    preserves all three swap-averaged states (0.0 / 0.5 / 1.0); 0.5 is position
    sensitivity, not a clean correct result.
    """
    if per_case.empty or baselines_case.empty:
        return pd.DataFrame()

    def _arm(arm: str, column: str) -> pd.DataFrame:
        return (
            baselines_case[baselines_case["arm"] == arm]
            .groupby(["family", "judge_model", "case_id"])["acc"]
            .mean()
            .rename(column)
            .reset_index()
        )

    merged = per_case.merge(
        _arm("B1", "blind_acc"), on=["family", "judge_model", "case_id"], how="left"
    )
    # `oracle_acc` is carried too so the curated bundle is self-sufficient for
    # the paired debate-vs-oracle tests. Without it those tests could only be
    # rebuilt by going back to the baseline judgement CSVs.
    merged = merged.merge(
        _arm("B2", "oracle_acc"), on=["family", "judge_model", "case_id"], how="left"
    )
    merged = merged.rename(columns={"acc": "debate_acc"})
    return merged[
        [
            "family",
            "condition",
            "judge_model",
            "case_id",
            "debate_acc",
            "blind_acc",
            "oracle_acc",
        ]
    ]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _remove_stale_artifact(path: Path, reason: str) -> None:
    """Remove an artifact that the current inputs cannot regenerate."""
    if path.exists():
        path.unlink()
        print(f"removed stale {path} ({reason})", file=sys.stderr)


_COND_ORDER = [
    "e1_info_asymmetry",
    "e2_double_asymmetry",
    "e3_capability_asymmetry",
    "e4_full_symmetry",
]
_COND_LABELS = {
    "e1_info_asymmetry": "E1\ninfo asym",
    "e2_double_asymmetry": "E2\ndouble asym",
    "e3_capability_asymmetry": "E3\ncap asym",
    "e4_full_symmetry": "E4\nsymmetry",
}


def _judge_roles_for_family(
    accuracy: pd.DataFrame, family: str
) -> dict[str, Optional[str]]:
    """Return {"weaker": <judge_model or None>, "frontier": <judge_model or None>}
    for one family, derived from the accuracy table itself.

    Uses `_CONDITION_JUDGE_TYPE` to identify which debate conditions are
    judged by the weaker vs frontier model, then reads the `judge_model`
    actually used for those conditions in this run. This avoids hard-coding
    the family→model mapping inside the aggregator.
    """
    weaker_conds = {c for c, t in _CONDITION_JUDGE_TYPE.items() if t == "weaker"}
    frontier_conds = {c for c, t in _CONDITION_JUDGE_TYPE.items() if t == "frontier"}
    w = accuracy[
        (accuracy["family"] == family) & (accuracy["condition"].isin(weaker_conds))
    ]
    f = accuracy[
        (accuracy["family"] == family) & (accuracy["condition"].isin(frontier_conds))
    ]
    return {
        "weaker": str(w["judge_model"].iloc[0]) if not w.empty else None,
        "frontier": str(f["judge_model"].iloc[0]) if not f.empty else None,
    }


def plot_accuracy(
    accuracy: pd.DataFrame, baselines: pd.DataFrame, out_path: Path
) -> None:
    """Two-panel per-condition accuracy plot with same-judge baseline pairing.

    Left panel ("weaker-judge view") shows B1/B2 anchored to the weaker
    judge alongside E2 and E3. Right panel ("frontier-judge view") shows
    B1/B2 anchored to the frontier judge alongside E1 and E4. Baselines
    are never averaged across judges — each debate arm sits next to the
    floor and ceiling generated by *its own* judge.
    """
    if accuracy.empty:
        _remove_stale_artifact(out_path, "no accuracy data")
        return
    families = sorted(accuracy["family"].unique())
    judge_roles = {fam: _judge_roles_for_family(accuracy, fam) for fam in families}

    fig, (ax_w, ax_f) = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    width = 0.35

    panels = [
        (
            ax_w,
            "Weaker-judge view (E2 / E3)",
            "weaker",
            ["B1", "B2", "e2_double_asymmetry", "e3_capability_asymmetry"],
        ),
        (
            ax_f,
            "Frontier-judge view (E1 / E4)",
            "frontier",
            ["B1", "B2", "e1_info_asymmetry", "e4_full_symmetry"],
        ),
    ]

    for ax, title, role, arms in panels:
        base_x = list(range(len(arms)))
        x_labels = [
            "B1\nblind" if k == "B1" else "B2\noracle" if k == "B2" else _COND_LABELS[k]
            for k in arms
        ]

        for fi, family in enumerate(families):
            target_judge = judge_roles[family].get(role)
            offset = (fi - (len(families) - 1) / 2) * width
            heights, lo, hi = [], [], []
            for k in arms:
                if k in ("B1", "B2"):
                    if target_judge is None:
                        rows = baselines.iloc[0:0]
                    else:
                        rows = baselines[
                            (baselines["family"] == family)
                            & (baselines["arm"] == k)
                            & (baselines["judge_model"] == target_judge)
                        ]
                else:
                    rows = accuracy[
                        (accuracy["family"] == family) & (accuracy["condition"] == k)
                    ]
                if rows.empty:
                    heights.append(np.nan)
                    lo.append(np.nan)
                    hi.append(np.nan)
                else:
                    heights.append(rows["acc_mean"].mean())
                    lo.append(rows["ci_lo"].mean())
                    hi.append(rows["ci_hi"].mean())
            xs = [b + offset for b in base_x]
            yerr = [
                [
                    h - l if not (np.isnan(h) or np.isnan(l)) else 0
                    for h, l in zip(heights, lo)
                ],
                [
                    u - h if not (np.isnan(h) or np.isnan(u)) else 0
                    for h, u in zip(heights, hi)
                ],
            ]
            label = f"{family} · {target_judge}" if target_judge else family
            ax.bar(xs, heights, width=width, label=label, yerr=yerr, capsize=3)

        ax.set_xticks(base_x)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)

    ax_w.set_ylabel("Judge accuracy")
    fig.suptitle(
        "Per-condition judge accuracy (same-judge baseline pairing, bootstrap 95% CI)"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pgr(pgr: pd.DataFrame, out_path: Path) -> None:
    if pgr.empty or pgr["pgr"].isna().all():
        _remove_stale_artifact(out_path, "no scoreable PGR data")
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


_SWAP_SCORE_STATES = (0.0, 0.5, 1.0)


def _lift_contingency(sub: pd.DataFrame) -> pd.DataFrame:
    """Build a wrong / position-sensitive / correct transition table."""
    counts = {
        (blind, debate): 0
        for blind in _SWAP_SCORE_STATES
        for debate in _SWAP_SCORE_STATES
    }
    for _, row in sub.iterrows():
        blind = next(
            (
                state
                for state in _SWAP_SCORE_STATES
                if np.isclose(row["blind_acc"], state)
            ),
            None,
        )
        debate = next(
            (
                state
                for state in _SWAP_SCORE_STATES
                if np.isclose(row["debate_acc"], state)
            ),
            None,
        )
        if blind is not None and debate is not None:
            counts[(blind, debate)] += 1
    return pd.DataFrame(
        [
            [counts[(blind, debate)] for debate in _SWAP_SCORE_STATES]
            for blind in _SWAP_SCORE_STATES
        ],
        index=_SWAP_SCORE_STATES,
        columns=_SWAP_SCORE_STATES,
    )


def _lift_contingencies_by_family(
    lift: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Return one 3x3 transition table per family and condition.

    Keeping the family split in this intermediate representation makes it hard
    for plotting changes to accidentally pool providers.  A pooled heatmap can
    look perfectly plausible while every cell count is wrong for both families.
    """
    if lift.empty:
        return {}
    if "family" not in lift.columns:
        raise ValueError("per-case lift data is missing the family column")
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for family in sorted(lift["family"].dropna().astype(str).unique()):
        family_rows = lift[lift["family"].astype(str) == family]
        tables[family] = {
            condition: _lift_contingency(
                family_rows[family_rows["condition"] == condition].dropna(
                    subset=["blind_acc", "debate_acc"]
                )
            )
            for condition in _COND_ORDER
        }
    return tables


def plot_per_case_lift(lift: pd.DataFrame, out_path: Path) -> None:
    if lift.empty:
        _remove_stale_artifact(out_path, "no per-case lift data")
        return
    family_tables = _lift_contingencies_by_family(lift)
    if not family_tables:
        _remove_stale_artifact(out_path, "no family-labelled per-case lift data")
        return
    families = list(family_tables)
    fig, axes = plt.subplots(
        len(families),
        max(1, len(_COND_ORDER)),
        figsize=(3 * len(_COND_ORDER), 3.5 * len(families)),
        squeeze=False,
    )
    for family_index, family in enumerate(families):
        family_rows = lift[lift["family"].astype(str) == family]
        for condition_index, cond in enumerate(_COND_ORDER):
            ax = axes[family_index][condition_index]
            sub = family_rows[family_rows["condition"] == cond]
            if sub.empty or sub["blind_acc"].isna().all():
                ax.set_title(_COND_LABELS[cond])
                ax.text(
                    0.5,
                    0.5,
                    "no baseline data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            # Swap averaging has three meaningful outcomes: 0.0 (wrong in both
            # orders), 0.5 (the answer changed with position), and 1.0 (correct
            # in both orders). Keep all three states visible. Treating 0.5 as
            # "right" would hide the position sensitivity swap judging exists
            # to expose.
            sub = sub.dropna(subset=["blind_acc", "debate_acc"]).copy()
            if sub.empty:
                ax.set_title(_COND_LABELS[cond])
                ax.text(
                    0.5,
                    0.5,
                    "no scoreable cases",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            ct = family_tables[family][cond]
            ax.imshow(ct.values, cmap="Blues")
            for ri in range(3):
                for ci in range(3):
                    ax.text(
                        ci,
                        ri,
                        ct.values[ri, ci],
                        ha="center",
                        va="center",
                        color="black",
                    )
            state_labels = ["wrong", "position-\nsensitive", "correct"]
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(state_labels)
            ax.set_xlabel("debate")
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(state_labels)
            ylabel = "blind"
            if len(families) > 1 and condition_index == 0:
                ylabel = f"{family}\n\nblind"
            ax.set_ylabel(ylabel)
            ax.set_title(_COND_LABELS[cond])
    family_note = "" if len(families) == 1 else " · one row per model family"
    fig.suptitle(
        "Per-case lift: debate vs blind baseline (3×3 per condition)" + family_note
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _family_label_from_path(family_dir: Path) -> str:
    """Return the validated family recorded in authoritative run metadata."""
    return _family_run_spec(family_dir)["family"]


def main(
    family_dirs: list[Path],
    baselines_dir: Optional[Path],
    out_dir: Path,
    plots_only: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    per_case_agg, baselines_case_agg = load_per_case_scores(family_dirs, baselines_dir)

    acc = accuracy_table(per_case_agg)
    bls = baseline_table(baselines_case_agg)
    pgr = pgr_table(acc, bls, per_case_agg, baselines_case_agg)
    lift = per_case_lift_table(per_case_agg, baselines_case_agg)

    if not plots_only:
        acc.to_csv(out_dir / "accuracy_by_condition.csv", index=False)
        # PGR is deliberately not written here. `compute_paired_tests.py`
        # emits it as part of `headline_results.csv`, alongside the blind /
        # debate / oracle accuracies it is derived from, so there is one PGR
        # table rather than two that can drift apart. The in-memory `pgr`
        # table below still drives plot 02.
        lift.to_csv(out_dir / "per_case_lift.csv", index=False)
        if not bls.empty:
            bls.to_csv(out_dir / "baselines_by_arm.csv", index=False)
        else:
            _remove_stale_artifact(out_dir / "baselines_by_arm.csv", "no baseline data")

    plot_accuracy(acc, bls, out_dir / "plots" / "01_accuracy_by_condition.png")
    plot_pgr(pgr, out_dir / "plots" / "02_pgr_by_condition.png")
    plot_per_case_lift(lift, out_dir / "plots" / "03_per_case_lift.png")

    print(f"wrote results to {out_dir}/")
    print(f"  accuracy_by_condition.csv  ({len(acc)} rows)")
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
        help="One or more family directories, e.g. exp/YYYY-MM-DD_HH-MM-SS_results/openai",
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
