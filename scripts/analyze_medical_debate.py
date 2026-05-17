"""Bias-control analyses for the medical debate pipeline.

Reads the cached debate transcripts and judgement CSVs produced by
`scripts/run_medical_debate.sh` and writes four diagnostic artefacts:

  1. verbosity.csv        — argument word counts per side per round
  2. quote_verification.csv — verified vs unverified <quote> tags per side
  3. concession.csv       — concession-judge results per side
                            (only populated if the concession judge has
                             been run; otherwise the file lists which
                             rows were missing)
  4. debate_outcomes.csv  — counts of complete, conceded, and usable debates

For every CSV, a matching `*_summary.txt` is also written that gives a
small human-readable summary — useful when N is small (e.g. on a smoke
run) and a full Pearson correlation would be noisy.

Usage:
    python scripts/analyze_medical_debate.py exp/YYYY-MM-DD_HH-MM-SS_results/openai
"""

from __future__ import annotations

import argparse
import json
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

# Use the same normalisation rule as the judge so the verification rate
# reported here matches what the judge actually trusted in the transcript.
# Importing core/transcript_parser keeps the two sides in lockstep.
from core.transcript_parser import TranscriptParser


QUOTE_RE = re.compile(r"<quote>(.*?)</quote>", re.DOTALL)
WORD_RE = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def load_transcripts(debate_csv: Path) -> list[dict]:
    """Return parsed transcript dicts from the debate-stage CSV."""
    df = pd.read_csv(debate_csv, keep_default_na=False)
    transcripts = []
    for i, row in df.iterrows():
        try:
            transcripts.append(json.loads(row["transcript"]))
        except json.JSONDecodeError:
            continue
    return transcripts


def load_debate_rows(debate_csv: Path) -> pd.DataFrame:
    """Return debate-stage rows with parse status for outcome accounting."""
    df = pd.read_csv(debate_csv, keep_default_na=False)
    if "complete" not in df.columns:
        df["complete"] = False
    df["complete"] = df["complete"].astype(str).str.lower().isin({"true", "1", "yes"})

    parsed_ids = []
    for _, row in df.iterrows():
        try:
            transcript = json.loads(row.get("transcript", ""))
            parsed_ids.append(transcript.get("question_set_id") or transcript.get("index") or row.get("id"))
        except (TypeError, json.JSONDecodeError):
            parsed_ids.append("")
    df["parsed_case_id"] = parsed_ids
    df["transcript_parsed"] = df["parsed_case_id"].astype(str).str.len() > 0
    return df


# ---------------------------------------------------------------------------
# Verbosity
# ---------------------------------------------------------------------------


def verbosity_rows(transcripts: list[dict]) -> list[dict]:
    rows = []
    for t in transcripts:
        case = t.get("question_set_id") or t.get("index")
        for round_idx, r in enumerate(t.get("rounds", []), start=1):
            for side in ("correct", "incorrect"):
                arg = r.get(side) or ""
                rows.append(
                    {
                        "case_id": case,
                        "round": round_idx,
                        "side": side,
                        "words": count_words(arg),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------


def quote_rows(transcripts: list[dict]) -> list[dict]:
    rows = []
    for t in transcripts:
        case = t.get("question_set_id") or t.get("index")
        story = t.get("story") or ""
        # Normalise the story once per case using the same rule the judge
        # applies (lowercase, strip punctuation, normalise smart quotes,
        # collapse whitespace) so the verification rate matches what the
        # judge actually saw as <v_quote> vs <u_quote>.
        story_norm = TranscriptParser.normalize_text(story)
        for round_idx, r in enumerate(t.get("rounds", []), start=1):
            for side in ("correct", "incorrect"):
                arg = r.get(side) or ""
                quotes = QUOTE_RE.findall(arg)
                verified = sum(
                    1
                    for q in quotes
                    if TranscriptParser.normalize_text(q) in story_norm
                )
                rows.append(
                    {
                        "case_id": case,
                        "round": round_idx,
                        "side": side,
                        "n_quotes": len(quotes),
                        "n_verified": verified,
                        "n_unverified": len(quotes) - verified,
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Concession
# ---------------------------------------------------------------------------


def concession_rows(exp_dir: Path) -> Optional[list[dict]]:
    """Return concession-judge results, combining no-swap and swap passes.

    Concession judging writes its CSVs under
    `${exp_dir}/debate_sim/concession_<model>/data0[_swap]_judgement.csv`.
    A case is flagged as "conceded" if EITHER pass returned Y — concession
    in just one ordering is still a concession.
    """
    candidates = sorted((exp_dir / "debate_sim").glob("concession_*"))
    if not candidates:
        return None

    rows_by_case: dict = {}
    for sub in candidates:
        for fname in ("data0_judgement.csv", "data0_swap_judgement.csv"):
            p = sub / fname
            if not p.exists():
                continue
            df = pd.read_csv(p, keep_default_na=False)
            for _, row in df.iterrows():
                case_id = row.get("id")
                verdict = str(row.get("answer_concession", "")).strip()
                bucket = rows_by_case.setdefault(
                    case_id,
                    {
                        "case_id": case_id,
                        "judge_dir": sub.name,
                        "noswap_verdict": "",
                        "swap_verdict": "",
                    },
                )
                key = "swap_verdict" if "swap" in fname else "noswap_verdict"
                bucket[key] = verdict

    out = []
    for case_id, b in rows_by_case.items():
        any_conceded = b["noswap_verdict"] == "Y" or b["swap_verdict"] == "Y"
        out.append({**b, "conceded": any_conceded})
    return out


def debate_outcome_rows(debate_df: pd.DataFrame, concession_df: Optional[pd.DataFrame]) -> list[dict]:
    """Summarise how many generated debates are usable after QC checks."""
    total = len(debate_df)
    complete = int(debate_df["complete"].sum()) if not debate_df.empty else 0
    parsed = int(debate_df["transcript_parsed"].sum()) if not debate_df.empty else 0
    complete_and_parsed = min(complete, parsed)
    incomplete_or_unparsed = total - complete_and_parsed

    conceded = 0
    concession_judged = 0
    if concession_df is not None and not concession_df.empty:
        concession_judged = min(int(len(concession_df)), complete_and_parsed)
        conceded = int(concession_df["conceded"].sum())

    usable = max(0, concession_judged - conceded)
    missing_concession_judgement = max(0, complete_and_parsed - concession_judged)

    return [
        {"status": "usable_non_conceded", "count": usable},
        {"status": "conceded", "count": conceded},
        {"status": "complete_missing_concession_judgement", "count": missing_concession_judgement},
        {"status": "incomplete_or_unparsed", "count": incomplete_or_unparsed},
    ]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_verbosity(verbosity_df: pd.DataFrame, family_label: str, out_path: Path) -> None:
    """Bar chart of mean words per round per side, plus the per-case
    correct-vs-incorrect-side word balance overlaid as light dots.
    Lets a reader see at a glance whether one side is systematically wordier.
    """
    if verbosity_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    per_side = verbosity_df.groupby("side")["words"].agg(["mean", "std"])
    sides = list(per_side.index)
    means = per_side["mean"].values
    stds = per_side["std"].fillna(0).values
    colours = ["#3b7bd6", "#d6643b"]
    bars = ax.bar(sides, means, yerr=stds, capsize=4, color=colours, alpha=0.75,
                  edgecolor="black", linewidth=0.5)
    # Overlay per-round word counts as light scatter for additional context
    for i, side in enumerate(sides):
        ys = verbosity_df[verbosity_df["side"] == side]["words"].values
        xs = np.full_like(ys, i, dtype=float) + np.random.uniform(-0.15, 0.15, size=len(ys))
        ax.scatter(xs, ys, color="black", alpha=0.25, s=18, zorder=3)
    ax.axhline(150, color="grey", linestyle="--", alpha=0.6, label="150-word cap")
    ax.set_ylabel("Words per round")
    ax.set_title(f"Verbosity per debater — {family_label}\n(bar = mean ± SD; dots = individual rounds)")
    ax.set_ylim(0, max(170, float(verbosity_df["words"].max()) * 1.1))
    ax.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_concession(concession_df: pd.DataFrame, family_label: str, out_path: Path) -> None:
    """Single-bar chart of % debates flagged for concession, with the
    20% suspect-threshold drawn for reference.
    """
    if concession_df.empty:
        return
    rate = concession_df["conceded"].mean() * 100
    n = len(concession_df)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([family_label], [rate], color="#d6643b", alpha=0.8,
           edgecolor="black", linewidth=0.5)
    ax.axhline(20, color="grey", linestyle="--", alpha=0.6, label="20% suspect threshold")
    ax.set_ylabel("% of debates with concession")
    ax.set_ylim(0, max(30, rate * 1.4))
    ax.set_title(f"Concession rate — {family_label} (n={n})")
    ax.annotate(f"{rate:.1f}%", xy=(0, rate), xytext=(0, rate + 1.5),
                ha="center", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_debate_outcomes(outcomes_df: pd.DataFrame, family_label: str, out_path: Path) -> None:
    """Stacked count bar: usable debates, concessions, and missing/incomplete rows."""
    if outcomes_df.empty:
        return
    order = [
        ("usable_non_conceded", "usable", "#3b8c4f"),
        ("conceded", "conceded", "#d6643b"),
        ("complete_missing_concession_judgement", "missing concession check", "#d6b33b"),
        ("incomplete_or_unparsed", "incomplete/unparsed", "#8a8a8a"),
    ]
    counts = {
        row["status"]: int(row["count"])
        for _, row in outcomes_df.iterrows()
    }
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bottom = 0
    for status, label, colour in order:
        count = counts.get(status, 0)
        if count == 0:
            continue
        ax.bar([family_label], [count], bottom=[bottom], label=label,
               color=colour, edgecolor="black", linewidth=0.5)
        ax.text(0, bottom + count / 2, str(count), ha="center", va="center",
                fontsize=10, fontweight="bold")
        bottom += count

    total = sum(counts.values())
    ax.set_ylabel("Debate count")
    ax.set_title(f"Debate outcomes — {family_label} (n={total})")
    ax.set_ylim(0, max(1, total) * 1.15)
    ax.legend(loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_quote_verification(quotes_df: pd.DataFrame, family_label: str, out_path: Path) -> None:
    """Stacked bar per side: verified (green) + unverified (red).
    Reviewers ask 'did the debaters confabulate?' — this answers it.
    """
    if quotes_df.empty:
        return
    per_side = quotes_df.groupby("side")[["n_verified", "n_unverified"]].sum()
    sides = list(per_side.index)
    verified = per_side["n_verified"].values
    unverified = per_side["n_unverified"].values
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(sides, verified, label="verified <v_quote>", color="#3b8c4f",
           edgecolor="black", linewidth=0.5)
    ax.bar(sides, unverified, bottom=verified, label="unverified <u_quote>",
           color="#b83b3b", edgecolor="black", linewidth=0.5)
    totals = verified + unverified
    for i, total in enumerate(totals):
        if total > 0:
            pct = 100 * verified[i] / total
            ax.annotate(f"{int(verified[i])}/{int(total)}\n({pct:.0f}% verified)",
                        xy=(i, total), xytext=(0, 5),
                        textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Quote count")
    ax.set_title(f"Quote verification per side — {family_label}")
    ax.legend(loc="lower right")
    ax.set_ylim(0, max(totals) * 1.25 if len(totals) and max(totals) > 0 else 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def write_summary(out_dir: Path, name: str, lines: list[str]) -> None:
    (out_dir / f"{name}_summary.txt").write_text("\n".join(lines) + "\n")


def main(exp_dir: Path) -> None:
    debate_csv = exp_dir / "debate_sim" / "data0.csv"
    if not debate_csv.exists():
        raise SystemExit(f"missing transcript CSV: {debate_csv}")

    out_dir = exp_dir / "analysis"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    transcripts = load_transcripts(debate_csv)
    debate_df = load_debate_rows(debate_csv)
    n_cases = len(transcripts)
    # Use the leaf dir name (typically `openai` or `anthropic`) as a label
    family_label = exp_dir.name or "run"
    print(f"loaded {n_cases} transcript(s) from {debate_csv}")

    # 1. Verbosity ---------------------------------------------------------
    v = pd.DataFrame(verbosity_rows(transcripts))
    v.to_csv(out_dir / "verbosity.csv", index=False)
    if not v.empty:
        per_side = v.groupby("side")["words"].agg(["mean", "min", "max"]).round(1)
        diff_per_case = (
            v.groupby(["case_id", "side"])["words"].sum().unstack("side")
        )
        diff_per_case["delta_correct_minus_incorrect"] = (
            diff_per_case["correct"] - diff_per_case["incorrect"]
        )
        write_summary(
            out_dir,
            "verbosity",
            [
                "Verbosity — total words per debate, per side.",
                "",
                "Per-side word counts across rounds (averaged over cases):",
                per_side.to_string(),
                "",
                "Per-case totals:",
                diff_per_case.to_string(),
                "",
                "Interpretation: a large persistent gap between correct- and",
                "incorrect-side word counts would be a verbosity-bias risk.",
                "Pearson r between argument length and judge wins becomes",
                "meaningful at N >= 20.",
            ],
        )
        plot_verbosity(v, family_label, plots_dir / "04_verbosity.png")

    # 2. Quote verification -------------------------------------------------
    q = pd.DataFrame(quote_rows(transcripts))
    q.to_csv(out_dir / "quote_verification.csv", index=False)
    if not q.empty:
        per_side = (
            q.groupby("side")[["n_quotes", "n_verified", "n_unverified"]]
            .sum()
            .assign(
                pct_verified=lambda d: (100 * d.n_verified / d.n_quotes.replace(0, pd.NA))
                .round(1)
            )
        )
        write_summary(
            out_dir,
            "quote_verification",
            [
                "Quote verification — count and pass rate of <quote> tags",
                "checked against the patient evidence (verified = direct",
                "substring match).",
                "",
                per_side.to_string(),
                "",
                "Interpretation: low pct_verified means the side is",
                "confabulating quotes; the judge sees <u_quote> tags for those.",
            ],
        )
        plot_quote_verification(q, family_label, plots_dir / "06_quote_verification.png")

    # 3. Concession ---------------------------------------------------------
    c_rows = concession_rows(exp_dir)
    c = None
    if c_rows is None:
        (out_dir / "concession.csv").write_text(
            "concession judge has not been run on this exp_dir; "
            "re-run with judge_type=concession to populate.\n"
        )
        write_summary(
            out_dir,
            "concession",
            [
                "Concession analysis skipped — no `answer_concession` column.",
                "",
                "To populate, run:",
                "  python -m core.judge \\",
                f"    exp_dir={exp_dir} \\",
                "    +experiment=medical_debate \\",
                "    ++judge_type=concession \\",
                "    ++judge_name=concession",
            ],
        )
    else:
        c = pd.DataFrame(c_rows)
        c.to_csv(out_dir / "concession.csv", index=False)
        rate = c["conceded"].mean() if not c.empty else float("nan")
        write_summary(
            out_dir,
            "concession",
            [
                "Concession rate — fraction of debates flagged by the",
                "concession judge as having either debater concede to the",
                "opposing side.",
                "",
                f"n debates: {len(c)}",
                f"concession rate: {rate:.2%}",
                "",
                "Interpretation: > 20% on the incorrect-side debater means",
                "the framing is failing and the headline E1/E2 numbers for",
                "that family should be held back.",
            ],
        )
        plot_concession(c, family_label, plots_dir / "05_concession_rate.png")

    # 4. Debate outcomes ----------------------------------------------------
    outcomes = pd.DataFrame(debate_outcome_rows(debate_df, c))
    outcomes.to_csv(out_dir / "debate_outcomes.csv", index=False)
    write_summary(
        out_dir,
        "debate_outcomes",
        [
            "Debate outcomes — count of generated debates after completion",
            "and concession quality-control checks.",
            "",
            outcomes.to_string(index=False),
            "",
            "Interpretation: headline debate accuracy should be read over",
            "usable_non_conceded debates. A large conceded or incomplete",
            "count means the debate framing needs revision before scaling.",
        ],
    )
    plot_debate_outcomes(outcomes, family_label, plots_dir / "07_debate_outcomes.png")

    print(f"wrote analysis to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    args = parser.parse_args()
    main(args.exp_dir)
