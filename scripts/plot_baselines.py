"""Plot blind vs oracle baseline accuracy from a baselines `results.csv`.

Reads the four-row CSV that `core.scoring.accuracy` appends after running
`baseline_blind` and `baseline_oracle` (original + swap each), and writes
two PNGs into a `plots/` subdirectory next to the CSV:

  01_blind_vs_oracle.png  — swap-averaged accuracy per arm, headroom annotation.
  02_position_bias.png    — original vs swap per arm, diagnostic for A/B bias.

Pure data manipulation — no API calls. Safe to re-run.

Usage:
    python scripts/plot_baselines.py exp/<run>/baselines/openai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARM_LABEL = {"baseline_blind": "B1 · blind", "baseline_oracle": "B2 · oracle"}
ARM_COLOR = {"baseline_blind": "#4C78A8", "baseline_oracle": "#F58518"}
ARM_ORDER = ["baseline_blind", "baseline_oracle"]


def _load_results(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    needed = {"method", "accuracy", "swap", "num_matches", "model"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{results_csv} is missing columns: {sorted(missing)}")
    df["swap"] = df["swap"].astype(str).str.lower().map({"true": True, "false": False})
    return df[df["method"].isin(ARM_ORDER)].copy()


def _swap_averaged(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("method", as_index=False)
        .agg(accuracy=("accuracy", "mean"), n=("num_matches", "max"))
        .set_index("method")
        .reindex(ARM_ORDER)
        .reset_index()
    )


def plot_blind_vs_oracle(df: pd.DataFrame, out_path: Path, judge_model: str) -> None:
    avg = _swap_averaged(df)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = np.arange(len(ARM_ORDER))
    bars = ax.bar(
        xs,
        avg["accuracy"] * 100,
        color=[ARM_COLOR[a] for a in avg["method"]],
        width=0.5,
    )
    for bar, acc in zip(bars, avg["accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{acc * 100:.1f}%",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    # Headroom annotation
    if len(avg) == 2 and not avg["accuracy"].isna().any():
        b1, b2 = float(avg["accuracy"].iloc[0]), float(avg["accuracy"].iloc[1])
        gap = (b2 - b1) * 100
        ax.annotate(
            f"headroom\n{gap:+.1f} pp",
            xy=(1, b2 * 100),
            xytext=(0.5, (b1 + b2) / 2 * 100),
            ha="center",
            fontsize=9,
            color="#444",
            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8),
        )

    ax.axhline(50, color="grey", linestyle="--", alpha=0.5, label="chance (50%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABEL[a] for a in avg["method"]])
    ax.set_ylabel("Judge accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title(
        f"Blind vs oracle baseline · {judge_model} · n={int(avg['n'].iloc[0])}\n"
        f"(swap-averaged across original + swapped A/B)"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_position_bias(df: pd.DataFrame, out_path: Path, judge_model: str) -> None:
    """Show original vs swap accuracy per arm, so any A/B position bias is visible."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = np.arange(len(ARM_ORDER))
    width = 0.35
    for i, swap_val in enumerate([False, True]):
        sub = df[df["swap"] == swap_val].set_index("method").reindex(ARM_ORDER)
        offset = (i - 0.5) * width
        ax.bar(
            xs + offset,
            sub["accuracy"] * 100,
            width=width,
            label="original A/B" if not swap_val else "swapped A/B",
            color=["#7baad6", "#f7b27e"] if not swap_val else ["#365a8c", "#c97318"],
        )
    ax.axhline(50, color="grey", linestyle="--", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER])
    ax.set_ylabel("Judge accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title(
        f"Position-bias diagnostic · {judge_model}\n"
        f"(gap between bars within an arm = A/B presentation effect)"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(baselines_dir: Path) -> int:
    results_csv = baselines_dir / "results.csv"
    if not results_csv.exists():
        print(f"missing {results_csv}", file=sys.stderr)
        return 1
    df = _load_results(results_csv)
    if df.empty:
        print(f"no baseline rows in {results_csv}", file=sys.stderr)
        return 1
    judge_model = str(df["model"].iloc[0])
    plots_dir = baselines_dir / "plots"
    plot_blind_vs_oracle(df, plots_dir / "01_blind_vs_oracle.png", judge_model)
    plot_position_bias(df, plots_dir / "02_position_bias.png", judge_model)
    print(f"wrote {plots_dir}/01_blind_vs_oracle.png and 02_position_bias.png")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "baselines_dir",
        type=Path,
        help="Directory containing results.csv (e.g. exp/<run>/baselines/openai).",
    )
    args = parser.parse_args()
    sys.exit(main(args.baselines_dir))
