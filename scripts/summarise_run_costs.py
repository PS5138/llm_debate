"""Walk an exp_dir's Hydra logs and emit a cost summary CSV.

Both adapters log per-call cost lines, e.g.

    [...][core.agents.judge_quality][INFO] - Total cost: 0.019

This script greps those out, attributes each line to its stage (debate /
judge / scoring / concession) by walking the logs/ subtree, and writes
`<exp_dir>/cost_summary.csv` plus a short stdout report.

The line totals only count what the adapters actually billed for. For a
reconciliation against the predicted budget see §6.4 of the funding doc
and the Phase E7 entry in PLAN.md.

Usage:
    python scripts/summarise_run_costs.py exp/YYYY-MM-DD_HH-MM-SS_results/openai
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Hydra logs land at <exp_dir>/<stage>/logs/<date>/<time>/<stage>.log
# But also some pipelines write <exp_dir>/logs/... — handle both.
_COST_RE = re.compile(r"Total cost:\s*([0-9.]+)")
_STAGE_FROM_PATH = re.compile(r"/(?:logs/)?[^/]+\.log$")


def _stage_for(log_path: Path, exp_root: Path) -> str:
    """Best-effort stage attribution from the log file's path."""
    rel = log_path.relative_to(exp_root).as_posix()
    # Filename without extension is the stage name (debate, judge, scoring, ...)
    return log_path.stem


def collect(exp_dir: Path) -> pd.DataFrame:
    if not exp_dir.exists():
        raise SystemExit(f"no such exp_dir: {exp_dir}")

    rows: list[dict] = []
    for log_path in sorted(exp_dir.rglob("*.log")):
        stage = _stage_for(log_path, exp_dir)
        try:
            text = log_path.read_text(errors="ignore")
        except OSError:
            continue
        for m in _COST_RE.finditer(text):
            rows.append({
                "log": str(log_path.relative_to(exp_dir)),
                "stage": stage,
                "cost_usd": float(m.group(1)),
            })
    return pd.DataFrame(rows)


def main(exp_dir: Path) -> None:
    df = collect(exp_dir)
    out_path = exp_dir / "cost_summary.csv"
    if df.empty:
        out_path.write_text("stage,cost_usd\n")
        print(f"no cost lines found under {exp_dir}/")
        return

    df.to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print()
    print("Per-stage totals:")
    print(df.groupby("stage")["cost_usd"].agg(["count", "sum"]).round(4).to_string())
    print()
    print(f"Total spend across all logs in {exp_dir}: ${df['cost_usd'].sum():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    args = parser.parse_args()
    main(args.exp_dir)
