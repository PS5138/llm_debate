"""Walk an exp_dir's Hydra logs and emit a cost summary CSV.

Both adapters log lines like

    [...][core.agents.judge_quality][INFO] - Total cost: 0.019

**These are cumulative, not per-call.** The value logged is
`api_handler.running_cost` (`core/llm_api/llm.py`), a counter that only ever
increases: each API call does `running_cost += ...` and `reset_cost()` is
never called anywhere in the codebase. Each stage entrypoint
(`core/debate.py`, `core/judge.py`) builds exactly one `ModelAPI` per process,
and Hydra gives each process its own `logs/<date>/<time>/<stage>.log`. So
within a single log file every "Total cost" line is a snapshot of the *same*
running counter, and the file's spend is the **maximum** of those snapshots,
not their sum.

Summing the lines instead over-counts roughly in proportion to how many
snapshots each log contains: on the OpenAI 100-case run it reported $1554.60
against an actual $161.90.

Total spend for the run is therefore the sum, over log files, of each file's
maximum. This is a very slight *under*-count: the counter is snapshotted when
a debate/judgement completes, so any calls billed after the final snapshot in
a process are invisible to the log.

Usage:
    python scripts/summarise_run_costs.py exp/YYYY-MM-DD_HH-MM-SS_results/openai
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

_COST_RE = re.compile(r"Total cost:\s*([0-9.]+)")


def _stage_for(log_path: Path) -> str:
    """Best-effort stage attribution: the filename is the stage name
    (debate, judge, accuracy, ...)."""
    return log_path.stem


def collect(roots: list[Path]) -> pd.DataFrame:
    """One row per log file across every root, holding that process's spend.

    Columns: source_root, log, stage, cost_usd, snapshots, non_monotonic.

    Several roots are accepted because a run's spend is not all under the
    family directory: the blind and oracle baseline stages log under
    `<run>/baselines/<family>/`. Summarising only the family dir silently
    omits them, which matters whenever baselines were computed fresh rather
    than reused from the committed canonical set.
    """
    missing = [r for r in roots if not r.exists()]
    if missing and len(missing) == len(roots):
        raise SystemExit(f"no such directory: {missing[0]}")

    rows: list[dict] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for log_path in sorted(root.rglob("*.log")):
            resolved = log_path.resolve()
            # Roots may nest (e.g. a run root and its own family subdir), so
            # guard against counting the same process twice.
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                text = log_path.read_text(errors="ignore")
            except OSError:
                continue
            values = [float(m.group(1)) for m in _COST_RE.finditer(text)]
            if not values:
                continue
            # A shared, never-reset counter should never decrease. If it does,
            # this log mixes several handlers (or the adapter changed) and
            # max() would under-count -- flag it rather than silently
            # reporting a number we can no longer defend.
            non_monotonic = any(b < a for a, b in zip(values, values[1:]))
            rows.append(
                {
                    "source_root": str(root),
                    "log": str(log_path.relative_to(root)),
                    "stage": _stage_for(log_path),
                    "cost_usd": max(values),
                    "snapshots": len(values),
                    "non_monotonic": non_monotonic,
                }
            )
    return pd.DataFrame(rows)


def main(roots: list[Path]) -> None:
    df = collect(roots)
    out_path = roots[0] / "cost_summary.csv"
    if df.empty:
        out_path.write_text("source_root,log,stage,cost_usd,snapshots,non_monotonic\n")
        print(f"no cost lines found under {', '.join(str(r) for r in roots)}")
        return

    df.to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print()
    print("Per-stage totals (sum of per-process maxima):")
    per_stage = df.groupby("stage").agg(
        processes=("cost_usd", "size"),
        snapshots=("snapshots", "sum"),
        cost_usd=("cost_usd", "sum"),
    )
    print(per_stage.round(4).to_string())
    print()
    print("Per-root totals:")
    print(df.groupby("source_root")["cost_usd"].sum().round(4).to_string())
    print()
    print(f"Total spend across all logs scanned: ${df['cost_usd'].sum():.4f}")
    print(
        "Note: this is what the adapters logged. The counter is snapshotted "
        "when a debate or judgement completes, so calls billed after the "
        "final snapshot in a process are not represented."
    )

    flagged = df[df["non_monotonic"]]
    if not flagged.empty:
        print()
        print(
            "WARNING: cost snapshots decrease within these logs, so they may "
            "mix multiple API handlers and the totals above may under-count:"
        )
        for log in flagged["log"]:
            print(f"  {log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="One or more directories to scan for Hydra logs. "
        "cost_summary.csv is written into the first one. Pass the family dir "
        "and its baselines dir to cover a whole run.",
    )
    args = parser.parse_args()
    main(args.roots)
