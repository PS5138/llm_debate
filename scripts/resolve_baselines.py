"""Resolve which baselines directory a debate run should use.

Prints the chosen directory to stdout and exits 0, or explains what is
missing on stderr and exits 1.

Resolution order:
  1. the candidate directory, if it is fully populated for this family
  2. the committed canonical baselines for this family
  3. failure

This exists so `run_medical_debate.sh` can settle the question *before*
spending anything on API calls. PGR needs a same-judge blind and oracle
baseline for every debate arm, and that requirement used to surface only
at the very end of the pipeline — after the debate and judging stages had
already been paid for.

Usage:
    python scripts/resolve_baselines.py --family openai \\
        --candidate exp/<run>/baselines/openai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.runner import (  # noqa: E402
    FAMILY_MODELS,
    canonical_baselines_path,
    missing_baseline_files,
    pilot_case_ids,
)


def resolve(
    family: str,
    candidate: Path | None,
    judges: tuple[str, str] | None = None,
    min_rows: int | None = None,
) -> tuple[Path | None, str]:
    """Return (resolved_dir, human-readable explanation).

    `judges` and `min_rows` are checked against the run that is about to
    start, not against the family defaults: a run overriding FRONTIER/WEAKER
    needs baselines for *those* models, and an N-case debate needs baselines
    covering at least N cases.
    """
    if family not in FAMILY_MODELS:
        return None, f"unknown family: {family}"

    expected_ids = pilot_case_ids(min_rows) if min_rows is not None else None

    def _check(path: Path) -> list[str]:
        return missing_baseline_files(
            path,
            family,
            judges=judges,
            min_rows=min_rows,
            expected_ids=expected_ids,
        )

    if candidate is not None and candidate.exists():
        missing = _check(candidate)
        if not missing:
            return candidate, f"using baselines in {candidate}"
        shortfall = f"{len(missing)} issue(s), first: {missing[0]}"
        note = f"{candidate} is not a usable baseline set ({shortfall})"
    elif candidate is not None:
        note = f"{candidate} does not exist"
    else:
        note = "no candidate directory given"

    canonical = canonical_baselines_path(family)
    if canonical is not None:
        canonical_missing = _check(canonical)
        if not canonical_missing:
            return (
                canonical,
                f"{note}; falling back to canonical baselines at {canonical}",
            )
        note = (
            f"{note}; the canonical baselines at {canonical} do not match this "
            f"run either ({len(canonical_missing)} issue(s), first: "
            f"{canonical_missing[0]})"
        )

    required = judges or (
        FAMILY_MODELS[family]["weaker"],
        FAMILY_MODELS[family]["frontier"],
    )
    return None, (
        f"{note}.\n"
        f"PGR needs a same-judge blind and oracle baseline for every debate "
        f"arm, covering both '{required[0]}' and '{required[1]}'"
        + (f", over at least {min_rows} case(s)" if min_rows else "")
        + ".\n"
        f"Generate them first:\n"
        f"    ./scripts/run_medical_baselines.sh <N> <dir> {family}\n"
        f"then pass that directory as the 4th argument to "
        f"run_medical_debate.sh."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--weaker",
        default=None,
        help="Weaker judge for this run (FRONTIER/WEAKER override).",
    )
    parser.add_argument("--frontier", default=None, help="Frontier judge for this run.")
    parser.add_argument(
        "--min-rows",
        type=int,
        default=None,
        help="Require each baseline file to cover at least this many cases.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help="Preferred baselines directory; used if it is complete.",
    )
    args = parser.parse_args()

    judges = None
    if args.weaker and args.frontier:
        judges = (args.weaker, args.frontier)
    resolved, explanation = resolve(
        args.family, args.candidate, judges=judges, min_rows=args.min_rows
    )
    if resolved is None:
        print(explanation, file=sys.stderr)
        return 1
    print(explanation, file=sys.stderr)
    print(str(resolved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
