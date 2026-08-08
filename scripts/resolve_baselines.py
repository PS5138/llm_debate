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
)


def resolve(family: str, candidate: Path | None) -> tuple[Path | None, str]:
    """Return (resolved_dir, human-readable explanation)."""
    if family not in FAMILY_MODELS:
        return None, f"unknown family: {family}"

    if candidate is not None and candidate.exists():
        missing = missing_baseline_files(candidate, family)
        if not missing:
            return candidate, f"using baselines in {candidate}"
        shortfall = f"{len(missing)} file(s) missing, first: {missing[0]}"
        note = f"{candidate} is not a complete baseline set ({shortfall})"
    elif candidate is not None:
        note = f"{candidate} does not exist"
    else:
        note = "no candidate directory given"

    canonical = canonical_baselines_path(family)
    if canonical is not None:
        return canonical, f"{note}; falling back to canonical baselines at {canonical}"

    judges = FAMILY_MODELS[family]
    return None, (
        f"{note}, and no complete canonical baselines for '{family}' either.\n"
        f"PGR needs a same-judge blind and oracle baseline for every debate "
        f"arm, covering both '{judges['weaker']}' and '{judges['frontier']}'.\n"
        f"Generate them first:\n"
        f"    ./scripts/run_medical_baselines.sh <N> <dir> {family}\n"
        f"then pass that directory as the 4th argument to "
        f"run_medical_debate.sh."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help="Preferred baselines directory; used if it is complete.",
    )
    args = parser.parse_args()

    resolved, explanation = resolve(args.family, args.candidate)
    if resolved is None:
        print(explanation, file=sys.stderr)
        return 1
    print(explanation, file=sys.stderr)
    print(str(resolved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
