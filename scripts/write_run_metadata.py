"""Write metadata for a medical debate run.

This is deliberately tiny and dependency-free so shell wrappers can record
which model family and settings produced each timestamped results folder.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--family", required=True)
    parser.add_argument("--n-cases", required=True, type=int)
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--weaker", required=True)
    parser.add_argument("--concession-model", required=True)
    parser.add_argument("--family-dir", required=True, type=Path)
    parser.add_argument("--baselines-dir", required=True, type=Path)
    parser.add_argument("--entrypoint", default="script")
    args = parser.parse_args()

    args.run_root.mkdir(parents=True, exist_ok=True)
    args.family_dir.mkdir(parents=True, exist_ok=True)

    family_record = {
        "family": args.family,
        "n_cases": args.n_cases,
        "frontier_model": args.frontier,
        "weaker_model": args.weaker,
        "concession_model": args.concession_model,
        "entrypoint": args.entrypoint,
        "family_dir": str(args.family_dir),
        "baselines_dir": str(args.baselines_dir),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    family_metadata = args.family_dir / "run_metadata.json"
    family_metadata.write_text(
        json.dumps(family_record, indent=2) + "\n",
        encoding="utf-8",
    )

    root_metadata = args.run_root / "run_metadata.json"
    root_record = load_json(root_metadata)
    root_record.setdefault("run_root", str(args.run_root))
    root_record.setdefault("created_or_updated_at_utc", family_record["recorded_at_utc"])
    root_record["last_updated_at_utc"] = family_record["recorded_at_utc"]
    root_record.setdefault("families", {})
    root_record["families"][args.family] = family_record
    root_metadata.write_text(json.dumps(root_record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
