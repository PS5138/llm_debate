"""Regression tests for family-aware, path-portable Streamlit resume."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import runner  # noqa: E402


def _write_root_metadata(
    run_dir: Path,
    families: dict[str, dict],
    updated: str | None = None,
    recorded_run_root: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for family, record in families.items():
        if family in runner.FAMILY_MODELS:
            record.setdefault(
                "frontier_model", runner.FAMILY_MODELS[family]["frontier"]
            )
            record.setdefault("weaker_model", runner.FAMILY_MODELS[family]["weaker"])
            record.setdefault("concession_model", "gpt-4o-mini")
    doc = {"families": families}
    if updated is not None:
        doc["last_updated_at_utc"] = updated
    if recorded_run_root is not None:
        doc["run_root"] = recorded_run_root
    (run_dir / "run_metadata.json").write_text(
        json.dumps(doc, indent=2) + "\n", encoding="utf-8"
    )


def _family_record(family: str, timestamp: str, n_cases: int = 1) -> dict:
    return {
        "family": family,
        "n_cases": n_cases,
        "frontier_model": runner.FAMILY_MODELS[family]["frontier"],
        "weaker_model": runner.FAMILY_MODELS[family]["weaker"],
        "concession_model": "gpt-4o-mini",
        "recorded_at_utc": timestamp,
        # Boolean-only legacy metadata identifies the app-owned per-run path.
        # A bare ``baselines/<family>`` string is repository-relative metadata,
        # not a path relative to this results folder.
        "using_canonical_baselines": False,
    }


def _check(description: str, condition: bool) -> bool:
    print(f"{'ok   ' if condition else 'FAIL '} [{description}]")
    return condition


def _write_judgements(path: Path, n_cases: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": runner.pilot_case_ids(n_cases),
            "answer_judge": ["Answer: A"] * n_cases,
            "complete_judge": [True] * n_cases,
        }
    ).to_csv(path, index=False)


def _write_baseline_bundle(
    path: Path,
    family: str,
    frontier: str,
    weaker: str,
    n_cases: int = 1,
) -> None:
    for arm in ("baseline_blind", "baseline_oracle"):
        for judge in dict.fromkeys((weaker, frontier)):
            for filename in ("data0_judgement.csv", "data0_swap_judgement.csv"):
                _write_judgements(path / arm / judge / filename, n_cases)


def _write_family_outputs(
    run_dir: Path,
    family: str,
    frontier: str,
    weaker: str,
    n_cases: int = 1,
) -> None:
    judges = {
        "e1_info_asymmetry": frontier,
        "e2_double_asymmetry": weaker,
        "e3_capability_asymmetry": weaker,
        "e4_full_symmetry": frontier,
    }
    for condition, judge in judges.items():
        for filename in ("data0_judgement.csv", "data0_swap_judgement.csv"):
            _write_judgements(
                run_dir / family / "debate_sim" / f"{condition}_{judge}" / filename,
                n_cases,
            )


def main() -> int:
    results: list[bool] = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        newest_openai = base / "2026-01-01_00-00-00_results"
        _write_root_metadata(
            newest_openai,
            {"openai": _family_record("openai", "2026-01-03T00:00:00Z", 3)},
        )
        # Written later (newer directory mtime) but carrying an older saved
        # timestamp. Metadata time, not filesystem churn, should win.
        older_openai = base / "2026-01-02_00-00-00_results"
        _write_root_metadata(
            older_openai,
            {"openai": _family_record("openai", "2026-01-01T00:00:00Z", 2)},
        )
        mixed = base / "2026-01-03_00-00-00_results"
        _write_root_metadata(
            mixed,
            {
                "openai": _family_record("openai", "2026-01-02T00:00:00Z"),
                "anthropic": _family_record("anthropic", "2026-01-04T00:00:00Z", 4),
            },
        )

        selected_openai = runner.find_latest_run(base, family="openai")
        selected_anthropic = runner.find_latest_run(base, family="anthropic")
        results.append(
            _check(
                "OpenAI selector gets latest saved OpenAI run",
                selected_openai is not None
                and selected_openai["exp_dir"] == newest_openai
                and selected_openai["n_cases"] == 3,
            )
        )
        results.append(
            _check(
                "Anthropic selector gets Anthropic from multi-family metadata",
                selected_anthropic is not None
                and selected_anthropic["exp_dir"] == mixed
                and selected_anthropic["family"] == "anthropic"
                and selected_anthropic["n_cases"] == 4,
            )
        )
        results.append(
            _check(
                "unknown family has no resume target",
                runner.find_latest_run(base, family="unknown") is None,
            )
        )

        collision = base / "2026-01-04_00-00-00_results_2"
        _write_root_metadata(
            collision,
            {"openai": _family_record("openai", "2026-01-05T00:00:00Z", 5)},
        )
        results.append(
            _check(
                "timestamp collision suffix remains resumable",
                runner.find_latest_run(base, family="openai")["exp_dir"] == collision,
            )
        )

        invalid = base / "2026-01-06_00-00-00_results"
        invalid_record = _family_record("openai", "2026-01-06T00:00:00Z")
        invalid_record["n_cases"] = "not-a-number"
        _write_root_metadata(invalid, {"openai": invalid_record})
        results.append(
            _check(
                "invalid saved metadata is skipped",
                runner.find_latest_run(base, family="openai")["exp_dir"] == collision,
            )
        )

        invalid_shape = base / "2026-01-07_00-00-00_results"
        invalid_shape.mkdir()
        (invalid_shape / "run_metadata.json").write_text(
            json.dumps({"families": "openai"}) + "\n", encoding="utf-8"
        )
        results.append(
            _check(
                "invalid metadata container is skipped",
                runner.find_latest_run(base, family="openai")["exp_dir"] == collision,
            )
        )

        legacy_run = base / "2026-02-01_00-00-00_results"
        old_run_root = Path("/old/checkout/exp") / legacy_run.name
        legacy_canonical = {
            "baselines_dir": ("/old/checkout/data/canonical_baselines/openai")
        }
        canonical_path, canonical_flag = runner._baseline_source_from_record(
            legacy_canonical,
            legacy_run,
            "openai",
            recorded_run_root=old_run_root,
        )
        results.append(
            _check(
                "legacy canonical path rebases to current checkout",
                canonical_flag is True
                and canonical_path == runner.CANONICAL_BASELINES_ROOT / "openai",
            )
        )

        legacy_per_run = {
            "baselines_dir": (
                f"/old/checkout/exp/{legacy_run.name}/baselines/anthropic"
            )
        }
        per_run_path, per_run_flag = runner._baseline_source_from_record(
            legacy_per_run,
            legacy_run,
            "anthropic",
            recorded_run_root=old_run_root,
        )
        results.append(
            _check(
                "legacy per-run path rebases to selected run",
                per_run_flag is False
                and per_run_path == legacy_run / "baselines" / "anthropic",
            )
        )

        (
            canonical_false_path,
            canonical_false_flag,
        ) = runner._baseline_source_from_record(
            {
                "baselines_dir": str(runner.CANONICAL_BASELINES_ROOT / "openai"),
                "using_canonical_baselines": False,
            },
            legacy_run,
            "openai",
        )
        results.append(
            _check(
                "contradictory canonical path/source flag is rejected",
                canonical_false_flag is None and canonical_false_path is None,
            )
        )

        external_collision = (
            Path("/mnt/archive") / legacy_run.name / "baselines" / "anthropic"
        )
        external_path, external_flag = runner._baseline_source_from_record(
            {
                "baselines_dir": str(external_collision),
                "using_canonical_baselines": False,
            },
            legacy_run,
            "anthropic",
            recorded_run_root=old_run_root,
        )
        results.append(
            _check(
                "external suffix collision is not silently rebased",
                external_flag is False and external_path == external_collision,
            )
        )

        alternate_old_root = Path("/alternate/checkout/exp") / legacy_run.name
        alternate_path, alternate_flag = runner._baseline_source_from_record(
            {
                "family_dir": str(alternate_old_root / "anthropic"),
                "baselines_dir": str(alternate_old_root / "baselines" / "anthropic"),
                "using_canonical_baselines": False,
            },
            legacy_run,
            "anthropic",
            recorded_run_root=old_run_root,
        )
        results.append(
            _check(
                "all metadata-derived old roots are considered",
                alternate_flag is False
                and alternate_path == legacy_run / "baselines" / "anthropic",
            )
        )

        resume_root = base / "resume"
        resume_family = resume_root / "openai"
        resume_family.mkdir(parents=True)
        old_canonical = "/old/checkout/data/canonical_baselines/openai"
        (resume_family / "run_metadata.json").write_text(
            json.dumps(
                {
                    "family": "openai",
                    "n_cases": 1,
                    "baselines_dir": old_canonical,
                    "using_canonical_baselines": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        _write_root_metadata(
            resume_root,
            {
                "openai": {
                    "family": "openai",
                    "n_cases": 1,
                    "baselines_dir": old_canonical,
                    "using_canonical_baselines": True,
                }
            },
            recorded_run_root=f"/old/checkout/exp/{resume_root.name}",
        )
        resumed = runner.DebateRun(
            "openai",
            1,
            "test-openai-key",
            "",
            resume_from=resume_root,
        )
        try:
            healed_family = json.loads(
                (resume_family / "run_metadata.json").read_text(encoding="utf-8")
            )
            healed_root = json.loads(
                (resume_root / "run_metadata.json").read_text(encoding="utf-8")
            )
            results.append(
                _check(
                    "resume heals a moved canonical path",
                    resumed.using_canonical_baselines
                    and resumed.baselines_dir
                    == runner.CANONICAL_BASELINES_ROOT / "openai"
                    and healed_family["baselines_dir"]
                    == "data/canonical_baselines/openai"
                    and healed_root["families"]["openai"]["baselines_dir"]
                    == "data/canonical_baselines/openai",
                )
            )
        finally:
            resumed.cleanup()

        root_only = base / "root-only-resume"
        root_only_family = root_only / "openai"
        root_only_family.mkdir(parents=True)
        (root_only_family / "run_metadata.json").write_text("{}\n", encoding="utf-8")
        _write_root_metadata(
            root_only,
            {
                "openai": {
                    "family": "openai",
                    "n_cases": 1,
                    "baselines_dir": old_canonical,
                    "using_canonical_baselines": True,
                }
            },
            recorded_run_root=f"/old/checkout/exp/{root_only.name}",
        )
        root_resumed = runner.DebateRun(
            "openai",
            1,
            "test-openai-key",
            "",
            resume_from=root_only,
        )
        try:
            healed_root_only_family = json.loads(
                (root_only_family / "run_metadata.json").read_text(encoding="utf-8")
            )
            healed_root_only_root = json.loads(
                (root_only / "run_metadata.json").read_text(encoding="utf-8")
            )["families"]["openai"]
            results.append(
                _check(
                    "partial metadata is fully merged and healed in both copies",
                    root_resumed.using_canonical_baselines
                    and root_resumed.baselines_dir
                    == runner.CANONICAL_BASELINES_ROOT / "openai"
                    and all(
                        healed_root_only_family[key] == healed_root_only_root[key]
                        for key in (
                            "family",
                            "n_cases",
                            "frontier_model",
                            "weaker_model",
                            "concession_model",
                            "baselines_dir",
                            "using_canonical_baselines",
                        )
                    ),
                )
            )
        finally:
            root_resumed.cleanup()

        conflicting = base / "2026-04-01_00-00-00_results"
        conflicting_family = conflicting / "openai"
        conflicting_family.mkdir(parents=True)
        (conflicting_family / "run_metadata.json").write_text(
            json.dumps(
                {
                    "family": "openai",
                    "n_cases": 1,
                    "recorded_at_utc": "2026-04-01T00:00:00Z",
                    "baselines_dir": str(conflicting / "baselines" / "openai"),
                    "using_canonical_baselines": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        _write_root_metadata(
            conflicting,
            {
                "openai": {
                    "family": "openai",
                    "n_cases": 1,
                    "recorded_at_utc": "2026-04-02T00:00:00Z",
                    "baselines_dir": "data/canonical_baselines/openai",
                    "using_canonical_baselines": True,
                }
            },
            updated="2026-04-02T00:00:00Z",
        )
        discovered_conflict = runner.find_latest_run(base, family="openai")
        conflict_resumed = runner.DebateRun(
            "openai",
            1,
            "test-openai-key",
            "",
            resume_from=conflicting,
        )
        try:
            results.append(
                _check(
                    "discovery and construction use the same newer metadata",
                    discovered_conflict is not None
                    and discovered_conflict["exp_dir"] == conflicting
                    and discovered_conflict["using_canonical_baselines"] is True
                    and conflict_resumed.using_canonical_baselines is True
                    and conflict_resumed.baselines_dir
                    == runner.CANONICAL_BASELINES_ROOT / "openai",
                )
            )
        finally:
            conflict_resumed.cleanup()

        partial_upgrade = base / "2026-04-03_00-00-00_results"
        partial_family = partial_upgrade / "openai"
        partial_family.mkdir(parents=True)
        partial_per_run = partial_upgrade / "baselines" / "openai"
        (partial_family / "run_metadata.json").write_text(
            json.dumps(
                {
                    "family": "openai",
                    "n_cases": 1,
                    "recorded_at_utc": "2026-04-04T00:00:00Z",
                    "baselines_dir": str(partial_per_run),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        _write_root_metadata(
            partial_upgrade,
            {
                "openai": {
                    "family": "openai",
                    "n_cases": 1,
                    "recorded_at_utc": "2026-04-03T00:00:00Z",
                    "baselines_dir": "data/canonical_baselines/openai",
                    "using_canonical_baselines": True,
                }
            },
            updated="2026-04-03T00:00:00Z",
        )
        partial_record, partial_old_root, _ = runner._merged_family_record(
            partial_upgrade, "openai"
        )
        partial_path, partial_flag = runner._baseline_source_from_record(
            partial_record,
            partial_upgrade,
            "openai",
            recorded_run_root=partial_old_root,
        )
        results.append(
            _check(
                "newer baseline path does not inherit an older source flag",
                partial_flag is False and partial_path == partial_per_run,
            )
        )

        family_only_base = base / "family-only-scan"
        family_only = family_only_base / "2026-04-05_00-00-00_results"
        family_only_dir = family_only / "openai"
        family_only_dir.mkdir(parents=True)
        family_only_baselines = family_only / "baselines" / "openai"
        (family_only_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "family": "openai",
                    "n_cases": 1,
                    "frontier_model": runner.FAMILY_MODELS["openai"]["frontier"],
                    "weaker_model": runner.FAMILY_MODELS["openai"]["weaker"],
                    "concession_model": "gpt-4o-mini",
                    "recorded_at_utc": "2026-04-05T00:00:00Z",
                    "family_dir": str(family_only_dir),
                    "baselines_dir": str(family_only_baselines),
                    "using_canonical_baselines": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        discovered_family_only = runner.find_latest_run(
            family_only_base, family="openai"
        )
        family_only_resumed = runner.DebateRun(
            "openai",
            1,
            "test-openai-key",
            "",
            resume_from=family_only,
        )
        try:
            healed_family_only_root = json.loads(
                (family_only / "run_metadata.json").read_text(encoding="utf-8")
            )
            results.append(
                _check(
                    "family-only metadata is discoverable and heals root metadata",
                    discovered_family_only is not None
                    and discovered_family_only["exp_dir"] == family_only
                    and family_only_resumed.baselines_dir.resolve()
                    == family_only_baselines.resolve()
                    and healed_family_only_root["families"]["openai"]["n_cases"] == 1,
                )
            )
        finally:
            family_only_resumed.cleanup()

        missing_external = base / "missing-external-baselines"
        external_resume = base / "external-resume"
        external_family = external_resume / "openai"
        external_family.mkdir(parents=True)
        (external_family / "run_metadata.json").write_text(
            json.dumps(
                {
                    "family": "openai",
                    "n_cases": 1,
                    "frontier_model": runner.FAMILY_MODELS["openai"]["frontier"],
                    "weaker_model": runner.FAMILY_MODELS["openai"]["weaker"],
                    "concession_model": "gpt-4o-mini",
                    "baselines_dir": str(missing_external),
                    "using_canonical_baselines": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            runner.DebateRun(
                "openai",
                1,
                "test-openai-key",
                "",
                resume_from=external_resume,
            )
            external_rejected = False
        except ValueError as exc:
            external_rejected = "external baselines" in str(exc)
        results.append(
            _check(
                "missing external baseline path fails closed",
                external_rejected and not missing_external.exists(),
            )
        )

        fallback_base = base / "read-only-baseline-fallback"
        older_valid = fallback_base / "2026-05-01_00-00-00_results"
        _write_root_metadata(
            older_valid,
            {"openai": _family_record("openai", "2026-05-01T00:00:00Z", 1)},
        )
        newer_missing = fallback_base / "2026-05-02_00-00-00_results"
        newer_missing_record = _family_record("openai", "2026-05-02T00:00:00Z", 1)
        newer_missing_record["baselines_dir"] = str(base / "missing-read-only-archive")
        _write_root_metadata(newer_missing, {"openai": newer_missing_record})

        invalid_archive = base / "invalid-read-only-archive"
        _write_baseline_bundle(
            invalid_archive,
            "openai",
            runner.FAMILY_MODELS["openai"]["frontier"],
            runner.FAMILY_MODELS["openai"]["weaker"],
        )
        invalid_file = (
            invalid_archive
            / "baseline_blind"
            / runner.FAMILY_MODELS["openai"]["weaker"]
            / "data0_judgement.csv"
        )
        invalid_frame = pd.read_csv(invalid_file)
        invalid_frame.loc[0, "complete_judge"] = False
        invalid_frame.to_csv(invalid_file, index=False)
        newest_invalid = fallback_base / "2026-05-03_00-00-00_results"
        newest_invalid_record = _family_record("openai", "2026-05-03T00:00:00Z", 1)
        newest_invalid_record["baselines_dir"] = str(invalid_archive)
        _write_root_metadata(newest_invalid, {"openai": newest_invalid_record})

        selected_fallback = runner.find_latest_run(fallback_base, family="openai")
        results.append(
            _check(
                "selector skips missing/invalid read-only baselines for older valid run",
                selected_fallback is not None
                and selected_fallback["exp_dir"] == older_valid
                and selected_fallback["baselines_writable"] is True,
            )
        )

        # A newly touched empty family copy must not outrank the root record's
        # explicit timestamp and make an older run look newest.
        mtime_base = base / "mtime-selection"
        genuinely_new = mtime_base / "2026-06-02_00-00-00_results"
        stale_but_touched = mtime_base / "2026-06-01_00-00-00_results"
        _write_root_metadata(
            genuinely_new,
            {"openai": _family_record("openai", "2026-06-02T00:00:00Z", 2)},
        )
        _write_root_metadata(
            stale_but_touched,
            {"openai": _family_record("openai", "2026-06-01T00:00:00Z", 1)},
        )
        stale_family_meta = stale_but_touched / "openai" / "run_metadata.json"
        stale_family_meta.parent.mkdir()
        stale_family_meta.write_text("{}\n", encoding="utf-8")
        os.utime(stale_family_meta, (2_000_000_000, 2_000_000_000))
        results.append(
            _check(
                "empty family metadata mtime cannot beat a saved timestamp",
                runner.find_latest_run(mtime_base, family="openai")["exp_dir"]
                == genuinely_new,
            )
        )

        # A root-level time can be bumped by another family. It is not a
        # family-specific timestamp and cannot beat an explicit family copy.
        cross_family = base / "cross-family-time"
        cross_run = cross_family / "2026-06-03_00-00-00_results"
        root_openai = _family_record("openai", "2026-06-01T00:00:00Z", 1)
        root_openai.pop("recorded_at_utc")
        _write_root_metadata(
            cross_run,
            {"openai": root_openai},
            updated="2026-06-10T00:00:00Z",
        )
        cross_family_meta = cross_run / "openai" / "run_metadata.json"
        cross_family_meta.parent.mkdir()
        newer_family_copy = _family_record("openai", "2026-06-02T00:00:00Z", 2)
        cross_family_meta.write_text(
            json.dumps(newer_family_copy) + "\n", encoding="utf-8"
        )
        merged_cross, _, _ = runner._merged_family_record(cross_run, "openai")
        results.append(
            _check(
                "other-family root update cannot outrank family timestamp",
                merged_cross is not None and merged_cross["n_cases"] == 2,
            )
        )

        # If the root does not contain this family at all, its global update
        # time belongs to some other family and must not become this family's
        # discovery timestamp.
        orphan_copy = cross_family / "2026-06-03_00-00-01_results"
        _write_root_metadata(
            orphan_copy,
            {"anthropic": _family_record("anthropic", "2026-06-11T00:00:00Z")},
            updated="2026-06-11T00:00:00Z",
        )
        orphan_family_meta = orphan_copy / "openai" / "run_metadata.json"
        orphan_family_meta.parent.mkdir()
        timestampless_openai = _family_record("openai", "unused")
        timestampless_openai.pop("recorded_at_utc")
        orphan_family_meta.write_text(
            json.dumps(timestampless_openai) + "\n", encoding="utf-8"
        )
        family_mtime = 1_700_000_000
        os.utime(orphan_family_meta, (family_mtime, family_mtime))
        _, _, orphan_timestamp = runner._merged_family_record(orphan_copy, "openai")
        results.append(
            _check(
                "other-family root time is ignored when root family record is absent",
                orphan_timestamp == family_mtime,
            )
        )

        shared_legacy = cross_family / "shared-legacy-selection"
        genuinely_new_shared = shared_legacy / "2026-06-03_00-00-02_results"
        _write_root_metadata(
            genuinely_new_shared,
            {"openai": _family_record("openai", "2026-06-03T00:00:00Z", 2)},
        )
        stale_shared = shared_legacy / "2025-06-03_00-00-00_results"
        stale_openai = _family_record("openai", "unused", 1)
        stale_openai.pop("recorded_at_utc")
        _write_root_metadata(
            stale_shared,
            {
                "openai": stale_openai,
                "anthropic": _family_record("anthropic", "2030-06-03T00:00:00Z", 1),
            },
            updated="2030-06-03T00:00:00Z",
        )
        stale_shared_family = stale_shared / "openai" / "run_metadata.json"
        stale_shared_family.parent.mkdir()
        stale_shared_family.write_text(
            json.dumps(stale_openai) + "\n", encoding="utf-8"
        )
        stale_mtime = 1_750_000_000
        os.utime(stale_shared_family, (stale_mtime, stale_mtime))
        os.utime(stale_shared, (stale_mtime, stale_mtime))
        results.append(
            _check(
                "shared root update cannot become a timestampless family timestamp",
                runner.find_latest_run(shared_legacy, family="openai")["exp_dir"]
                == genuinely_new_shared,
            )
        )

        tie_base = base / "tie-selection"
        tie_first = tie_base / "2026-06-04_00-00-00_results"
        tie_second = tie_base / "2026-06-04_00-00-00_results_2"
        for target, count in ((tie_first, 1), (tie_second, 2)):
            _write_root_metadata(
                target,
                {"openai": _family_record("openai", "2026-06-04T00:00:00Z", count)},
            )
        results.append(
            _check(
                "same-second collision selection is deterministic",
                runner.find_latest_run(tie_base, family="openai")["exp_dir"]
                == tie_second,
            )
        )

        mismatch_base = base / "family-mismatch"
        mismatch_run = mismatch_base / "2026-06-05_00-00-00_results"
        wrong_family_record = _family_record("anthropic", "2026-06-05T00:00:00Z", 1)
        _write_root_metadata(mismatch_run, {"openai": wrong_family_record})
        results.append(
            _check(
                "explicit family mismatch is not resumable",
                runner.find_latest_run(mismatch_base, family="openai") is None,
            )
        )

        missing_models_base = base / "missing-models"
        missing_models_run = missing_models_base / "2026-06-06_00-00-00_results"
        missing_models_family = missing_models_run / "openai"
        missing_models_family.mkdir(parents=True)
        missing_models_record = {
            "family": "openai",
            "n_cases": 1,
            "recorded_at_utc": "2026-06-06T00:00:00Z",
            "baselines_dir": str(missing_models_run / "baselines" / "openai"),
            "using_canonical_baselines": False,
        }
        (missing_models_family / "run_metadata.json").write_text(
            json.dumps(missing_models_record) + "\n", encoding="utf-8"
        )
        try:
            runner.DebateRun(
                "openai",
                1,
                "test-openai-key",
                "",
                resume_from=missing_models_run,
            )
            missing_models_rejected = False
        except ValueError as exc:
            missing_models_rejected = "missing the recorded" in str(exc)
        results.append(
            _check(
                "missing saved models fail closed and are hidden",
                missing_models_rejected
                and runner.find_latest_run(missing_models_base, family="openai")
                is None,
            )
        )

        no_baseline_base = base / "missing-baseline-source"
        no_baseline_run = no_baseline_base / "2026-06-07_00-00-00_results"
        no_baseline_record = _family_record("openai", "2026-06-07T00:00:00Z", 1)
        no_baseline_record.pop("baselines_dir", None)
        no_baseline_record.pop("using_canonical_baselines")
        _write_root_metadata(no_baseline_run, {"openai": no_baseline_record})
        try:
            runner.DebateRun(
                "openai",
                1,
                "test-openai-key",
                "",
                resume_from=no_baseline_run,
            )
            no_baseline_rejected = False
        except ValueError as exc:
            no_baseline_rejected = "does not record a baseline source" in str(exc)
        results.append(
            _check(
                "missing saved baseline source fails closed and is hidden",
                no_baseline_rejected
                and runner.find_latest_run(no_baseline_base, family="openai") is None,
            )
        )

        custom_models = base / "unsupported-model-runs" / "2026-06-08_00-00-00_results"
        custom_family = custom_models / "openai"
        custom_family.mkdir(parents=True)
        custom_record = _family_record("openai", "2026-06-08T00:00:00Z", 1)
        custom_record["frontier_model"] = "saved-frontier"
        (custom_family / "run_metadata.json").write_text(
            json.dumps(custom_record) + "\n", encoding="utf-8"
        )
        try:
            runner.DebateRun(
                "openai",
                1,
                "test-openai-key",
                "",
                resume_from=custom_models,
            )
            unsupported_model_rejected = False
        except ValueError as exc:
            unsupported_model_rejected = "unsupported frontier model" in str(exc)
        results.append(
            _check(
                "unsupported saved model aliases fail closed and are hidden",
                unsupported_model_rejected
                and runner.find_latest_run(custom_models.parent, family="openai")
                is None,
            )
        )

        interpolation_run = base / "hydra-interpolation-resume"
        interpolation_family = interpolation_run / "openai"
        interpolation_family.mkdir(parents=True)
        interpolation_record = _family_record("openai", "2026-06-08T01:00:00Z", 1)
        interpolation_record["frontier_model"] = "${oc.env:ANTHROPIC_API_KEY}"
        (interpolation_family / "run_metadata.json").write_text(
            json.dumps(interpolation_record) + "\n", encoding="utf-8"
        )
        try:
            runner.DebateRun(
                "openai",
                1,
                "test-openai-key",
                "test-anthropic-key",
                resume_from=interpolation_run,
            )
            interpolation_rejected = False
        except ValueError as exc:
            interpolation_rejected = "unsafe Hydra or path syntax" in str(exc)
        results.append(
            _check(
                "saved Hydra interpolation is rejected before any stage",
                interpolation_rejected,
            )
        )

        valid_models_run = base / "saved-current-models"
        valid_models_family = valid_models_run / "anthropic"
        valid_models_family.mkdir(parents=True)
        valid_record = _family_record("anthropic", "2026-06-08T02:00:00Z", 1)
        valid_record["baselines_dir"] = str(
            valid_models_run / "baselines" / "anthropic"
        )
        (valid_models_family / "run_metadata.json").write_text(
            json.dumps(valid_record) + "\n", encoding="utf-8"
        )
        valid_resume = runner.DebateRun(
            "anthropic",
            1,
            "test-openai-key",
            "test-anthropic-key",
            resume_from=valid_models_run,
        )
        try:
            current_aliases_preserved = (
                valid_resume.frontier_model,
                valid_resume.weaker_model,
                valid_resume.concession_model,
            ) == (
                runner.FAMILY_MODELS["anthropic"]["frontier"],
                runner.FAMILY_MODELS["anthropic"]["weaker"],
                runner.CONCESSION_MODEL,
            )
        finally:
            valid_resume.cleanup()
        try:
            runner.DebateRun(
                "anthropic",
                1,
                "test-openai-key",
                "test-anthropic-key",
                resume_from=valid_models_run,
                frontier_model="different-frontier",
            )
            model_mismatch_rejected = False
        except ValueError as exc:
            model_mismatch_rejected = "does not match the saved run" in str(exc)
        results.append(
            _check(
                "current aliases resume and caller model mismatches fail closed",
                current_aliases_preserved and model_mismatch_rejected,
            )
        )

        missing_key_base = base / "fresh-missing-key"
        real_create_results_root = runner.create_results_root
        with patch.object(
            runner,
            "create_results_root",
            side_effect=lambda: real_create_results_root(missing_key_base),
        ):
            try:
                runner.DebateRun("openai", 1, "", "")
                missing_key_rejected_cleanly = False
            except ValueError as exc:
                missing_key_rejected_cleanly = "OpenAI API key is required" in str(exc)
        leftover_fresh_runs = (
            list(missing_key_base.iterdir()) if missing_key_base.exists() else []
        )
        results.append(
            _check(
                "missing fresh-run key leaves no empty results folder",
                missing_key_rejected_cleanly and not leftover_fresh_runs,
            )
        )

        external_complete_run = base / "external resume (quoted)"
        external_complete_family = external_complete_run / "openai"
        external_complete_family.mkdir(parents=True)
        external_archive = base / "archive baselines (read only)"
        default_openai = runner.FAMILY_MODELS["openai"]
        _write_baseline_bundle(
            external_archive,
            "openai",
            default_openai["frontier"],
            default_openai["weaker"],
        )
        external_complete_record = _family_record("openai", "2026-06-09T00:00:00Z", 1)
        external_complete_record["baselines_dir"] = str(external_archive)
        (external_complete_family / "run_metadata.json").write_text(
            json.dumps(external_complete_record) + "\n", encoding="utf-8"
        )
        archive_before = {
            path.relative_to(external_archive): path.read_bytes()
            for path in external_archive.rglob("*")
            if path.is_file()
        }
        external_complete = runner.DebateRun(
            "openai",
            1,
            "test-openai-key",
            "",
            resume_from=external_complete_run,
        )
        try:
            from hydra.core.override_parser.overrides_parser import OverridesParser

            hydra_value = external_complete._hydra_exp_dir(external_complete.family_dir)
            parsed_path = (
                OverridesParser.create()
                .parse_overrides([f"exp_dir={hydra_value}"])[0]
                .value()
            )
            archive_after = {
                path.relative_to(external_archive): path.read_bytes()
                for path in external_archive.rglob("*")
                if path.is_file()
            }
            external_safe = (
                external_complete.baselines_read_only
                and not external_complete.using_canonical_baselines
                and parsed_path == external_complete.family_dir.as_posix()
                and archive_after == archive_before
            )
        finally:
            external_complete.cleanup()
        results.append(
            _check(
                "complete external baselines stay read-only and Hydra paths parse",
                external_safe,
            )
        )

        symlink_resume = base / "symlinked-external-resume"
        symlink_family = symlink_resume / "openai"
        symlink_family.mkdir(parents=True)
        symlink_archive = base / "symlinked-baseline-archive"
        _write_baseline_bundle(
            symlink_archive,
            "openai",
            default_openai["frontier"],
            default_openai["weaker"],
            n_cases=2,
        )
        archived_working_csv = symlink_archive / "baseline_blind" / "data0.csv"
        pd.DataFrame({"id": [runner.pilot_case_ids(1)[0]], "complete": [True]}).to_csv(
            archived_working_csv, index=False
        )
        symlink_baselines = symlink_resume / "baselines" / "openai"
        symlink_baselines.parent.mkdir(parents=True)
        symlink_baselines.symlink_to(symlink_archive, target_is_directory=True)
        symlink_record = _family_record("openai", "2026-06-09T01:00:00Z", 1)
        symlink_record["baselines_dir"] = str(symlink_baselines)
        (symlink_family / "run_metadata.json").write_text(
            json.dumps(symlink_record) + "\n", encoding="utf-8"
        )
        archived_working_before = archived_working_csv.read_bytes()
        symlinked_external = runner.DebateRun(
            "openai",
            2,
            "test-openai-key",
            "",
            resume_from=symlink_resume,
        )
        try:
            symlink_archive_safe = (
                symlinked_external.baselines_read_only
                and archived_working_csv.read_bytes() == archived_working_before
                and len(pd.read_csv(archived_working_csv)) == 1
            )
        finally:
            symlinked_external.cleanup()
        results.append(
            _check(
                "symlinked external baselines are detected and never extended",
                symlink_archive_safe,
            )
        )

        nested_symlink_resume = base / "nested-symlink-resume"
        nested_symlink_family = nested_symlink_resume / "openai"
        nested_symlink_family.mkdir(parents=True)
        nested_baselines = nested_symlink_resume / "baselines" / "openai"
        _write_baseline_bundle(
            nested_baselines,
            "openai",
            default_openai["frontier"],
            default_openai["weaker"],
            n_cases=2,
        )
        nested_external_csv = base / "nested-external-data0.csv"
        pd.DataFrame({"id": [runner.pilot_case_ids(1)[0]], "complete": [True]}).to_csv(
            nested_external_csv, index=False
        )
        nested_working_link = nested_baselines / "baseline_blind" / "data0.csv"
        nested_working_link.symlink_to(nested_external_csv)
        nested_record = _family_record("openai", "2026-06-09T02:00:00Z", 1)
        nested_record["baselines_dir"] = str(nested_baselines)
        (nested_symlink_family / "run_metadata.json").write_text(
            json.dumps(nested_record) + "\n", encoding="utf-8"
        )
        nested_before = nested_external_csv.read_bytes()
        nested_resume = runner.DebateRun(
            "openai",
            2,
            "test-openai-key",
            "",
            resume_from=nested_symlink_resume,
        )
        try:
            nested_symlink_safe = (
                nested_resume.baselines_read_only
                and nested_external_csv.read_bytes() == nested_before
                and len(pd.read_csv(nested_external_csv)) == 1
            )
        finally:
            nested_resume.cleanup()
        results.append(
            _check(
                "nested baseline symlinks make the source read-only",
                nested_symlink_safe,
            )
        )

        combined_run = base / "combined-analysis"
        openai_models = runner.FAMILY_MODELS["openai"]
        anthropic_models = runner.FAMILY_MODELS["anthropic"]
        openai_baselines = combined_run / "baselines" / "openai"
        anthropic_baselines = base / "anthropic archive (external)"
        _write_baseline_bundle(
            openai_baselines,
            "openai",
            openai_models["frontier"],
            openai_models["weaker"],
        )
        _write_baseline_bundle(
            anthropic_baselines,
            "anthropic",
            anthropic_models["frontier"],
            anthropic_models["weaker"],
        )
        _write_family_outputs(
            combined_run,
            "openai",
            openai_models["frontier"],
            openai_models["weaker"],
        )
        _write_family_outputs(
            combined_run,
            "anthropic",
            anthropic_models["frontier"],
            anthropic_models["weaker"],
        )
        combined_openai = _family_record("openai", "2026-06-10T00:00:00Z", 1)
        combined_openai["baselines_dir"] = str(openai_baselines)
        # Model an interrupted extension: metadata was raised to n=2 before
        # the sibling's raw outputs grew beyond its previously finished case.
        combined_anthropic = _family_record("anthropic", "2026-06-10T00:00:00Z", 2)
        combined_anthropic["baselines_dir"] = str(anthropic_baselines)
        _write_root_metadata(
            combined_run,
            {"openai": combined_openai, "anthropic": combined_anthropic},
        )
        combined_resume = runner.DebateRun(
            "openai",
            1,
            "test-openai-key",
            "",
            resume_from=combined_run,
        )
        try:
            (
                combined_dirs,
                combined_baselines_view,
            ) = combined_resume._combined_analysis_inputs()
            combined_ok = (
                [path.name for path in combined_dirs] == ["openai", "anthropic"]
                and (combined_baselines_view / "openai").resolve()
                == openai_baselines.resolve()
                and (combined_baselines_view / "anthropic").resolve()
                == anthropic_baselines.resolve()
            )
            from scripts.aggregate_medical_results import main as aggregate_results
            from scripts.compute_paired_tests import main as compute_paired_tests

            verification_out = combined_run / "verification_results"
            aggregate_results(
                combined_dirs,
                combined_baselines_view,
                verification_out,
                plots_only=False,
            )
            compute_paired_tests(
                combined_dirs,
                combined_baselines_view,
                verification_out,
                analysis_dir=None,
            )
            aggregate_families = set(
                pd.read_csv(verification_out / "accuracy_by_condition.csv")["family"]
            )
            headline_families = set(
                pd.read_csv(verification_out / "headline_results.csv")["family"]
            )
            combined_ok = combined_ok and aggregate_families == headline_families == {
                "openai",
                "anthropic",
            }
        finally:
            combined_resume.cleanup()
        results.append(
            _check(
                "combined analysis preserves interrupted sibling and each baseline source",
                combined_ok,
            )
        )

    failures = len([result for result in results if not result])
    print()
    if failures:
        print(f"{failures} regression(s) detected.")
        return 1
    print(f"{len(results)} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
