"""Pipeline runner for the Streamlit demo.

Each `DebateRun` owns a timestamped results directory under `exp/`, plus
a tempdir for session-scoped secrets and transient logs. A background
thread drives the medical debate pipeline stage by stage. Progress is
exposed through `snapshot()` so the Streamlit UI can render a live status
table without holding the subprocess.

Experiment outputs persist under `exp/YYYY-MM-DD_HH-MM-SS_results/`.
`cleanup()` removes only the tempdir and the session SECRETS file; the
keys never touch the user's repo SECRETS.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

from core.scoring.accuracy import find_answer

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_CSV = REPO_ROOT / "data" / "ddxplus" / "ddxplus_debate_pilot_100.csv"
CANONICAL_BASELINES_ROOT = REPO_ROOT / "data" / "canonical_baselines"


FAMILY_MODELS = {
    "openai": {"frontier": "gpt-5.5", "weaker": "gpt-5.4-mini"},
    "anthropic": {"frontier": "claude-opus-4-6", "weaker": "claude-sonnet-4-6"},
}
CONCESSION_MODEL = "gpt-4o-mini"

# Model values are interpolated into Hydra overrides and also become judgement
# directory names.  This app intentionally exposes one fixed, reproducible
# model trio per family, so saved metadata must never be treated as a free-form
# Hydra expression or path component.
_SAFE_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def missing_baseline_files(
    path: Path,
    family: str,
    judges: tuple[str, str] | None = None,
    min_rows: int | None = None,
    expected_ids: tuple[str, ...] | None = None,
) -> list[str]:
    """Return the baseline judgement files unusable in `path` for `family`.

    An empty list means the directory is fully populated for BOTH judges.

    `judges` overrides the family defaults, so a run using FRONTIER/WEAKER
    overrides validates the baselines it will actually pair against rather
    than the ones the defaults imply. `min_rows` additionally requires each
    file to hold at least that many completed, parseable judgements. When
    `expected_ids` is supplied, those exact cases must provide the judgements;
    unrelated rows cannot make up the requested count.

    Required layout (per judge role for this family):
      <path>/baseline_blind/<weaker>/data0_judgement.csv
      <path>/baseline_blind/<weaker>/data0_swap_judgement.csv
      <path>/baseline_blind/<frontier>/data0_judgement.csv
      <path>/baseline_blind/<frontier>/data0_swap_judgement.csv
      <path>/baseline_oracle/<weaker>/data0_judgement.csv
      <path>/baseline_oracle/<weaker>/data0_swap_judgement.csv
      <path>/baseline_oracle/<frontier>/data0_judgement.csv
      <path>/baseline_oracle/<frontier>/data0_swap_judgement.csv

    Same-judge PGR pairing (E1/E4 ↔ frontier baselines, E2/E3 ↔ weaker
    baselines) means a folder missing either judge would silently produce
    NaN PGR for half the debate arms, so both judges are required by name.

    This does not detect stale baselines generated with the right case IDs and
    judges but an older prompt. Freshness is governed by the regeneration
    rules in `data/canonical_baselines/README.md`.
    """
    if family not in FAMILY_MODELS:
        return [f"unknown family: {family}"]
    required_judges = judges or (
        FAMILY_MODELS[family]["weaker"],
        FAMILY_MODELS[family]["frontier"],
    )
    missing: list[str] = []
    for arm in ("baseline_blind", "baseline_oracle"):
        for judge in required_judges:
            for fname in ("data0_judgement.csv", "data0_swap_judgement.csv"):
                rel = Path(arm) / judge / fname
                target = path / rel
                if not target.exists():
                    missing.append(rel.as_posix())
                    continue
                issue = _baseline_csv_issue(
                    target, min_rows=min_rows, expected_ids=expected_ids
                )
                if issue is not None:
                    missing.append(f"{rel.as_posix()} ({issue})")
    return missing


def pilot_case_ids(n_cases: int) -> tuple[str, ...]:
    """Return the first `n_cases` IDs from the prepared medical pilot.

    An empty or short tuple is intentionally returned when the pilot cannot be
    read. Passing that result into baseline validation fails closed instead of
    silently accepting rows from an unknown dataset.
    """
    if n_cases < 1:
        return ()
    try:
        pilot = pd.read_csv(
            PILOT_CSV,
            usecols=["case_id"],
            dtype={"case_id": str},
            keep_default_na=False,
        )
    except Exception:
        return ()
    return tuple(pilot["case_id"].astype(str).str.strip().head(n_cases))


def _judgement_is_parseable(value: object) -> bool:
    """Whether the analysis will be able to score this judgement.

    This must agree with `_judge_correctness` in
    `scripts/aggregate_medical_results.py`, which is what actually produces
    the per-case scores: exactly one of A/B found means scoreable, anything
    else becomes NaN and is dropped.

    `func_correct_ab` is deliberately NOT used here. It maps an explicitly
    inconclusive answer ("Answer: Inconclusive", "Answer: Unclear", ...) to
    False rather than "Unknown", so a baseline full of inconclusive verdicts
    would pass validation and then score as all-NaN downstream — precisely
    the silent failure this preflight exists to catch.
    """
    text = str(value).strip()
    lowered = text.lower()
    if not text or "error message:" in lowered or "error occurred" in lowered:
        return False
    try:
        found_a = bool(find_answer(text, "A"))
        found_b = bool(find_answer(text, "B"))
    except (IndexError, TypeError, ValueError):
        return False
    return found_a != found_b


def arm_judgement_is_complete(answers: object) -> bool:
    """Whether both orderings of one live arm are finished and scoreable."""
    if not isinstance(answers, dict):
        return False
    return (
        answers.get("orig_complete") is True
        and answers.get("swap_complete") is True
        and _judgement_is_parseable(answers.get("orig", ""))
        and _judgement_is_parseable(answers.get("swap", ""))
    )


def _baseline_csv_issue(
    path: Path,
    min_rows: int | None = None,
    expected_ids: tuple[str, ...] | None = None,
) -> Optional[str]:
    """Return why one baseline CSV is unusable, or None when it is sound.

    A usable requested row has a unique, non-empty case ID, an explicitly true
    completion flag, and a judgement the production accuracy parser accepts.
    """
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except Exception:
        return "unreadable"
    if df.empty:
        return "empty"

    id_column = (
        "id" if "id" in df.columns else "case_id" if "case_id" in df.columns else None
    )
    if id_column is None:
        return "missing id column"
    ids = df[id_column].astype(str).str.strip()
    blank_ids = int(ids.eq("").sum())
    if blank_ids:
        return f"{blank_ids} blank case ID(s)"
    duplicate_ids = sorted(set(ids[ids.duplicated(keep=False)]))
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:3])
        return f"duplicate case ID(s): {preview}"

    answer_columns = [c for c in df.columns if c.startswith("answer_judge")]
    if "answer_judge" in answer_columns:
        answer_columns.remove("answer_judge")
        answer_columns.insert(0, "answer_judge")
    pair = next(
        (
            (answer, f"complete_{answer.removeprefix('answer_')}")
            for answer in answer_columns
            if f"complete_{answer.removeprefix('answer_')}" in df.columns
        ),
        None,
    )
    if pair is None:
        if not answer_columns:
            return "missing judgement column"
        return "missing completion column for judgement"
    answer_column, complete_column = pair

    complete = df[complete_column].map(
        lambda value: value is True or str(value).strip().lower() == "true"
    )
    parseable = df[answer_column].map(_judgement_is_parseable)
    valid = complete & parseable

    if expected_ids is not None:
        expected = tuple(str(case_id).strip() for case_id in expected_ids)
        if not expected:
            return "no expected pilot case IDs available"
        if len(set(expected)) != len(expected):
            return "expected pilot case IDs are not unique"
        if min_rows is not None and len(expected) < min_rows:
            return f"only {len(expected)} expected pilot case ID(s), need {min_rows}"
        missing_ids = [case_id for case_id in expected if case_id not in set(ids)]
        if missing_ids:
            preview = ", ".join(missing_ids[:3])
            return f"missing expected case ID(s): {preview}"
        required = ids.isin(expected)
        required_count = len(expected)
    else:
        required_count = len(df) if min_rows is None else min_rows
        if required_count < 1:
            return "minimum row count must be positive"
        if len(df) < required_count:
            return f"only {len(df)} case row(s), need {required_count}"
        required = pd.Series(False, index=df.index)
        required.iloc[:required_count] = True

    valid_count = int((valid & required).sum())
    if valid_count < required_count:
        incomplete_count = int((~complete & required).sum())
        unparseable_count = int((complete & ~parseable & required).sum())
        details = []
        if incomplete_count:
            details.append(f"{incomplete_count} incomplete")
        if unparseable_count:
            details.append(f"{unparseable_count} unparseable/error")
        suffix = f"; {', '.join(details)}" if details else ""
        return (
            f"only {valid_count}/{required_count} requested case(s) have completed, "
            f"parseable judgements{suffix}"
        )
    return None


def baselines_dir_is_complete(
    path: Path,
    family: str,
    judges: tuple[str, str] | None = None,
    min_rows: int | None = None,
    expected_ids: tuple[str, ...] | None = None,
) -> bool:
    """True if `path` holds a full set of baseline judgements for `family`."""
    return not missing_baseline_files(
        path,
        family,
        judges=judges,
        min_rows=min_rows,
        expected_ids=expected_ids,
    )


def canonical_baselines_path(
    family: str,
    judges: tuple[str, str] | None = None,
    min_rows: int | None = None,
    expected_ids: tuple[str, ...] | None = None,
) -> Optional[Path]:
    """Return the committed canonical baselines dir for a family if it has
    been *fully* populated for BOTH judges, else None.
    """
    if family not in FAMILY_MODELS:
        return None
    path = CANONICAL_BASELINES_ROOT / family
    if not (path / "results.csv").exists():
        return None
    return (
        path
        if baselines_dir_is_complete(
            path,
            family,
            judges=judges,
            min_rows=min_rows,
            expected_ids=expected_ids,
        )
        else None
    )


# Per-family concurrency caps, expressed as "how many cases can run in
# parallel at a time". Anthropic stays conservative because Opus has
# long per-call latency and lower-tier accounts have tight RPM limits;
# OpenAI can safely go wider since gpt-5.5 has no adaptive-thinking
# overhead and per-account RPM is more generous.
#
# The pipeline scales these into the underlying `anthropic_num_threads`
# Hydra setting differently for debate vs judge:
#   * debate-stage threads = CASES_AT_A_TIME × BoN  (because core.debate
#     divides by BoN internally to derive case concurrency)
#   * judge/score-stage threads = CASES_AT_A_TIME   (no BoN fan-out at
#     judge time, so it maps 1:1 to concurrent cases)
CASES_AT_A_TIME = {
    "openai": 5,
    "anthropic": 2,
}
BON = 4

# Per-case cost estimates in USD. The OpenAI number is calibrated against
# the published gpt-5.5 pricing × the BoN=4 × 3-round × 2-debater debate
# stage + 4 final-judge arms × 2 swap orderings + concession judging.
#
# The Anthropic number is for Opus 4.6, which has the same $5/$25 list
# price as 4.7 but no adaptive-thinking overhead. A 4.7 smoke came in at
# ~$4/case end-to-end; 4.6 should land lower because output tokens aren't
# inflated by hidden reasoning, but the exact number won't be known until
# a 4.6 smoke completes. Treat this as a ballpark only.
COST_PER_CASE_USD = {
    "openai": 2.50,
    "anthropic": 2.50,
}

# Extra planning allowance when blind + oracle baselines are deliberately
# recomputed instead of reusing a validated bundle. OpenAI uses the documented
# ~$2/100-case upper ballpark. Anthropic is intentionally more conservative
# until a full Opus + Sonnet baseline pass has been calibrated locally.
BASELINE_RECOMPUTE_COST_PER_CASE_USD = {
    "openai": 0.02,
    "anthropic": 0.50,
}

CONDITIONS = [
    "e1_info_asymmetry",
    "e2_double_asymmetry",
    "e3_capability_asymmetry",
    "e4_full_symmetry",
]


def create_results_root(base_dir: Path = REPO_ROOT / "exp") -> Path:
    """Create a fresh timestamped results directory under a trusted base.

    The default app path is repository-owned.  Reject links in that relative
    path before creating anything so a replaced ``exp`` directory cannot
    redirect a fresh run outside the checkout.  Custom bases (used by tests
    and embedders) still reject a symlink at the base itself.
    """
    if _path_has_unsafe_hydra_syntax(base_dir):
        raise ValueError(f"unsafe results base path: {base_dir}")
    trusted_root = _trusted_output_anchor(base_dir)
    issue = _symlink_component_issue(base_dir, trusted_root)
    if issue is not None:
        raise ValueError(f"unsafe results base path: {issue}")
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    candidate = base_dir / f"{stamp}_results"
    suffix = 2
    while candidate.exists():
        candidate = base_dir / f"{stamp}_results_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    if candidate.is_symlink() or not _path_is_within(candidate, base_dir):
        try:
            candidate.rmdir()
        except OSError:
            pass
        raise ValueError(f"unsafe fresh results path: {candidate}")
    return candidate


def _portable_metadata_path(path: Path) -> str:
    """Store in-repository paths relative to the repo so runs survive moves."""
    absolute = path.resolve()
    try:
        return absolute.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(absolute)


def _resolve_metadata_path(value: object) -> Optional[Path]:
    """Resolve a path read from metadata written before or after portability."""
    if value is None or not str(value).strip():
        return None
    raw = str(value).strip()
    path = Path(raw)
    if _path_has_unsafe_hydra_syntax(path):
        return None
    if not path.is_absolute() and ".." in path.parts:
        return None
    return path if path.is_absolute() else REPO_ROOT / path


def _path_is_within(path: Path, root: Path) -> bool:
    """Whether ``path`` resolves inside ``root``, following any symlinks."""
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _paths_resolve_equal(left: Path, right: Path) -> bool:
    """Fail-closed resolved path equality for metadata-controlled paths."""
    try:
        return left.resolve() == right.resolve()
    except (OSError, RuntimeError):
        return False


def _path_is_lexically_within(path: Path, root: Path) -> bool:
    """Whether an absolute path is lexically below ``root`` without links."""
    try:
        path.absolute().relative_to(root.absolute())
    except ValueError:
        return False
    return True


def _symlink_component_issue(path: Path, trusted_root: Path) -> Optional[str]:
    """Describe a symlink component between ``trusted_root`` and ``path``.

    Checking component-by-component, before resolving, prevents a top-level
    app output directory or one of its parents inside the checkout from
    redirecting fresh writes.  System-level aliases above ``trusted_root``
    (for example macOS ``/var``) are intentionally outside this boundary.
    """
    absolute_path = path.absolute()
    absolute_root = trusted_root.absolute()
    try:
        relative = absolute_path.relative_to(absolute_root)
    except ValueError:
        return f"{absolute_path} is outside trusted root {absolute_root}"
    current = absolute_root
    if current.is_symlink():
        return f"path component is a symlink: {current}"
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return f"path component is a symlink: {current}"
    return None


def _trusted_output_anchor(path: Path) -> Path:
    """Choose the boundary whose descendants must be free of symlinks."""
    for candidate in (REPO_ROOT, Path(tempfile.gettempdir())):
        if _path_is_lexically_within(path, candidate):
            return candidate
    return path.parent


def _tree_contains_symlink(path: Path) -> bool:
    """Fail-closed check for links inside a potentially writable tree."""
    if path.is_symlink():
        return True
    if not path.exists():
        return False
    try:
        return any(child.is_symlink() for child in path.rglob("*"))
    except OSError:
        return True


def _writable_run_tree_issue(
    exp_dir: Path,
    family: str,
    *,
    include_baselines: bool = False,
) -> Optional[str]:
    """Return why an app-owned output tree is unsafe to write.

    Saved runs are local data, but they may have been copied from elsewhere or
    modified since the app created them.  Refuse symlinks in every subtree the
    app will mutate so a crafted resume cannot redirect CSV, plot, or metadata
    writes outside its results directory.  Canonical/external baseline trees
    are deliberately excluded when they are read-only.
    """
    if family not in FAMILY_MODELS:
        return f"unknown family {family!r}"
    if _path_has_unsafe_hydra_syntax(exp_dir):
        return f"results root contains Hydra interpolation/control syntax: {exp_dir}"
    if _path_is_lexically_within(exp_dir, REPO_ROOT):
        component_issue = _symlink_component_issue(exp_dir, REPO_ROOT)
        if component_issue is not None:
            return component_issue
    if exp_dir.is_symlink():
        return f"results root is a symlink: {exp_dir}"
    if not exp_dir.exists() or not exp_dir.is_dir():
        return f"results root is not a directory: {exp_dir}"

    root_metadata = exp_dir / "run_metadata.json"
    if root_metadata.is_symlink():
        return f"root metadata is a symlink: {root_metadata}"

    writable_trees = [exp_dir / family, exp_dir / "medical_results"]
    if include_baselines:
        writable_trees.append(exp_dir / "baselines" / family)
    for tree in writable_trees:
        if not _path_is_within(tree, exp_dir):
            return f"writable path escapes the results root: {tree}"
        if _tree_contains_symlink(tree):
            return f"writable path contains a symlink: {tree}"
    return None


def _validate_writable_run_tree(
    exp_dir: Path,
    family: str,
    *,
    include_baselines: bool = False,
) -> None:
    """Raise before any app-owned run path can be mutated unsafely."""
    issue = _writable_run_tree_issue(
        exp_dir, family, include_baselines=include_baselines
    )
    if issue is not None:
        raise ValueError(f"unsafe saved run: {issue}")


def _validate_model_id(value: object, *, label: str, expected: str) -> str:
    """Validate one fixed model alias read from UI input or saved metadata."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} model is missing")
    model = value.strip()
    if (
        not _SAFE_MODEL_ID.fullmatch(model)
        or "${" in model
        or any(ord(char) < 32 or ord(char) == 127 for char in model)
        or Path(model).name != model
    ):
        raise ValueError(
            f"{label} model contains unsafe Hydra or path syntax: {model!r}"
        )
    if model != expected:
        raise ValueError(
            f"unsupported {label} model for this fixed experiment: {model!r}; "
            f"expected {expected!r}"
        )
    return model


def _validate_model_trio(
    family: str,
    frontier: object,
    weaker: object,
    concession: object,
) -> tuple[str, str, str]:
    """Return the exact supported model trio for ``family`` or fail closed."""
    if family not in FAMILY_MODELS:
        raise ValueError(f"unknown family: {family!r}")
    expected = FAMILY_MODELS[family]
    return (
        _validate_model_id(frontier, label="frontier", expected=expected["frontier"]),
        _validate_model_id(weaker, label="weaker", expected=expected["weaker"]),
        _validate_model_id(concession, label="concession", expected=CONCESSION_MODEL),
    )


def _path_has_unsafe_hydra_syntax(path: Path) -> bool:
    """Whether a filesystem path could become Hydra/OmegaConf syntax."""
    value = path.as_posix()
    return "${" in value or any(ord(char) < 32 or ord(char) == 127 for char in value)


def _load_json_dict(path: Path) -> Optional[dict]:
    """Load a JSON object, rejecting malformed data and other JSON shapes."""
    if path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _record_matches_family(record: Optional[dict], family: str) -> bool:
    """Whether a metadata copy can safely describe ``family``.

    Older partial metadata sometimes omitted ``family`` entirely, so absence
    is recoverable from the root key or directory name. An explicit different
    family is not recoverable and must never be merged into this record.
    """
    if record is None:
        return False
    recorded = record.get("family")
    return recorded is None or (
        isinstance(recorded, str) and recorded.strip() == family
    )


def _saved_metadata_timestamp(meta: dict, family_record: dict) -> Optional[float]:
    """Return an explicit metadata timestamp, without filesystem fallback."""
    values = (
        family_record.get("last_updated_at_utc"),
        family_record.get("recorded_at_utc"),
        meta.get("last_updated_at_utc"),
        meta.get("created_or_updated_at_utc"),
    )
    for value in values:
        if not value:
            continue
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
    return None


def _saved_models(record: dict, family: str) -> tuple[str, str, str]:
    """Read the exact model trio a resumable run recorded.

    Model defaults are deliberately not substituted for missing metadata.
    Doing so after a configuration change could mix models inside one results
    directory on resume.
    """
    if not _record_matches_family(record, family):
        raise ValueError(
            f"resume metadata family mismatch: requested {family!r}, "
            f"recorded {record.get('family')!r}"
        )
    labels = (
        ("frontier_model", "frontier"),
        ("weaker_model", "weaker"),
        ("concession_model", "concession"),
    )
    values: list[str] = []
    missing: list[str] = []
    for key, label in labels:
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            missing.append(label)
        else:
            values.append(value.strip())
    if missing:
        raise ValueError(
            "resume metadata is missing the recorded "
            f"{', '.join(missing)} model(s); refusing to guess and mix model "
            "versions in the existing run"
        )
    return _validate_model_trio(family, values[0], values[1], values[2])


def _recorded_run_roots(
    record: dict,
    exp_dir: Path,
    recorded_run_root: object = None,
) -> list[Path]:
    """Recover metadata-derived old roots that identify this exact run.

    Matching the directory name prevents a similarly-shaped external baseline
    path from being rebased merely because it ends in
    ``<run>/baselines/<family>``.
    """
    candidates = [recorded_run_root]
    family_dir = _resolve_metadata_path(record.get("family_dir"))
    if family_dir is not None:
        candidates.append(family_dir.parent)
    roots: list[Path] = []
    for value in candidates:
        path = _resolve_metadata_path(value)
        if (
            path is not None
            and path.name == exp_dir.name
            and all(not _paths_resolve_equal(path, root) for root in roots)
        ):
            roots.append(path)
    return roots


def _baseline_source_from_record(
    record: dict,
    exp_dir: Path,
    family: str,
    recorded_run_root: object = None,
) -> tuple[Optional[Path], Optional[bool]]:
    """Resolve one run's saved baseline source after a repository move.

    New metadata records relative in-repo paths. Older metadata used absolute
    paths, so use its recorded run root to rebase canonical and per-run paths
    onto the current checkout. Arbitrary external baseline paths remain
    absolute.
    """
    raw_value = record.get("baselines_dir")
    raw_path = _resolve_metadata_path(raw_value)
    explicit_canonical = record.get("using_canonical_baselines")
    if "using_canonical_baselines" in record and not isinstance(
        explicit_canonical, bool
    ):
        return None, None
    canonical = CANONICAL_BASELINES_ROOT / family
    per_run = exp_dir / "baselines" / family
    old_run_roots = _recorded_run_roots(record, exp_dir, recorded_run_root)

    inferred_path: Optional[Path] = None
    inferred_canonical: Optional[bool] = None

    if raw_path is not None and _paths_resolve_equal(raw_path, canonical):
        inferred_path, inferred_canonical = canonical, True

    # Legacy metadata stored absolute paths. Rebase only when the path is
    # provably tied to this record's old run root; suffix-only matching can
    # mistake an intentional external baseline archive for an in-run path.
    if inferred_path is None:
        for old_run_root in old_run_roots:
            old_per_run = old_run_root / "baselines" / family
            if raw_path is not None and _paths_resolve_equal(raw_path, old_per_run):
                inferred_path, inferred_canonical = per_run, False
                break
            if old_run_root.parent.name == "exp":
                old_canonical = (
                    old_run_root.parent.parent / "data" / "canonical_baselines" / family
                )
                if raw_path is not None and _paths_resolve_equal(
                    raw_path, old_canonical
                ):
                    inferred_path, inferred_canonical = canonical, True
                    break

    if inferred_path is None and raw_path is not None:
        if _paths_resolve_equal(raw_path, per_run):
            inferred_path, inferred_canonical = per_run, False
        else:
            inferred_path, inferred_canonical = raw_path, False

    if inferred_path is not None:
        # The flag and path are a single trust decision. A crafted or partially
        # edited record must not use one to silently override the other.
        if (
            isinstance(explicit_canonical, bool)
            and explicit_canonical is not inferred_canonical
        ):
            return None, None
        return inferred_path, inferred_canonical
    if raw_value is not None and str(raw_value).strip():
        # A present-but-unsafe path must not silently degrade into a writable
        # in-run baseline target.
        return None, None
    # Older metadata sometimes recorded only the boolean source choice. Keep
    # that legitimate format resumable while never guessing when both fields
    # are absent.
    if explicit_canonical is True:
        return canonical, True
    if explicit_canonical is False:
        return per_run, False
    return None, None


def _baseline_source_is_writable_in_run(path: Path, exp_dir: Path, family: str) -> bool:
    """Whether resume may safely continue baseline stages in this run tree."""
    expected = exp_dir / "baselines" / family
    try:
        return (
            _paths_resolve_equal(path, expected)
            and _path_is_within(path, exp_dir)
            and not _tree_contains_symlink(path)
        )
    except (OSError, RuntimeError):
        return False


def _merged_family_record(
    exp_dir: Path,
    family: str,
    root_meta: Optional[dict] = None,
) -> tuple[Optional[dict], object, float]:
    """Return one consistent family record for discovery and construction.

    Root and family metadata are normally written together, but an interrupted
    write can leave one copy newer or incomplete. Prefer the newer copy and
    fill only its missing keys from the other so the sidebar and DebateRun use
    the same saved source.
    """
    if root_meta is None:
        root_meta = _load_json_dict(exp_dir / "run_metadata.json") or {}
    families = root_meta.get("families")
    root_record = families.get(family) if isinstance(families, dict) else None
    if not isinstance(root_record, dict) or not _record_matches_family(
        root_record, family
    ):
        root_record = None

    family_meta_path = exp_dir / family / "run_metadata.json"
    family_record = _load_json_dict(family_meta_path)
    if not _record_matches_family(family_record, family):
        family_record = None
    if root_record is None and family_record is None:
        return None, root_meta.get("run_root"), 0.0

    family_saved_timestamp = (
        _saved_metadata_timestamp({}, family_record)
        if family_record is not None
        else None
    )
    root_saved_timestamp = (
        _saved_metadata_timestamp({}, root_record) if root_record is not None else None
    )
    # The root-level update time may have been bumped by the *other* family.
    # Use it only for a genuinely single-family legacy root when neither copy
    # has a family-specific timestamp; a shared root timestamp cannot safely
    # be attributed to either family.
    if (
        root_record is not None
        and root_saved_timestamp is None
        and family_saved_timestamp is None
        and isinstance(families, dict)
        and len(families) == 1
    ):
        root_saved_timestamp = _saved_metadata_timestamp(root_meta, {})
    root_timestamp = (
        root_saved_timestamp
        if root_saved_timestamp is not None
        else _metadata_timestamp({}, {}, exp_dir)
        if root_record is not None
        else 0.0
    )
    family_timestamp = (
        family_saved_timestamp
        if family_saved_timestamp is not None
        else (
            _metadata_timestamp({}, {}, family_meta_path)
            if family_record is not None
            else 0.0
        )
    )

    # An explicit saved timestamp always outranks a timestamp-less partial
    # copy. In particular, merely writing ``{}`` to family metadata must not
    # make an older run look newer through that file's mtime.
    if root_saved_timestamp is not None and family_saved_timestamp is None:
        primary, secondary = root_record, family_record
        timestamp = root_timestamp
    elif family_saved_timestamp is not None and root_saved_timestamp is None:
        primary, secondary = family_record, root_record
        timestamp = family_timestamp
    elif family_record is not None and family_timestamp > root_timestamp:
        primary, secondary = family_record, root_record
        timestamp = family_timestamp
    else:
        primary, secondary = root_record, family_record
        timestamp = root_timestamp

    merged = dict(secondary or {})
    merged.update(primary or {})
    # Treat baseline path + canonical flag as one decision. Mixing a newer
    # path with an older flag can change the source despite choosing the right
    # metadata copy.
    if primary is not None and (
        primary.get("baselines_dir") or "using_canonical_baselines" in primary
    ):
        for key in ("baselines_dir", "using_canonical_baselines"):
            if key in primary:
                merged[key] = primary[key]
            else:
                merged.pop(key, None)
    return merged, root_meta.get("run_root"), timestamp


def _metadata_timestamp(meta: dict, family_record: dict, candidate: Path) -> float:
    """Best available saved-run timestamp, with directory mtime as fallback."""
    saved = _saved_metadata_timestamp(meta, family_record)
    if saved is not None:
        return saved
    try:
        return candidate.stat().st_mtime
    except OSError:
        return 0.0


def _results_dir_tiebreaker(path: Path) -> tuple[str, int]:
    """Stable ordering for runs whose metadata has second-level precision."""
    stem, suffix = path.name.rsplit("_results", 1)
    collision = int(suffix[1:]) if suffix.startswith("_") else 1
    return stem, collision


def _is_results_dir(path: Path) -> bool:
    """True for timestamped result dirs, including collision suffixes."""
    marker = "_results"
    if marker not in path.name:
        return False
    suffix = path.name.rsplit(marker, 1)[1]
    return suffix == "" or (suffix.startswith("_") and suffix[1:].isdigit())


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    """Replace a CSV atomically without ever following a destination link."""
    if destination.is_symlink():
        raise ValueError(f"refusing to replace symlinked working CSV: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _extend_working_csv_from_pilot(working_csv: Path, target_n: int) -> int:
    """Extend a `data{seed}.csv` (internal schema) so it has at least
    `target_n` rows. New rows are pulled from the DDXPlus pilot CSV via
    the medical loader, then merged in while preserving any per-row
    state already on disk (complete flag, transcript, answer, etc.).

    Returns the number of rows added.
    """
    if not working_csv.exists():
        return 0
    if target_n < 1 or target_n > 100:
        raise ValueError("target_n must be between 1 and 100")
    if working_csv.is_symlink():
        raise ValueError(f"refusing to extend symlinked working CSV: {working_csv}")
    try:
        df = pd.read_csv(working_csv, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"working CSV is unreadable: {working_csv}") from exc
    current = len(df)

    id_column = (
        "id" if "id" in df.columns else "case_id" if "case_id" in df.columns else None
    )
    if id_column is None:
        raise ValueError(f"working CSV is missing an id/case_id column: {working_csv}")
    raw_existing_ids = tuple(df[id_column].astype(str))
    existing_ids = tuple(case_id.strip() for case_id in raw_existing_ids)
    if raw_existing_ids != existing_ids:
        raise ValueError(f"working CSV contains a non-canonical case ID: {working_csv}")
    if any(not case_id for case_id in existing_ids):
        raise ValueError(f"working CSV contains a blank case ID: {working_csv}")
    if len(set(existing_ids)) != len(existing_ids):
        raise ValueError(f"working CSV contains duplicate case IDs: {working_csv}")

    required_pilot_count = max(current, target_n)
    expected_ids = pilot_case_ids(required_pilot_count)
    if len(expected_ids) != required_pilot_count:
        raise ValueError(
            f"pilot does not contain {required_pilot_count} unique usable case IDs"
        )
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("pilot case IDs are not unique")
    if existing_ids != expected_ids[:current]:
        raise ValueError(
            "working CSV case IDs must be the exact ordered prefix of the "
            f"medical pilot; refusing to extend malformed cohort: {working_csv}"
        )
    if current >= target_n:
        return 0

    # Use the project's own loader to emit `target_n` fresh rows in
    # internal schema, then take the slice we don't already have.
    from core.load.medical import main as medical_loader

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        tmp_path = Path(tf.name)
    try:
        medical_loader(tmp_path, source_path=PILOT_CSV, limit=target_n)
        fresh = pd.read_csv(tmp_path, keep_default_na=False)
    finally:
        tmp_path.unlink(missing_ok=True)

    fresh_id_column = (
        "id"
        if "id" in fresh.columns
        else "case_id"
        if "case_id" in fresh.columns
        else None
    )
    if fresh_id_column is None:
        raise ValueError("medical loader output is missing an id/case_id column")
    fresh_ids = tuple(fresh[fresh_id_column].astype(str).str.strip())
    if fresh_ids != expected_ids[:target_n] or len(set(fresh_ids)) != len(fresh_ids):
        raise ValueError(
            "medical loader output does not match the exact ordered pilot cohort"
        )

    # Start from canonical pilot rows, then overlay every existing cell by ID.
    # This preserves transcript/completion/judgement state while avoiding a
    # positional append that could attach state to the wrong patient.
    out = fresh.copy()
    out["__merge_case_id"] = out[fresh_id_column].astype(str).str.strip()
    out.set_index("__merge_case_id", inplace=True, drop=True)
    output_columns = list(df.columns) + [
        column for column in fresh.columns if column not in df.columns
    ]
    for column in df.columns:
        if column not in out.columns:
            if column == id_column:
                out[column] = out[fresh_id_column]
            else:
                out[column] = False if column.startswith("complete") else ""
    for row_index, case_id in enumerate(existing_ids):
        for column in df.columns:
            out.at[case_id, column] = df.iloc[row_index][column]
    out = out.loc[list(expected_ids[:target_n]), output_columns].reset_index(drop=True)
    _atomic_write_csv(out, working_csv)
    return target_n - current


def extend_run_to_n_cases(
    exp_dir: Path,
    family: str,
    target_n: int,
    *,
    include_baselines: bool = True,
) -> dict:
    """Walk an existing run's exp dir and extend every working `data*.csv`
    to `target_n` rows. New rows have `complete=False` so the next pipeline
    pass will process them.

    Judgement CSVs (`<arm>_<judge>/data0[_swap]_judgement.csv`) are NOT
    extended here. `core/judge.py` reconciles them against their sibling
    `data0.csv` at the start of every judge stage — that path correctly
    handles both row appends and `complete`/`transcript` re-sync, and is
    the single place where judgement CSVs get mutated.

    ``include_baselines=False`` leaves baseline working files untouched. It
    is required when a resume uses canonical/external read-only baselines,
    including an in-run path that resolves through a symlink to an archive.

    Returns a {relative_path: rows_added} report for logging.
    """
    _validate_writable_run_tree(exp_dir, family, include_baselines=include_baselines)
    report: dict = {}
    working_files = [
        exp_dir / family / "debate_sim" / "data0.csv",
        exp_dir / family / "debate_sim" / "data0_swap.csv",
    ]
    if include_baselines:
        working_files[:0] = [
            exp_dir / "baselines" / family / "baseline_blind" / "data0.csv",
            exp_dir / "baselines" / family / "baseline_oracle" / "data0.csv",
        ]
    for wf in working_files:
        if wf.exists():
            added = _extend_working_csv_from_pilot(wf, target_n)
            report[str(wf.relative_to(exp_dir))] = added
    return report


def _judgement_csv_is_complete(path: Path, expected_ids: tuple[str, ...]) -> bool:
    """Whether a judge output finished every requested case.

    Judgements may legitimately be inconclusive, so this completion check does
    not require the answer to be scoreable. It is used only to decide whether
    a sibling family's cached output is complete enough to preserve during a
    combined re-aggregation.
    """
    if not expected_ids:
        return False
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except Exception:
        return False
    id_column = (
        "id" if "id" in df.columns else "case_id" if "case_id" in df.columns else None
    )
    if id_column is None:
        return False
    ids = df[id_column].astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated(keep=False).any():
        return False
    complete_columns = [
        column
        for column in df.columns
        if column == "complete_judge" or column.startswith("complete_judge_")
    ]
    if not complete_columns:
        return False
    complete = df[complete_columns[0]].map(
        lambda value: value is True or str(value).strip().lower() == "true"
    )
    by_id = dict(zip(ids, complete))
    return all(by_id.get(str(case_id).strip(), False) for case_id in expected_ids)


def _family_outputs_are_complete(exp_dir: Path, family: str, record: dict) -> bool:
    """True when all E1-E4 cached judge outputs cover the saved cohort."""
    try:
        frontier, weaker, _ = _saved_models(record, family)
        n_cases = int(record.get("n_cases", 0))
    except (TypeError, ValueError):
        return False
    if n_cases < 1 or n_cases > 100:
        return False
    expected_ids = pilot_case_ids(n_cases)
    judge_for_condition = {
        "e1_info_asymmetry": frontier,
        "e2_double_asymmetry": weaker,
        "e3_capability_asymmetry": weaker,
        "e4_full_symmetry": frontier,
    }
    debate_root = exp_dir / family / "debate_sim"
    return all(
        _judgement_csv_is_complete(
            debate_root / f"{condition}_{judge}" / filename,
            expected_ids,
        )
        for condition, judge in judge_for_condition.items()
        for filename in ("data0_judgement.csv", "data0_swap_judgement.csv")
    )


def _family_output_case_ids(
    exp_dir: Path, family: str, record: dict
) -> Optional[tuple[str, ...]]:
    """Return the saved-cohort IDs present in any expected debate output.

    A sibling family may be midway through an extension: its metadata already
    records the new target while its cached judgement CSVs still contain only
    the earlier completed cohort. Aggregation can safely retain those existing
    rows, provided their IDs are structurally sound and their baselines cover
    the rows that are actually present. ``None`` means the cached output is
    malformed or outside the saved pilot cohort and aggregation must fail
    closed rather than overwrite shared derived tables.
    """
    try:
        frontier, weaker, _ = _saved_models(record, family)
        n_cases = int(record.get("n_cases", 0))
    except (TypeError, ValueError):
        return None
    if n_cases < 1 or n_cases > 100:
        return None
    expected_ids = pilot_case_ids(n_cases)
    if len(expected_ids) != n_cases:
        return None
    expected_set = set(expected_ids)
    observed: set[str] = set()
    judge_for_condition = {
        "e1_info_asymmetry": frontier,
        "e2_double_asymmetry": weaker,
        "e3_capability_asymmetry": weaker,
        "e4_full_symmetry": frontier,
    }
    debate_root = exp_dir / family / "debate_sim"
    for condition, judge in judge_for_condition.items():
        for filename in ("data0_judgement.csv", "data0_swap_judgement.csv"):
            path = debate_root / f"{condition}_{judge}" / filename
            if not path.exists():
                continue
            try:
                frame = pd.read_csv(path, keep_default_na=False)
            except Exception:
                return None
            id_column = (
                "id"
                if "id" in frame.columns
                else "case_id"
                if "case_id" in frame.columns
                else None
            )
            if id_column is None:
                return None
            ids = frame[id_column].astype(str).str.strip()
            if ids.eq("").any() or ids.duplicated(keep=False).any():
                return None
            file_ids = set(ids)
            if not file_ids.issubset(expected_set):
                return None
            observed.update(file_ids)
    return tuple(case_id for case_id in expected_ids if case_id in observed)


def find_latest_run(
    base_dir: Path = REPO_ROOT / "exp",
    family: Optional[str] = None,
) -> Optional[dict]:
    """Return the latest saved run, optionally limited to one model family.

    A directory is resumable when its name follows the timestamped
    `*_results` convention and either root or family metadata contains a valid
    record. Saved metadata timestamps determine recency; directory mtime is
    only a fallback for older records.
    """
    if base_dir.is_symlink() or not base_dir.exists() or not base_dir.is_dir():
        return None
    if family is not None and family not in FAMILY_MODELS:
        return None

    resumable: list[tuple[float, tuple[str, int], str, dict]] = []
    for cand in (
        p
        for p in base_dir.iterdir()
        if p.is_dir() and not p.is_symlink() and _is_results_dir(p)
    ):
        meta = _load_json_dict(cand / "run_metadata.json") or {}
        families = meta.get("families")
        root_families = families if isinstance(families, dict) else {}

        if family is None:
            family_names = [
                name
                for name in FAMILY_MODELS
                if isinstance(root_families.get(name), dict)
                or _load_json_dict(cand / name / "run_metadata.json") is not None
            ]
        elif (
            isinstance(root_families.get(family), dict)
            or _load_json_dict(cand / family / "run_metadata.json") is not None
        ):
            family_names = [family]
        else:
            continue
        for family_name in family_names:
            if _writable_run_tree_issue(cand, family_name) is not None:
                # Do not display a target that construction must reject, and
                # never follow a symlinked family/derived-results tree merely
                # to populate the resume card.
                continue
            family_record, recorded_run_root, saved_timestamp = _merged_family_record(
                cand, family_name, root_meta=meta
            )
            if family_record is None:
                continue
            try:
                frontier_model, weaker_model, concession_model = _saved_models(
                    family_record, family_name
                )
            except ValueError:
                # A resume target without an unambiguous saved model trio is
                # unsafe to display: constructing it would have to guess.
                continue
            try:
                n_cases = int(family_record.get("n_cases", 0))
            except (TypeError, ValueError):
                continue
            if n_cases < 1 or n_cases > 100:
                continue
            baselines_dir, using_canonical = _baseline_source_from_record(
                family_record,
                cand,
                family_name,
                recorded_run_root=recorded_run_root,
            )
            if baselines_dir is None:
                # Resuming without the original denominator would silently
                # change the experiment, so do not offer this as a target.
                continue
            expected_ids = pilot_case_ids(n_cases)
            if len(expected_ids) != n_cases:
                continue
            baselines_writable = not bool(
                using_canonical
            ) and _baseline_source_is_writable_in_run(baselines_dir, cand, family_name)
            # Partial app-owned baselines are deliberately resumable: their
            # missing rows are exactly what the paid baseline stage will fill.
            # Canonical and arbitrary external sources are read-only, so they
            # must exist and be fully valid before they can be offered.
            if not baselines_writable:
                canonical_candidate = (
                    canonical_baselines_path(
                        family_name,
                        judges=(weaker_model, frontier_model),
                        min_rows=n_cases,
                        expected_ids=expected_ids,
                    )
                    if using_canonical
                    else None
                )
                invalid_canonical = bool(using_canonical) and (
                    canonical_candidate is None
                    or not _paths_resolve_equal(canonical_candidate, baselines_dir)
                )
                invalid_external = not bool(using_canonical) and (
                    not baselines_dir.exists()
                    or bool(
                        missing_baseline_files(
                            baselines_dir,
                            family_name,
                            judges=(weaker_model, frontier_model),
                            min_rows=n_cases,
                            expected_ids=expected_ids,
                        )
                    )
                )
                if invalid_canonical or invalid_external:
                    continue
            result = {
                "exp_dir": cand,
                "family": family_name,
                "n_cases": n_cases,
                "frontier_model": frontier_model,
                "weaker_model": weaker_model,
                "concession_model": concession_model,
                "recorded_at_utc": family_record.get("recorded_at_utc"),
                "last_updated_at_utc": family_record.get("last_updated_at_utc")
                or meta.get("last_updated_at_utc"),
                "baselines_dir": baselines_dir,
                "using_canonical_baselines": bool(using_canonical),
                "baselines_writable": baselines_writable,
            }
            resumable.append(
                (
                    saved_timestamp,
                    _results_dir_tiebreaker(cand),
                    family_name,
                    result,
                )
            )

    if not resumable:
        return None
    return max(resumable, key=lambda item: item[:3])[3]


class Phase(str, Enum):
    PENDING = "pending"
    BASELINES = "baselines"
    DEBATE = "debate"
    JUDGING = "judging"
    ANALYSIS = "analysis"
    DONE = "done"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class CaseStatus:
    case_id: str
    question: str
    correct: str
    distractor: str
    debate_complete: bool = False
    arm_answers: dict = field(
        default_factory=dict
    )  # condition -> {"answer": str, "correct": Optional[bool]}
    transcript_json: Optional[str] = None


@dataclass
class RunSnapshot:
    phase: Phase
    message: str
    cases: list[CaseStatus]
    log_tail: list[str]
    exp_dir: Optional[str] = None
    family: Optional[str] = None
    results_dir: Optional[str] = None
    started_at: float = 0.0
    finished_at: Optional[float] = None
    return_code: Optional[int] = None


class DebateRun:
    """One isolated, ephemeral pipeline run."""

    def __init__(
        self,
        family: str,
        n_cases: int,
        openai_key: str,
        anthropic_key: str,
        concession_model: Optional[str] = None,
        resume_from: Optional[Path] = None,
        use_canonical_baselines: bool = True,
        frontier_model: Optional[str] = None,
        weaker_model: Optional[str] = None,
    ) -> None:
        if family not in FAMILY_MODELS:
            raise ValueError(f"unknown family: {family!r}")
        if n_cases < 1 or n_cases > 100:
            raise ValueError("n_cases must be between 1 and 100")
        self.family = family
        self.n_cases = int(n_cases)
        self.is_resume = resume_from is not None
        expected_baseline_ids = pilot_case_ids(self.n_cases)

        # Resume mode: reuse the existing exp dir so the underlying
        # pipeline picks up where it left off (rows with complete=True
        # are skipped by core.debate; judgements with complete_judge=True
        # are skipped by core.judge). Fresh mode: create a new timestamped
        # results root under exp/.
        self.extension_report: dict = {}
        self._created_fresh_root = False
        if resume_from is not None:
            resume_anchor = _trusted_output_anchor(resume_from)
            component_issue = _symlink_component_issue(resume_from, resume_anchor)
            if (
                resume_from.is_symlink()
                or component_issue is not None
                or ".." in resume_from.parts
                or _path_has_unsafe_hydra_syntax(resume_from)
            ):
                self._fail_setup(
                    "unsafe saved run: resume path may not be a symlink, "
                    "pass through a symlinked app path, contain '..', or "
                    f"contain Hydra interpolation/control syntax: {resume_from}"
                )
            resolved = resume_from.resolve()
            if not resolved.exists():
                self._fail_setup(f"resume directory does not exist: {resolved}")
            self.exp_root = resolved
        else:
            try:
                self.exp_root = create_results_root()
            except (OSError, ValueError) as exc:
                self._fail_setup(str(exc))
            self._created_fresh_root = True
        self.exp_dir = self.exp_root
        self.family_dir = self.exp_dir / family
        self.results_dir = self.exp_dir / "medical_results"
        try:
            _validate_writable_run_tree(self.exp_dir, family)
        except ValueError as exc:
            self._fail_setup(str(exc))
        self._resume_record: Optional[dict] = None
        if self.is_resume:
            record, _, _ = _merged_family_record(self.exp_dir, self.family)
            if record is None:
                self._fail_setup(
                    f"no valid {family} metadata was found in {self.exp_dir}; "
                    "refusing to guess which models produced this run"
                )
            try:
                saved_frontier, saved_weaker, saved_concession = _saved_models(
                    record, family
                )
            except ValueError as exc:
                self._fail_setup(str(exc))

            requested_models = (
                ("frontier", frontier_model, saved_frontier),
                ("weaker", weaker_model, saved_weaker),
                ("concession", concession_model, saved_concession),
            )
            mismatches = [
                f"{label}={requested!r} (saved {saved!r})"
                for label, requested, saved in requested_models
                if requested is not None
                and (not isinstance(requested, str) or requested.strip() != saved)
            ]
            if mismatches:
                self._fail_setup(
                    "resume model selection does not match the saved run: "
                    + "; ".join(mismatches)
                )
            self.frontier_model = saved_frontier
            self.weaker_model = saved_weaker
            self.concession_model = saved_concession
            self._resume_record = record
        else:
            defaults = FAMILY_MODELS[family]
            self.frontier_model = (
                frontier_model.strip()
                if isinstance(frontier_model, str) and frontier_model.strip()
                else defaults["frontier"]
            )
            self.weaker_model = (
                weaker_model.strip()
                if isinstance(weaker_model, str) and weaker_model.strip()
                else defaults["weaker"]
            )
            self.concession_model = (
                concession_model.strip()
                if isinstance(concession_model, str) and concession_model.strip()
                else CONCESSION_MODEL
            )

            try:
                (
                    self.frontier_model,
                    self.weaker_model,
                    self.concession_model,
                ) = _validate_model_trio(
                    family,
                    self.frontier_model,
                    self.weaker_model,
                    self.concession_model,
                )
            except ValueError as exc:
                self._fail_setup(str(exc))

        selected_models = (
            self.frontier_model,
            self.weaker_model,
            self.concession_model,
        )
        uses_anthropic = any(model.startswith("claude-") for model in selected_models)
        uses_openai = any(not model.startswith("claude-") for model in selected_models)
        if uses_openai and not openai_key:
            self._fail_setup(
                "OpenAI API key is required because this run uses an OpenAI model."
            )
        if uses_anthropic and not anthropic_key:
            self._fail_setup(
                "Anthropic API key is required because this run uses an "
                "Anthropic model."
            )

        # Resolve the baselines source. Three cases:
        #   1. Fresh run with canonical requested + canonical fully populated
        #      → skip baseline-API stages, read from canonical.
        #   2. Fresh run otherwise → compute baselines in-run.
        #   3. Resume run → ALWAYS honour the metadata-recorded baseline
        #      source for that run. Switching baseline source mid-run would
        #      silently change PGR denominators and confuse comparisons, so
        #      the sidebar toggle only applies to fresh runs.
        fresh_baselines_dir = self.exp_dir / "baselines" / family
        self.baselines_read_only = False
        if self.is_resume:
            recorded, recorded_is_canonical = self._read_recorded_baseline_source()
            if recorded is not None:
                # Honour the saved source. Legacy absolute in-repo paths are
                # rebased by `_read_recorded_baseline_source`, so moving or
                # forking the checkout does not silently turn a canonical
                # source into a fresh paid baseline run.
                if recorded_is_canonical:
                    canonical = canonical_baselines_path(
                        family,
                        judges=(self.weaker_model, self.frontier_model),
                        min_rows=self.n_cases,
                        expected_ids=expected_baseline_ids,
                    )
                    if canonical is None:
                        # Never turn an invalid committed reference bundle into a
                        # writable resume target. Failing here prevents both paid
                        # downstream work and accidental changes under data/.
                        self._fail_setup(
                            "the canonical baselines recorded for this run are "
                            f"incomplete or invalid for its first {self.n_cases} "
                            "pilot case(s); rebuild the canonical bundle or start "
                            "a fresh run with baseline recomputation enabled"
                        )
                    self.baselines_dir = canonical
                    self.using_canonical_baselines = True
                    self.baselines_read_only = True
                else:
                    # A partial per-run baseline directory is a valid resume
                    # target: the pipeline creates missing rows/files there.
                    writable_in_run_baselines = _baseline_source_is_writable_in_run(
                        recorded, self.exp_dir, family
                    )
                    if not writable_in_run_baselines and not recorded.exists():
                        self._fail_setup(
                            "the external baselines recorded for this run no "
                            f"longer exist at {recorded}; restore them or start "
                            "a fresh run with baseline recomputation enabled"
                        )
                    self.baselines_dir = recorded
                    self.using_canonical_baselines = False
                    if not writable_in_run_baselines:
                        issues = missing_baseline_files(
                            recorded,
                            family,
                            judges=(self.weaker_model, self.frontier_model),
                            min_rows=self.n_cases,
                            expected_ids=expected_baseline_ids,
                        )
                        if issues:
                            preview = "; ".join(issues[:3])
                            if len(issues) > 3:
                                preview += f"; and {len(issues) - 3} more"
                            self._fail_setup(
                                "the external baselines recorded for this run are "
                                f"incomplete or invalid for its first {self.n_cases} "
                                f"pilot case(s): {preview}. External baselines are "
                                "read-only during resume; restore a complete archive "
                                "or start a fresh run with baseline recomputation enabled"
                            )
                        self.baselines_read_only = True
            else:
                self._fail_setup(
                    "resume metadata does not record a baseline source; refusing "
                    "to guess and change this run's PGR denominator. Start a fresh "
                    "run or repair run_metadata.json with its original baselines_dir"
                )
        else:
            canonical = (
                canonical_baselines_path(
                    family,
                    judges=(self.weaker_model, self.frontier_model),
                    min_rows=self.n_cases,
                    expected_ids=expected_baseline_ids,
                )
                if use_canonical_baselines
                else None
            )
            self.using_canonical_baselines = canonical is not None
            self.baselines_dir = (
                canonical if canonical is not None else fresh_baselines_dir
            )
            self.baselines_read_only = canonical is not None
        if not self.is_resume:
            self._write_run_metadata("app/streamlit_app.py")
        else:
            # Extend any working CSVs that already exist; the pipeline
            # itself will create the missing ones with the new n_cases.
            self.extension_report = extend_run_to_n_cases(
                self.exp_dir,
                family,
                self.n_cases,
                include_baselines=not self.baselines_read_only,
            )
            # Always sync metadata to the requested n on resume — even when
            # no rows were physically appended (e.g. a prior run crashed
            # before its data0.csv files were created, so there was nothing
            # to extend yet but the user's intent is still to run at the
            # new n). Without this, the next "Resume / extend" slider's
            # `prior_n` would be stale.
            self._update_run_metadata_for_extension()

        self._lock = threading.Lock()
        self._phase = Phase.PENDING
        self._message = "Ready to start."
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._started_at = 0.0
        self._finished_at: Optional[float] = None
        self._return_code: Optional[int] = None
        self._stop_requested = False
        self._log_tail: list[str] = []
        self._max_log_tail = 200

        # Pre-load pilot rows so the UI table can be populated immediately,
        # before any subprocess writes data0.csv.
        try:
            df = pd.read_csv(PILOT_CSV, encoding="utf-8")
            df = df.head(self.n_cases)
            self._initial_cases = [
                CaseStatus(
                    case_id=str(r["case_id"]),
                    question=str(r.get("question_stem", "")),
                    correct=str(r.get("pathology", "")),
                    distractor=str(r.get("top_differential", "")),
                )
                for _, r in df.iterrows()
            ]
        except FileNotFoundError:
            self._initial_cases = []

        # Create the session key file only after every metadata, resume, and
        # pilot-loading operation that can reject this setup has succeeded.
        # This ordering means a corrupt saved run or an output write failure
        # cannot orphan credentials for an object that is never returned.
        self._create_session_temp_files(openai_key, anthropic_key)

    # ---------------------------------------------------------------- lifecycle

    def _create_session_temp_files(self, openai_key: str, anthropic_key: str) -> None:
        """Create private session storage, cleaning it on every failure."""
        try:
            self._tempdir = Path(tempfile.mkdtemp(prefix="medical_debate_app_"))
            self._secrets_path = self._tempdir / "SECRETS"
            payload = "\n".join(
                [
                    f"API_KEY={openai_key}",
                    f"ANTHROPIC_API_KEY={anthropic_key or ''}",
                    "DEFAULT_ORG=",
                    "",
                ]
            )

            # Set 0600 at creation time rather than writing first and chmodding
            # afterwards, so there is no interval with broader permissions.
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor: Optional[int] = os.open(self._secrets_path, flags, 0o600)
            try:
                stream = os.fdopen(descriptor, "w", encoding="utf-8")
                descriptor = None
                with stream:
                    stream.write(payload)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            self.log_path = self._tempdir / "pipeline.log"
        except BaseException:
            # Includes cancellation signals such as KeyboardInterrupt and
            # SystemExit: no failure after key creation may leave credentials.
            self._discard_setup_artifacts()
            raise

    def _discard_setup_artifacts(self) -> None:
        """Best-effort cleanup for a constructor that will not return."""
        secrets_path = getattr(self, "_secrets_path", None)
        if isinstance(secrets_path, Path):
            try:
                secrets_path.unlink(missing_ok=True)
            except OSError:
                pass
        tempdir = getattr(self, "_tempdir", None)
        if isinstance(tempdir, Path):
            shutil.rmtree(tempdir, ignore_errors=True)
        if getattr(self, "_created_fresh_root", False):
            try:
                self.exp_root.rmdir()
            except OSError:
                # Only remove the empty results directory this constructor
                # just created. If anything reached it, leave it untouched.
                pass

    def _fail_setup(self, message: str) -> None:
        """Remove session secrets before rejecting an unsafe run setup."""
        self._discard_setup_artifacts()
        raise ValueError(message)

    def _read_recorded_baseline_source(
        self,
    ) -> tuple[Optional[Path], Optional[bool]]:
        """Return the same merged baseline source shown by discovery."""
        record, recorded_run_root, _ = _merged_family_record(self.exp_dir, self.family)
        if self._resume_record is not None:
            record = self._resume_record
        if record is not None:
            return _baseline_source_from_record(
                record,
                self.exp_dir,
                self.family,
                recorded_run_root=recorded_run_root,
            )
        return None, None

    def _update_run_metadata_for_extension(self) -> None:
        """Bump n_cases + last_updated_at_utc in run_metadata.json after
        extending a previous run. Preserves the original creation time
        and any pre-existing fields the writer didn't know about. Also
        backfills `baselines_dir` and `using_canonical_baselines` if
        absent, so very old metadata gains the explicit fields once
        touched.
        """
        _validate_writable_run_tree(
            self.exp_dir,
            self.family,
            include_baselines=not self.baselines_read_only,
        )
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        merged_record, _, _ = _merged_family_record(self.exp_dir, self.family)
        family_meta = self.family_dir / "run_metadata.json"
        # Write one fully healed effective record to both metadata locations.
        # Starting from a merely non-empty family copy would otherwise leave
        # its missing model/entrypoint/timestamp fields missing forever.
        doc = dict(merged_record or {})
        doc["family"] = self.family
        doc["n_cases"] = self.n_cases
        doc["frontier_model"] = self.frontier_model
        doc["weaker_model"] = self.weaker_model
        doc["concession_model"] = self.concession_model
        doc["last_updated_at_utc"] = now
        doc["family_dir"] = _portable_metadata_path(self.family_dir)
        doc["baselines_dir"] = _portable_metadata_path(self.baselines_dir)
        doc["using_canonical_baselines"] = bool(self.using_canonical_baselines)
        self.family_dir.mkdir(parents=True, exist_ok=True)
        family_meta.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

        root_meta = self.exp_dir / "run_metadata.json"
        root_doc = _load_json_dict(root_meta) or {}
        root_doc["run_root"] = _portable_metadata_path(self.exp_dir)
        root_doc.setdefault("created_or_updated_at_utc", now)
        root_doc["last_updated_at_utc"] = now
        families = root_doc.get("families")
        if not isinstance(families, dict):
            families = {}
            root_doc["families"] = families
        families[self.family] = dict(doc)
        root_meta.write_text(json.dumps(root_doc, indent=2) + "\n", encoding="utf-8")

    def _write_run_metadata(self, entrypoint: str) -> None:
        _validate_writable_run_tree(
            self.exp_dir,
            self.family,
            include_baselines=not self.baselines_read_only,
        )
        recorded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        family_record = {
            "family": self.family,
            "n_cases": self.n_cases,
            "frontier_model": self.frontier_model,
            "weaker_model": self.weaker_model,
            "concession_model": self.concession_model,
            "entrypoint": entrypoint,
            "family_dir": _portable_metadata_path(self.family_dir),
            "baselines_dir": _portable_metadata_path(self.baselines_dir),
            "using_canonical_baselines": bool(self.using_canonical_baselines),
            "recorded_at_utc": recorded_at,
        }
        self.family_dir.mkdir(parents=True, exist_ok=True)
        (self.family_dir / "run_metadata.json").write_text(
            json.dumps(family_record, indent=2) + "\n",
            encoding="utf-8",
        )
        root_record = {
            "run_root": _portable_metadata_path(self.exp_dir),
            "created_or_updated_at_utc": recorded_at,
            "last_updated_at_utc": recorded_at,
            "families": {self.family: family_record},
        }
        (self.exp_dir / "run_metadata.json").write_text(
            json.dumps(root_record, indent=2) + "\n",
            encoding="utf-8",
        )

    def start(self) -> None:
        try:
            with self._lock:
                if self._thread is not None:
                    return
                self._phase = Phase.BASELINES
                self._message = "Running blind + oracle baselines."
                self._started_at = time.time()
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
        except BaseException:
            # If the OS cannot create the worker thread, the run never reaches
            # Streamlit session state and therefore cannot be cleared from the
            # UI. Remove credentials here at the lifecycle boundary before the
            # original error is surfaced to the user.
            with self._lock:
                self._thread = None
                self._phase = Phase.ERROR
                self._message = "Failed to start the background worker."
                self._finished_at = time.time()
            self._discard_setup_artifacts()
            raise

    def stop(self) -> None:
        """Signal stop and escalate from SIGTERM → SIGKILL if the subprocess
        doesn't exit promptly. Long frontier-model API calls can take tens
        of seconds and don't always respond to SIGTERM mid-call, so we
        give a short grace period and then hard-kill.
        """
        with self._lock:
            self._stop_requested = True
            proc = self._proc

        if proc is None:
            return
        self._request_process_stop(proc)

    def _request_process_stop(self, proc: subprocess.Popen) -> None:
        """Terminate one owned subprocess and escalate without blocking UI."""
        if proc.poll() is not None:
            return

        def _signal(sig: int) -> None:
            try:
                if os.name == "posix" and getattr(proc, "pid", None) is not None:
                    os.killpg(os.getpgid(proc.pid), sig)
                else:
                    proc.terminate()
            except (ProcessLookupError, PermissionError, OSError):
                pass

        # First: ask nicely.
        _signal(signal.SIGTERM)
        self._append_log("[stop] SIGTERM sent; waiting up to 5s for graceful exit")

        # Then escalate in a background thread so stop() returns fast.
        def _escalate() -> None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._append_log("[stop] SIGTERM ignored; sending SIGKILL")
                _signal(signal.SIGKILL if os.name == "posix" else signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._append_log(
                        "[stop] SIGKILL ignored; subprocess may be stuck in C code"
                    )

        threading.Thread(target=_escalate, daemon=True).start()

    def cleanup(self) -> None:
        self.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)
        try:
            shutil.rmtree(self._tempdir, ignore_errors=True)
        except Exception:
            pass

    def _delete_sensitive_temp_files(self) -> None:
        """Remove session credentials and provider prompt caches promptly."""
        try:
            self._secrets_path.unlink(missing_ok=True)
        except OSError:
            pass
        shutil.rmtree(self._tempdir / "prompt_history", ignore_errors=True)

    # ---------------------------------------------------------------- snapshot

    def snapshot(self) -> RunSnapshot:
        with self._lock:
            phase = self._phase
            message = self._message
            tail = list(self._log_tail[-40:])
            started = self._started_at
            finished = self._finished_at
            rc = self._return_code

        cases = self._read_cases()
        return RunSnapshot(
            phase=phase,
            message=message,
            cases=cases,
            log_tail=tail,
            exp_dir=str(self.exp_dir),
            family=self.family,
            results_dir=str(self.results_dir) if self.results_dir.exists() else None,
            started_at=started,
            finished_at=finished,
            return_code=rc,
        )

    # ---------------------------------------------------------------- internals

    def _read_cases(self) -> list[CaseStatus]:
        cases: dict[str, CaseStatus] = {
            c.case_id: CaseStatus(**vars(c)) for c in self._initial_cases
        }

        debate_csv = self.family_dir / "debate_sim" / "data0.csv"
        if debate_csv.exists():
            try:
                df = pd.read_csv(debate_csv, encoding="utf-8", keep_default_na=False)
            except Exception:
                df = None
            if df is not None:
                for _, row in df.iterrows():
                    case_id = str(row.get("id", ""))
                    if not case_id:
                        continue
                    c = cases.get(case_id) or CaseStatus(
                        case_id=case_id,
                        question=str(row.get("question", "")),
                        correct=str(row.get("correct answer", "")),
                        distractor=str(row.get("negative answer", "")),
                    )
                    complete_val = row.get("complete", False)
                    c.debate_complete = (
                        bool(complete_val) and str(complete_val).lower() != "false"
                    )
                    transcript = row.get("transcript", "")
                    if transcript:
                        c.transcript_json = transcript
                    cases[case_id] = c

        # Layer in final-judge answers for any arm directory that exists.
        debate_sim = self.family_dir / "debate_sim"
        if debate_sim.exists():
            judge_for_condition = {
                "e1_info_asymmetry": self.frontier_model,
                "e2_double_asymmetry": self.weaker_model,
                "e3_capability_asymmetry": self.weaker_model,
                "e4_full_symmetry": self.frontier_model,
            }
            for condition, judge_model in judge_for_condition.items():
                sub = debate_sim / f"{condition}_{judge_model}"
                if not sub.is_dir() or sub.is_symlink():
                    continue
                for fname, swap in (
                    ("data0_judgement.csv", False),
                    ("data0_swap_judgement.csv", True),
                ):
                    p = sub / fname
                    if not p.exists():
                        continue
                    try:
                        jdf = pd.read_csv(p, keep_default_na=False)
                    except Exception:
                        continue
                    for _, row in jdf.iterrows():
                        case_id = str(row.get("id", ""))
                        if not case_id or case_id not in cases:
                            continue
                        bucket = cases[case_id].arm_answers.setdefault(condition, {})
                        order = "swap" if swap else "orig"
                        bucket[order] = str(row.get("answer_judge", ""))
                        complete_value = row.get("complete_judge", False)
                        bucket[f"{order}_complete"] = complete_value is True or (
                            str(complete_value).strip().lower() == "true"
                        )

        return list(cases.values())

    def _append_log(self, line: str) -> None:
        with self._lock:
            self._log_tail.append(line.rstrip())
            if len(self._log_tail) > self._max_log_tail:
                self._log_tail = self._log_tail[-self._max_log_tail :]

    def _set_phase(self, phase: Phase, message: str) -> None:
        with self._lock:
            self._phase = phase
            self._message = message

    def _stop_check(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _stage_outcome(self, label: str, rc: int) -> Optional[str]:
        """Inspect a stage's return code and the stop flag.

        Returns:
            None if the stage succeeded and the run should continue.
            "stopped" if the user asked to stop (regardless of rc).
            "error: ..." if the stage failed for any other reason.
        """
        if self._stop_check():
            return "stopped"
        if rc != 0:
            return f"error: {label} exited with code {rc}"
        return None

    def _run_stage(self, label: str, cmd: list[str]) -> int:
        """Spawn one subprocess, stream its output to the in-memory tail,
        and return its exit code. Honours stop requests.
        """
        _validate_writable_run_tree(
            self.exp_dir,
            self.family,
            include_baselines=not self.baselines_read_only,
        )
        self._append_log(f"$ {' '.join(cmd[:2])} … ({label})")

        env = os.environ.copy()
        env["MEDICAL_DEBATE_APP_RUNNER"] = "1"
        env["MEDICAL_DEBATE_SECRETS_PATH"] = str(self._secrets_path)
        env["MEDICAL_DEBATE_PROMPT_HISTORY_DIR"] = str(self._tempdir / "prompt_history")
        # The pipeline's matplotlib plots run headless; give them a private cache.
        env.setdefault("MPLCONFIGDIR", str(self._tempdir / "mpl"))
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

        popen_kwargs = dict(
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True  # so we can SIGTERM the group

        # This is intentionally the final operation before Popen. A stop that
        # already happened must not start another paid stage.
        with self._lock:
            if self._stop_requested:
                return 1
        proc = subprocess.Popen(cmd, **popen_kwargs)
        with self._lock:
            self._proc = proc
            stop_raced = self._stop_requested
        # A stop can land after the preflight check while Popen is creating the
        # child. Once the process is published, terminate that just-launched
        # child immediately rather than leaving it detached and billable.
        if stop_raced:
            self._request_process_stop(proc)

        assert proc.stdout is not None
        try:
            with open(self.log_path, "a", encoding="utf-8") as logf:
                for raw in proc.stdout:
                    if not raw:
                        continue
                    logf.write(raw)
                    self._append_log(raw)
                    if self._stop_check():
                        break
        finally:
            proc.stdout.close()
            rc = proc.wait()
            with self._lock:
                if self._proc is proc:
                    self._proc = None
        return rc

    def _python(self) -> str:
        # Use the same interpreter Streamlit is running under so the user
        # doesn't need a specific venv name.
        return sys.executable

    def _hydra_exp_dir(self, path: Path) -> str:
        """Return an exp_dir override value Hydra can parse.

        Hydra's override grammar treats spaces, parentheses and several other
        characters as syntax. Prefer a repo-relative path, then quote and
        escape the value so an explicitly selected external resume directory
        is safe too.
        """
        try:
            value = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            value = path.as_posix()
        if _path_has_unsafe_hydra_syntax(Path(value)):
            raise ValueError(f"unsafe Hydra path syntax: {value!r}")
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"

    def _combined_analysis_inputs(self) -> tuple[list[Path], Path]:
        """Build read-only, family-aware inputs for aggregate/stat stages.

        A run root may contain OpenAI and Anthropic outputs whose baselines
        come from different canonical, in-run, or external roots. The internal
        analysis utilities accept only one baseline root, so expose each
        validated source via a temporary ``<root>/<family>`` symlink. This
        keeps archived baselines read-only and prevents resuming one family
        from erasing the other from the shared derived tables.
        """
        family_dirs: list[Path] = []
        baseline_sources: dict[str, Path] = {}

        for family in FAMILY_MODELS:
            family_dir = self.exp_dir / family
            if family_dir.exists() or family_dir.is_symlink():
                family_issue = _writable_run_tree_issue(self.exp_dir, family)
                if family_issue is not None:
                    raise RuntimeError(
                        f"cannot safely inspect sibling {family}: {family_issue}"
                    )
            record, recorded_run_root, _ = _merged_family_record(self.exp_dir, family)

            if family == self.family:
                record = record or self._resume_record
                if record is None:
                    raise RuntimeError(
                        f"cannot aggregate {family}: its run metadata is missing"
                    )
                family_dirs.append(family_dir)
                baseline_source = self.baselines_dir
                frontier, weaker = self.frontier_model, self.weaker_model
                n_cases = self.n_cases
            else:
                debate_root = family_dir / "debate_sim"
                if not debate_root.exists():
                    continue
                if record is None:
                    if any(debate_root.rglob("*_judgement.csv")):
                        raise RuntimeError(
                            f"cannot safely preserve sibling {family} results: "
                            "their metadata is missing or has a family mismatch"
                        )
                    continue
                available_ids = _family_output_case_ids(self.exp_dir, family, record)
                if available_ids is None:
                    raise RuntimeError(
                        f"cannot safely preserve sibling {family} results: "
                        "their cached judgement case IDs are malformed or do "
                        "not match the saved pilot cohort"
                    )
                if not available_ids:
                    continue
                if not _family_outputs_are_complete(self.exp_dir, family, record):
                    self._append_log(
                        f"[runner] preserving partial sibling {family} output "
                        f"for {len(available_ids)} available case(s)"
                    )
                frontier, weaker, _ = _saved_models(record, family)
                baseline_source, _ = _baseline_source_from_record(
                    record,
                    self.exp_dir,
                    family,
                    recorded_run_root=recorded_run_root,
                )
                if baseline_source is None:
                    raise RuntimeError(
                        f"cannot safely preserve available sibling {family} results: "
                        "their baseline source is missing from metadata"
                    )
                family_dirs.append(family_dir)
                expected_ids = available_ids

            if family == self.family:
                expected_ids = pilot_case_ids(n_cases)
            issues = missing_baseline_files(
                baseline_source,
                family,
                judges=(weaker, frontier),
                min_rows=len(expected_ids),
                expected_ids=expected_ids,
            )
            if issues:
                preview = "; ".join(issues[:3])
                if len(issues) > 3:
                    preview += f"; and {len(issues) - 3} more"
                raise RuntimeError(
                    f"cannot safely aggregate {family}: its saved baselines are "
                    f"incomplete or invalid ({preview})"
                )
            baseline_sources[family] = baseline_source

        view = self._tempdir / "analysis_baselines"
        view.mkdir(exist_ok=True)
        for family, source in baseline_sources.items():
            link = view / family
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(source.resolve(), target_is_directory=True)

        self._append_log(
            "[runner] combined analysis families="
            + ",".join(path.name for path in family_dirs)
            + "; baseline sources="
            + ", ".join(
                f"{family}:{source}" for family, source in baseline_sources.items()
            )
        )
        return family_dirs, view

    def _run(self) -> None:
        try:
            py = self._python()
            family = self.family
            n = self.n_cases
            frontier, weaker = self.frontier_model, self.weaker_model
            limit_arg = f"++limit={n}"
            cases_at_a_time = CASES_AT_A_TIME[family]
            judge_threads_arg = f"++anthropic_num_threads={cases_at_a_time}"
            debate_threads_arg = f"++anthropic_num_threads={cases_at_a_time * BON}"
            self._append_log(
                f"[runner] family={family} cases_at_a_time={cases_at_a_time} "
                f"(debate_threads={cases_at_a_time * BON}, judge_threads={cases_at_a_time})"
            )

            # ---- Baselines (blind + oracle) ---------------------------------
            if self.baselines_read_only:
                source_label = (
                    "canonical" if self.using_canonical_baselines else "external"
                )
                self._set_phase(
                    Phase.BASELINES,
                    f"Using validated {source_label} baselines at "
                    f"{self.baselines_dir} (no baseline API spend).",
                )
                self._append_log(
                    f"[runner] skipping baseline stage; validated {source_label} "
                    f"baselines are read-only at {self.baselines_dir}"
                )
            else:
                self._set_phase(
                    Phase.BASELINES,
                    "Running blind + oracle baselines (fresh) for both weaker and frontier judges.",
                )
                # Run both weaker AND frontier judges so PGR can be computed
                # for every debate arm (E1/E4 use the frontier judge; E2/E3
                # use the weaker judge). core.debate writes data0.csv once
                # per arm and doesn't depend on the judge model — run it
                # against the first judge to satisfy Hydra's required
                # config, then loop core.judge + core.scoring.accuracy over
                # both models.
                baseline_judges = list(dict.fromkeys((weaker, frontier)))
                for arm in ("medical_blind", "medical_oracle"):
                    if self._stop_check():
                        self._finalise(Phase.STOPPED, "Stopped by user.")
                        return
                    debate_cmd = [
                        py,
                        "-m",
                        "core.debate",
                        f"exp_dir={self._hydra_exp_dir(self.baselines_dir)}",
                        f"+experiment={arm}",
                        limit_arg,
                        debate_threads_arg,
                        f"++judge.language_model.model={baseline_judges[0]}",
                        f"++judge_name={baseline_judges[0]}",
                    ]
                    rc = self._run_stage(f"{arm}:core.debate", debate_cmd)
                    outcome = self._stage_outcome(f"{arm}:core.debate", rc)
                    if outcome == "stopped":
                        self._finalise(Phase.STOPPED, "Stopped by user.")
                        return
                    if outcome is not None:
                        self._finalise(Phase.ERROR, outcome.removeprefix("error: "))
                        return

                    for judge_model in baseline_judges:
                        for stage in ("core.judge", "core.scoring.accuracy"):
                            if self._stop_check():
                                self._finalise(Phase.STOPPED, "Stopped by user.")
                                return
                            cmd = [
                                py,
                                "-m",
                                stage,
                                f"exp_dir={self._hydra_exp_dir(self.baselines_dir)}",
                                f"+experiment={arm}",
                                limit_arg,
                                judge_threads_arg,
                                f"++judge.language_model.model={judge_model}",
                                f"++judge_name={judge_model}",
                            ]
                            rc = self._run_stage(f"{arm}:{stage}:{judge_model}", cmd)
                            outcome = self._stage_outcome(
                                f"{arm}:{stage}:{judge_model}", rc
                            )
                            if outcome == "stopped":
                                self._finalise(Phase.STOPPED, "Stopped by user.")
                                return
                            if outcome is not None:
                                self._finalise(
                                    Phase.ERROR, outcome.removeprefix("error: ")
                                )
                                return

            # ---- Debate generation -----------------------------------------
            self._set_phase(Phase.DEBATE, f"Generating debate transcripts ({n} cases).")
            if self._stop_check():
                self._finalise(Phase.STOPPED, "Stopped by user.")
                return
            debate_cmd = [
                py,
                "-m",
                "core.debate",
                f"exp_dir={self._hydra_exp_dir(self.family_dir)}",
                "+experiment=medical_debate",
                limit_arg,
                debate_threads_arg,
                f"++correct_debater.language_model.model={frontier}",
                f"++incorrect_debater.language_model.model={frontier}",
                f"++correct_preference.language_model.model={frontier}",
                f"++incorrect_preference.language_model.model={frontier}",
                "++correct_debater.BoN=4",
                "++incorrect_debater.BoN=4",
                "++correct_debater.language_model.temperature=0.8",
                "++incorrect_debater.language_model.temperature=0.8",
            ]
            rc = self._run_stage("debate", debate_cmd)
            outcome = self._stage_outcome("debate", rc)
            if outcome == "stopped":
                self._finalise(Phase.STOPPED, "Stopped by user.")
                return
            if outcome is not None:
                self._finalise(Phase.ERROR, outcome.removeprefix("error: "))
                return

            # ---- Final-judge E1-E4 -----------------------------------------
            self._set_phase(Phase.JUDGING, "Running E1-E4 final judges.")
            arm_specs = [
                ("e1_info_asymmetry", "medical_debate", frontier),
                ("e2_double_asymmetry", "medical_debate_e2_double_asymmetry", weaker),
                (
                    "e3_capability_asymmetry",
                    "medical_debate_e3_capability_asymmetry",
                    weaker,
                ),
                ("e4_full_symmetry", "medical_debate_e4_full_symmetry", frontier),
            ]
            for cond, experiment, judge_model in arm_specs:
                if self._stop_check():
                    self._finalise(Phase.STOPPED, "Stopped by user.")
                    return
                judge_name = f"{cond}_{judge_model}"
                common = [
                    f"exp_dir={self._hydra_exp_dir(self.family_dir)}",
                    f"+experiment={experiment}",
                    limit_arg,
                    judge_threads_arg,
                    f"++judge.language_model.model={judge_model}",
                    f"++judge_name={judge_name}",
                ]
                rc = self._run_stage(f"{cond}:judge", [py, "-m", "core.judge", *common])
                outcome = self._stage_outcome(f"{cond}:judge", rc)
                if outcome == "stopped":
                    self._finalise(Phase.STOPPED, "Stopped by user.")
                    return
                if outcome is not None:
                    self._finalise(Phase.ERROR, outcome.removeprefix("error: "))
                    return
                rc = self._run_stage(
                    f"{cond}:score", [py, "-m", "core.scoring.accuracy", *common]
                )
                outcome = self._stage_outcome(f"{cond}:score", rc)
                if outcome == "stopped":
                    self._finalise(Phase.STOPPED, "Stopped by user.")
                    return
                if outcome is not None:
                    self._finalise(Phase.ERROR, outcome.removeprefix("error: "))
                    return

            # ---- Concession judge ------------------------------------------
            self._set_phase(
                Phase.JUDGING, f"Running concession judge ({self.concession_model})."
            )
            if self._stop_check():
                self._finalise(Phase.STOPPED, "Stopped by user.")
                return
            concession_cmd = [
                py,
                "-m",
                "core.judge",
                f"exp_dir={self._hydra_exp_dir(self.family_dir)}",
                "+experiment=medical_debate",
                limit_arg,
                judge_threads_arg,
                "++judge_type=concession",
                f"++concession_judge.language_model.model={self.concession_model}",
                f"++judge_name=concession_{self.concession_model}",
            ]
            rc = self._run_stage("concession", concession_cmd)
            outcome = self._stage_outcome("concession", rc)
            if outcome == "stopped":
                self._finalise(Phase.STOPPED, "Stopped by user.")
                return
            if outcome is not None:
                self._finalise(Phase.ERROR, outcome.removeprefix("error: "))
                return

            # ---- Analysis + aggregation ------------------------------------
            self._set_phase(Phase.ANALYSIS, "Aggregating results and plots.")
            if self._stop_check():
                self._finalise(Phase.STOPPED, "Stopped by user.")
                return
            rc = self._run_stage(
                "analysis",
                [
                    py,
                    "scripts/analyze_medical_debate.py",
                    str(self.family_dir),
                    "--baselines-dir",
                    str(self.baselines_dir),
                ],
            )
            outcome = self._stage_outcome("analysis", rc)
            if outcome == "stopped":
                self._finalise(Phase.STOPPED, "Stopped by user.")
                return
            if outcome is not None:
                self._finalise(Phase.ERROR, outcome.removeprefix("error: "))
                return
            family_dirs, analysis_baselines = self._combined_analysis_inputs()
            rc = self._run_stage(
                "aggregate",
                [
                    py,
                    "scripts/aggregate_medical_results.py",
                    *(str(path) for path in family_dirs),
                    "--baselines-dir",
                    str(analysis_baselines),
                    "--out-dir",
                    str(self.results_dir),
                ],
            )
            outcome = self._stage_outcome("aggregate", rc)
            if outcome == "stopped":
                self._finalise(Phase.STOPPED, "Stopped by user.")
                return
            if outcome is not None:
                self._finalise(Phase.ERROR, outcome.removeprefix("error: "))
                return
            # No --analysis-dir: the script derives one per family dir, which
            # keeps the QC table correct if more than one family is present.
            rc = self._run_stage(
                "paired_tests",
                [
                    py,
                    "scripts/compute_paired_tests.py",
                    *(str(path) for path in family_dirs),
                    "--baselines-dir",
                    str(analysis_baselines),
                    "--out-dir",
                    str(self.results_dir),
                ],
            )
            outcome = self._stage_outcome("paired_tests", rc)
            if outcome == "stopped":
                self._finalise(Phase.STOPPED, "Stopped by user.")
                return
            if outcome is not None:
                self._finalise(Phase.ERROR, outcome.removeprefix("error: "))
                return

            self._finalise(Phase.DONE, "Finished.")

        except Exception as exc:  # surface unexpected failures to the UI
            self._append_log(f"[runner exception] {exc!r}")
            self._finalise(Phase.ERROR, f"Runner crashed: {exc}")

    def _finalise(self, phase: Phase, message: str) -> None:
        self._delete_sensitive_temp_files()
        with self._lock:
            self._phase = phase
            self._message = message
            self._finished_at = time.time()
            self._return_code = 0 if phase == Phase.DONE else 1


def estimate_cost_usd(
    family: str, n_cases: int, *, include_baseline_recompute: bool = False
) -> float:
    per_case = COST_PER_CASE_USD.get(family, 0.4)
    if include_baseline_recompute:
        per_case += BASELINE_RECOMPUTE_COST_PER_CASE_USD.get(family, 0.0)
    return per_case * n_cases


def parse_transcript(transcript_json: str) -> Optional[dict]:
    if not transcript_json:
        return None
    try:
        return json.loads(transcript_json)
    except (TypeError, json.JSONDecodeError):
        return None


def transcript_rounds(transcript_json: str) -> list[dict]:
    """Return a list of {speaker, side, text} dicts in display order."""
    parsed = parse_transcript(transcript_json)
    if not parsed:
        return []
    swap = bool(parsed.get("swap", False))
    correct_label = parsed.get("answers", {}).get("correct", "Diagnosis A")
    incorrect_label = parsed.get("answers", {}).get("incorrect", "Diagnosis B")
    if swap:
        a_side, a_label = "incorrect", incorrect_label
        b_side, b_label = "correct", correct_label
    else:
        a_side, a_label = "correct", correct_label
        b_side, b_label = "incorrect", incorrect_label

    out: list[dict] = []
    for ri, rnd in enumerate(parsed.get("rounds", []) or []):
        for letter, side, label in (("A", a_side, a_label), ("B", b_side, b_label)):
            content = rnd.get(side)
            if isinstance(content, dict):
                content = (
                    content.get("response")
                    or content.get("content")
                    or json.dumps(content)
                )
            if not content:
                continue
            out.append(
                {
                    "round": ri + 1,
                    "speaker": f"Debater {letter} ({label})",
                    "side": side,
                    "text": str(content),
                }
            )
    return out
