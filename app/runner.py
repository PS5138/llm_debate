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
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_CSV = REPO_ROOT / "data" / "ddxplus" / "ddxplus_debate_pilot_100.csv"

FAMILY_MODELS = {
    "openai": {"frontier": "gpt-5.5", "weaker": "gpt-5.4-mini"},
    "anthropic": {"frontier": "claude-opus-4-6", "weaker": "claude-sonnet-4-6"},
}

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

CONDITIONS = ["e1_info_asymmetry", "e2_double_asymmetry", "e3_capability_asymmetry", "e4_full_symmetry"]


def create_results_root(base_dir: Path = REPO_ROOT / "exp") -> Path:
    """Create a fresh timestamped results directory under exp/."""
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    candidate = base_dir / f"{stamp}_results"
    suffix = 2
    while candidate.exists():
        candidate = base_dir / f"{stamp}_results_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _extend_working_csv_from_pilot(working_csv: Path, target_n: int) -> int:
    """Extend a `data{seed}.csv` (internal schema) so it has at least
    `target_n` rows. New rows are pulled from the DDXPlus pilot CSV via
    the medical loader, then merged in while preserving any per-row
    state already on disk (complete flag, transcript, answer, etc.).

    Returns the number of rows added.
    """
    if not working_csv.exists():
        return 0
    df = pd.read_csv(working_csv, keep_default_na=False)
    current = len(df)
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

    new_rows = fresh.iloc[current:target_n].copy()
    # Fill any columns that the live CSV has but the freshly-loaded slice
    # doesn't (e.g. complete_judge / answer_judge added by core.judge on
    # earlier runs). Defaults are False for completion flags, "" otherwise.
    for col in df.columns:
        if col not in new_rows.columns:
            if col.startswith("complete"):
                new_rows[col] = False
            else:
                new_rows[col] = ""
    new_rows = new_rows[df.columns]
    out = pd.concat([df, new_rows], ignore_index=True)
    out.to_csv(working_csv, index=False, encoding="utf-8")
    return target_n - current


def _extend_judgement_csv_from_data(judgement_csv: Path, data_csv: Path, target_n: int) -> int:
    """Extend a `*_judgement.csv` so it has at least `target_n` rows.

    New rows are taken from the sibling `data0.csv` (which should have
    already been extended to >= target_n). Completion flags are reset to
    False so the next pipeline pass will judge them.
    """
    if not judgement_csv.exists() or not data_csv.exists():
        return 0
    jdf = pd.read_csv(judgement_csv, keep_default_na=False)
    current = len(jdf)
    if current >= target_n:
        return 0
    sdf = pd.read_csv(data_csv, keep_default_na=False)
    if len(sdf) < target_n:
        return 0  # data wasn't extended yet; nothing safe to do here

    new_rows = sdf.iloc[current:target_n].copy()
    for col in jdf.columns:
        if col not in new_rows.columns:
            if col.startswith("complete"):
                new_rows[col] = False
            else:
                new_rows[col] = ""
    new_rows = new_rows[jdf.columns]
    out = pd.concat([jdf, new_rows], ignore_index=True)
    out.to_csv(judgement_csv, index=False, encoding="utf-8")
    return target_n - current


def extend_run_to_n_cases(exp_dir: Path, family: str, target_n: int) -> dict:
    """Walk an existing run's exp dir and extend every working CSV +
    judgement CSV to `target_n` rows. The pipeline can then re-run and
    will skip already-complete rows, only spending API calls on the
    newly-appended cases.

    Returns a {relative_path: rows_added} report for logging.
    """
    report: dict = {}
    working_files = [
        exp_dir / "baselines" / family / "baseline_blind" / "data0.csv",
        exp_dir / "baselines" / family / "baseline_oracle" / "data0.csv",
        exp_dir / family / "debate_sim" / "data0.csv",
        exp_dir / family / "debate_sim" / "data0_swap.csv",
    ]
    # Step 1: extend the data CSVs from the pilot.
    for wf in working_files:
        if wf.exists():
            added = _extend_working_csv_from_pilot(wf, target_n)
            report[str(wf.relative_to(exp_dir))] = added

    # Step 2: extend any judgement CSVs from their sibling data0.csv.
    for wf in working_files:
        if not wf.exists():
            continue
        parent = wf.parent
        # Only data0.csv has judge subdirs; data0_swap.csv lives alongside
        # but is consumed by the same judge dirs.
        if wf.name != "data0.csv":
            continue
        for sub in parent.iterdir():
            if not sub.is_dir():
                continue
            for fname in ("data0_judgement.csv", "data0_swap_judgement.csv"):
                jp = sub / fname
                if jp.exists():
                    added = _extend_judgement_csv_from_data(jp, wf, target_n)
                    report[str(jp.relative_to(exp_dir))] = added
    return report


def find_latest_run(base_dir: Path = REPO_ROOT / "exp") -> Optional[dict]:
    """Scan exp/ for the most recent `<timestamp>_results/` directory and
    return its metadata (family, n_cases, frontier/weaker models, path).

    Returns None if no resumable run is found. The dir is considered
    resumable if it has a `run_metadata.json` at the root level so we
    can recover family + n_cases without asking the user.
    """
    if not base_dir.exists():
        return None
    candidates = sorted(
        (p for p in base_dir.iterdir() if p.is_dir() and p.name.endswith("_results")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for cand in candidates:
        meta_path = cand / "run_metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        families = meta.get("families") or {}
        if not families:
            continue
        # Take the first (usually only) family recorded.
        family_name, family_record = next(iter(families.items()))
        return {
            "exp_dir": cand,
            "family": family_name,
            "n_cases": int(family_record.get("n_cases", 0)),
            "frontier_model": family_record.get("frontier_model"),
            "weaker_model": family_record.get("weaker_model"),
            "recorded_at_utc": family_record.get("recorded_at_utc"),
        }
    return None


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
    arm_answers: dict = field(default_factory=dict)  # condition -> {"answer": str, "correct": Optional[bool]}
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
        concession_model: str = "gpt-4o-mini",
        resume_from: Optional[Path] = None,
    ) -> None:
        if family not in FAMILY_MODELS:
            raise ValueError(f"unknown family: {family!r}")
        if n_cases < 1 or n_cases > 100:
            raise ValueError("n_cases must be between 1 and 100")
        if not openai_key:
            # OpenAI key is required because concession judging defaults to gpt-4o-mini.
            raise ValueError("OpenAI API key is required (concession judge is gpt-4o-mini)")
        if family == "anthropic" and not anthropic_key:
            raise ValueError("Anthropic API key is required for the Anthropic family")

        self.family = family
        self.n_cases = int(n_cases)
        self.concession_model = concession_model
        self.is_resume = resume_from is not None

        self._tempdir = Path(tempfile.mkdtemp(prefix="medical_debate_app_"))
        self._secrets_path = self._tempdir / "SECRETS"
        self._secrets_path.write_text(
            "\n".join(
                [
                    f"API_KEY={openai_key}",
                    f"ANTHROPIC_API_KEY={anthropic_key or ''}",
                    "DEFAULT_ORG=",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        # Lock down on POSIX so other users can't read the key file.
        try:
            os.chmod(self._secrets_path, 0o600)
        except OSError:
            pass

        # Resume mode: reuse the existing exp dir so the underlying
        # pipeline picks up where it left off (rows with complete=True
        # are skipped by core.debate; judgements with complete_judge=True
        # are skipped by core.judge). Fresh mode: create a new timestamped
        # results root under exp/.
        self.extension_report: dict = {}
        if resume_from is not None:
            resolved = resume_from.resolve()
            if not resolved.exists():
                raise ValueError(f"resume directory does not exist: {resolved}")
            self.exp_root = resolved
        else:
            self.exp_root = create_results_root()
        self.exp_dir = self.exp_root
        self.baselines_dir = self.exp_dir / "baselines" / family
        self.family_dir = self.exp_dir / family
        self.results_dir = self.exp_dir / "medical_results"
        self.log_path = self._tempdir / "pipeline.log"
        if not self.is_resume:
            self._write_run_metadata("app/streamlit_app.py")
        else:
            # Extend any existing working / judgement CSVs to target_n,
            # then update run_metadata so future resumes see the new n.
            self.extension_report = extend_run_to_n_cases(
                self.exp_dir, family, self.n_cases
            )
            if any(v > 0 for v in self.extension_report.values()):
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

    # ---------------------------------------------------------------- lifecycle

    def _update_run_metadata_for_extension(self) -> None:
        """Bump n_cases + last_updated_at_utc in run_metadata.json after
        extending a previous run. Preserves the original creation time
        and any pre-existing fields the writer didn't know about.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        family_meta = self.family_dir / "run_metadata.json"
        if family_meta.exists():
            try:
                doc = json.loads(family_meta.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                doc = {}
            doc["n_cases"] = self.n_cases
            doc["last_updated_at_utc"] = now
            family_meta.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

        root_meta = self.exp_dir / "run_metadata.json"
        if root_meta.exists():
            try:
                doc = json.loads(root_meta.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                doc = {}
            doc["last_updated_at_utc"] = now
            families = doc.setdefault("families", {})
            f = families.setdefault(self.family, {})
            f["n_cases"] = self.n_cases
            f["last_updated_at_utc"] = now
            root_meta.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    def _write_run_metadata(self, entrypoint: str) -> None:
        models = FAMILY_MODELS[self.family]
        recorded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        family_record = {
            "family": self.family,
            "n_cases": self.n_cases,
            "frontier_model": models["frontier"],
            "weaker_model": models["weaker"],
            "concession_model": self.concession_model,
            "entrypoint": entrypoint,
            "family_dir": str(self.family_dir),
            "baselines_dir": str(self.baselines_dir),
            "recorded_at_utc": recorded_at,
        }
        self.family_dir.mkdir(parents=True, exist_ok=True)
        (self.family_dir / "run_metadata.json").write_text(
            json.dumps(family_record, indent=2) + "\n",
            encoding="utf-8",
        )
        root_record = {
            "run_root": str(self.exp_dir),
            "created_or_updated_at_utc": recorded_at,
            "last_updated_at_utc": recorded_at,
            "families": {self.family: family_record},
        }
        (self.exp_dir / "run_metadata.json").write_text(
            json.dumps(root_record, indent=2) + "\n",
            encoding="utf-8",
        )

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._phase = Phase.BASELINES
            self._message = "Running blind + oracle baselines."
            self._started_at = time.time()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Signal stop and escalate from SIGTERM → SIGKILL if the subprocess
        doesn't exit promptly. Long frontier-model API calls can take tens
        of seconds and don't always respond to SIGTERM mid-call, so we
        give a short grace period and then hard-kill.
        """
        with self._lock:
            self._stop_requested = True
            proc = self._proc

        if proc is None or proc.poll() is not None:
            return

        def _signal(sig: int) -> None:
            try:
                if os.name == "posix":
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
                    self._append_log("[stop] SIGKILL ignored; subprocess may be stuck in C code")

        threading.Thread(target=_escalate, daemon=True).start()

    def cleanup(self) -> None:
        self.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)
        try:
            shutil.rmtree(self._tempdir, ignore_errors=True)
        except Exception:
            pass

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
        cases: dict[str, CaseStatus] = {c.case_id: CaseStatus(**vars(c)) for c in self._initial_cases}

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
                    c.debate_complete = bool(complete_val) and str(complete_val).lower() != "false"
                    transcript = row.get("transcript", "")
                    if transcript:
                        c.transcript_json = transcript
                    cases[case_id] = c

        # Layer in final-judge answers for any arm directory that exists.
        debate_sim = self.family_dir / "debate_sim"
        if debate_sim.exists():
            for sub in sorted(debate_sim.iterdir()):
                if not sub.is_dir() or sub.name.startswith("concession_"):
                    continue
                condition = next((c for c in CONDITIONS if sub.name.startswith(c + "_")), None)
                if condition is None:
                    continue
                for fname, swap in (("data0_judgement.csv", False), ("data0_swap_judgement.csv", True)):
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
                        bucket["swap" if swap else "orig"] = str(row.get("answer_judge", ""))

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
        self._append_log(f"$ {' '.join(cmd[:2])} … ({label})")

        env = os.environ.copy()
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

        proc = subprocess.Popen(cmd, **popen_kwargs)
        with self._lock:
            self._proc = proc

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
                self._proc = None
        return rc

    def _python(self) -> str:
        # Use the same interpreter Streamlit is running under so the user
        # doesn't need a specific venv name.
        return sys.executable

    def _hydra_exp_dir(self, path: Path) -> str:
        """Return an exp_dir override value Hydra can parse.

        Hydra's override grammar treats spaces and parentheses in absolute
        paths as syntax. The app always writes under the repo's exp/
        directory, so pass a repo-relative path to subprocesses.
        """
        try:
            return path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    def _run(self) -> None:
        try:
            py = self._python()
            family = self.family
            n = self.n_cases
            models = FAMILY_MODELS[family]
            frontier, weaker = models["frontier"], models["weaker"]
            limit_arg = f"++limit={n}"
            cases_at_a_time = CASES_AT_A_TIME[family]
            judge_threads_arg = f"++anthropic_num_threads={cases_at_a_time}"
            debate_threads_arg = f"++anthropic_num_threads={cases_at_a_time * BON}"
            self._append_log(
                f"[runner] family={family} cases_at_a_time={cases_at_a_time} "
                f"(debate_threads={cases_at_a_time * BON}, judge_threads={cases_at_a_time})"
            )

            # ---- Baselines (blind + oracle) ---------------------------------
            self._set_phase(Phase.BASELINES, "Running blind + oracle baselines.")
            for arm in ("medical_blind", "medical_oracle"):
                for stage in ("core.debate", "core.judge", "core.scoring.accuracy"):
                    if self._stop_check():
                        self._finalise(Phase.STOPPED, "Stopped by user.")
                        return
                    # core.debate needs BoN-aware thread count; judge/scoring don't.
                    stage_threads = debate_threads_arg if stage == "core.debate" else judge_threads_arg
                    cmd = [
                        py,
                        "-m",
                        stage,
                        f"exp_dir={self._hydra_exp_dir(self.baselines_dir)}",
                        f"+experiment={arm}",
                        limit_arg,
                        stage_threads,
                        f"++judge.language_model.model={weaker}",
                        f"++judge_name={weaker}",
                    ]
                    rc = self._run_stage(f"{arm}:{stage}", cmd)
                    outcome = self._stage_outcome(f"{arm}:{stage}", rc)
                    if outcome == "stopped":
                        self._finalise(Phase.STOPPED, "Stopped by user.")
                        return
                    if outcome is not None:
                        self._finalise(Phase.ERROR, outcome.removeprefix("error: "))
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
                ("e3_capability_asymmetry", "medical_debate_e3_capability_asymmetry", weaker),
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
                rc = self._run_stage(f"{cond}:score", [py, "-m", "core.scoring.accuracy", *common])
                outcome = self._stage_outcome(f"{cond}:score", rc)
                if outcome == "stopped":
                    self._finalise(Phase.STOPPED, "Stopped by user.")
                    return
                if outcome is not None:
                    self._finalise(Phase.ERROR, outcome.removeprefix("error: "))
                    return

            # ---- Concession judge ------------------------------------------
            self._set_phase(Phase.JUDGING, f"Running concession judge ({self.concession_model}).")
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
                [py, "scripts/analyze_medical_debate.py", str(self.family_dir)],
            )
            outcome = self._stage_outcome("analysis", rc)
            if outcome == "stopped":
                self._finalise(Phase.STOPPED, "Stopped by user.")
                return
            if outcome is not None:
                self._finalise(Phase.ERROR, outcome.removeprefix("error: "))
                return
            rc = self._run_stage(
                "aggregate",
                [
                    py,
                    "scripts/aggregate_medical_results.py",
                    str(self.family_dir),
                    "--baselines-dir",
                    str(self.baselines_dir),
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

            self._finalise(Phase.DONE, "Finished.")

        except Exception as exc:  # surface unexpected failures to the UI
            self._append_log(f"[runner exception] {exc!r}")
            self._finalise(Phase.ERROR, f"Runner crashed: {exc}")

    def _finalise(self, phase: Phase, message: str) -> None:
        with self._lock:
            self._phase = phase
            self._message = message
            self._finished_at = time.time()
            self._return_code = 0 if phase == Phase.DONE else 1


def estimate_cost_usd(family: str, n_cases: int) -> float:
    return COST_PER_CASE_USD.get(family, 0.4) * n_cases


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
                content = content.get("response") or content.get("content") or json.dumps(content)
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
