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
    "anthropic": {"frontier": "claude-opus-4-7", "weaker": "claude-sonnet-4-6"},
}

# Conservative per-case cost estimates in USD, derived from frontier + weaker
# pricing × BoN=4 × 3 rounds × 2 debaters × judge passes across E1-E4 + concession.
# Numbers are rounded-up ballparks meant to set expectations, not invoicing.
COST_PER_CASE_USD = {
    "openai": 0.35,
    "anthropic": 0.45,
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

        self.exp_root = create_results_root()
        self.exp_dir = self.exp_root
        self.baselines_dir = self.exp_dir / "baselines" / family
        self.family_dir = self.exp_dir / family
        self.results_dir = self.exp_dir / "medical_results"
        self.log_path = self._tempdir / "pipeline.log"
        self._write_run_metadata("app/streamlit_app.py")

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
        with self._lock:
            self._stop_requested = True
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except (ProcessLookupError, PermissionError, OSError):
                pass

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
            threads_arg = "++anthropic_num_threads=5"

            # ---- Baselines (blind + oracle) ---------------------------------
            self._set_phase(Phase.BASELINES, "Running blind + oracle baselines.")
            for arm in ("medical_blind", "medical_oracle"):
                for stage in ("core.debate", "core.judge", "core.scoring.accuracy"):
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
                        threads_arg,
                        f"++judge.language_model.model={weaker}",
                        f"++judge_name={weaker}",
                    ]
                    rc = self._run_stage(f"{arm}:{stage}", cmd)
                    if rc != 0:
                        self._finalise(Phase.ERROR, f"{arm}:{stage} exited with code {rc}")
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
                threads_arg,
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
            if rc != 0:
                self._finalise(Phase.ERROR, f"debate exited with code {rc}")
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
                    threads_arg,
                    f"++judge.language_model.model={judge_model}",
                    f"++judge_name={judge_name}",
                ]
                rc = self._run_stage(f"{cond}:judge", [py, "-m", "core.judge", *common])
                if rc != 0:
                    self._finalise(Phase.ERROR, f"{cond}:judge exited with code {rc}")
                    return
                rc = self._run_stage(f"{cond}:score", [py, "-m", "core.scoring.accuracy", *common])
                if rc != 0:
                    self._finalise(Phase.ERROR, f"{cond}:score exited with code {rc}")
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
                threads_arg,
                "++judge_type=concession",
                f"++concession_judge.language_model.model={self.concession_model}",
                f"++judge_name=concession_{self.concession_model}",
            ]
            rc = self._run_stage("concession", concession_cmd)
            if rc != 0:
                self._finalise(Phase.ERROR, f"concession judge exited with code {rc}")
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
            if rc != 0:
                self._finalise(Phase.ERROR, f"analysis exited with code {rc}")
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
            if rc != 0:
                self._finalise(Phase.ERROR, f"aggregate exited with code {rc}")
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
