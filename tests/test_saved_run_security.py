"""No-network regressions for saved-run and output-path hardening."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import runner  # noqa: E402
from core.file_handler import BaselineType, Experiment, Method  # noqa: E402


def _record(family: str = "openai") -> dict:
    return {
        "family": family,
        "n_cases": 1,
        "frontier_model": runner.FAMILY_MODELS[family]["frontier"],
        "weaker_model": runner.FAMILY_MODELS[family]["weaker"],
        "concession_model": runner.CONCESSION_MODEL,
        "recorded_at_utc": "2026-08-08T12:00:00Z",
        "baselines_dir": f"baselines/{family}",
        "using_canonical_baselines": False,
    }


def _write_family_metadata(run_root: Path, family: str = "openai") -> None:
    family_dir = run_root / family
    family_dir.mkdir(parents=True, exist_ok=True)
    (family_dir / "run_metadata.json").write_text(
        json.dumps(_record(family)) + "\n", encoding="utf-8"
    )


def _expect_value_error(action, text: str) -> bool:
    try:
        action()
    except ValueError as exc:
        return text in str(exc)
    return False


def _expect_runtime_error(action, text: str) -> bool:
    try:
        action()
    except RuntimeError as exc:
        return text in str(exc)
    return False


class _EmptyStdout:
    def __iter__(self):
        return iter(())

    def close(self) -> None:
        pass


class _SuccessfulProcess:
    stdout = _EmptyStdout()

    def wait(self, timeout=None) -> int:
        return 0

    def poll(self) -> int:
        return 0


class _TerminableProcess:
    def __init__(self) -> None:
        self.stdout = _EmptyStdout()
        self.terminated = False

    def poll(self):
        return -15 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None) -> int:
        return -15 if self.terminated else 0


def main() -> int:
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as raw_tmp:
        base = Path(raw_tmp)

        # Extension must be an ID merge, never a positional append. Existing
        # state is overlaid onto the canonical pilot row with the same ID.
        pilot_ids = runner.pilot_case_ids(4)
        working = base / "working.csv"
        pd.DataFrame(
            {
                "id": pilot_ids[:2],
                "complete": [True, False],
                "transcript": ["preserve-this", ""],
            }
        ).to_csv(working, index=False)
        added = runner._extend_working_csv_from_pilot(working, 4)
        extended = pd.read_csv(working, keep_default_na=False)
        checks.append(
            (
                "working CSV extension merges by pilot ID and preserves state",
                added == 2
                and tuple(extended["id"]) == pilot_ids
                and extended.loc[0, "transcript"] == "preserve-this"
                and bool(extended.loc[0, "complete"])
                and not bool(extended.loc[2, "complete"]),
            )
        )

        malformed_cohorts = {
            "duplicate": (pilot_ids[0], pilot_ids[0]),
            "gapped": (pilot_ids[0], pilot_ids[2]),
            "reordered": (pilot_ids[1], pilot_ids[0]),
        }
        for label, ids in malformed_cohorts.items():
            malformed = base / f"{label}.csv"
            pd.DataFrame({"id": ids, "complete": [False, False]}).to_csv(
                malformed, index=False
            )
            before = malformed.read_bytes()
            rejected = _expect_value_error(
                lambda path=malformed: runner._extend_working_csv_from_pilot(path, 4),
                "case ID",
            )
            checks.append(
                (
                    f"{label} working cohort is rejected before mutation",
                    rejected and malformed.read_bytes() == before,
                )
            )

        missing_id = base / "missing-id.csv"
        pd.DataFrame({"complete": [False]}).to_csv(missing_id, index=False)
        missing_id_before = missing_id.read_bytes()
        checks.append(
            (
                "working cohort without IDs is rejected before mutation",
                _expect_value_error(
                    lambda: runner._extend_working_csv_from_pilot(missing_id, 4),
                    "missing an id/case_id",
                )
                and missing_id.read_bytes() == missing_id_before,
            )
        )

        valid = _record()
        checks.append(
            (
                "fixed OpenAI aliases remain resumable",
                runner._saved_models(valid, "openai")
                == (
                    runner.FAMILY_MODELS["openai"]["frontier"],
                    runner.FAMILY_MODELS["openai"]["weaker"],
                    runner.CONCESSION_MODEL,
                ),
            )
        )

        for unsafe in (
            "${oc.env:ANTHROPIC_API_KEY}",
            "../../outside",
            "gpt-5.5\n++judge_name=outside",
        ):
            malicious = dict(valid)
            malicious["frontier_model"] = unsafe
            checks.append(
                (
                    f"unsafe saved model is rejected: {unsafe!r}",
                    _expect_value_error(
                        lambda record=malicious: runner._saved_models(record, "openai"),
                        "unsafe Hydra or path syntax",
                    ),
                )
            )

        unsupported = dict(valid)
        unsupported["frontier_model"] = "gpt-future"
        checks.append(
            (
                "unrecognized safe-looking model alias is rejected",
                _expect_value_error(
                    lambda: runner._saved_models(unsupported, "openai"),
                    "unsupported frontier model",
                ),
            )
        )

        contradictory_source = runner._baseline_source_from_record(
            {
                "baselines_dir": str(base / "external" / "baselines" / "openai"),
                "using_canonical_baselines": True,
            },
            base / "2026-08-08_12-00-00_results",
            "openai",
        )
        suffix_only_source = runner._baseline_source_from_record(
            {
                "baselines_dir": str(base / "data" / "canonical_baselines" / "openai"),
                "using_canonical_baselines": True,
            },
            base / "2026-08-08_12-00-00_results",
            "openai",
        )
        checks.append(
            (
                "canonical flag cannot override an external or suffix-only path",
                contradictory_source == (None, None)
                and suffix_only_source == (None, None),
            )
        )

        interpolation_root = base / "${oc.env:ANTHROPIC_API_KEY}_results"
        _write_family_metadata(interpolation_root)
        checks.append(
            (
                "resume root containing Hydra interpolation is rejected",
                _expect_value_error(
                    lambda: runner.DebateRun(
                        "openai",
                        1,
                        "test-openai-key",
                        "",
                        resume_from=interpolation_root,
                    ),
                    "Hydra interpolation/control syntax",
                ),
            )
        )

        external_family = base / "external-family"
        _write_family_metadata(external_family.parent / "holder")
        # Reuse the generated family contents as a target outside the run.
        generated_family = external_family.parent / "holder" / "openai"
        external_family.mkdir()
        for child in generated_family.iterdir():
            child.rename(external_family / child.name)
        symlink_family_root = base / "symlink-discovery" / "2026-08-08_12-00-00_results"
        symlink_family_root.parent.mkdir()
        symlink_family_root.mkdir()
        (symlink_family_root / "openai").symlink_to(
            external_family, target_is_directory=True
        )
        checks.append(
            (
                "symlinked writable family tree is rejected and undiscoverable",
                _expect_value_error(
                    lambda: runner.DebateRun(
                        "openai",
                        1,
                        "test-openai-key",
                        "",
                        resume_from=symlink_family_root,
                    ),
                    "unsafe saved run",
                )
                and runner.find_latest_run(symlink_family_root.parent, family="openai")
                is None,
            )
        )

        nested_link_root = base / "nested-link-results"
        _write_family_metadata(nested_link_root)
        outside_debate = base / "outside-debate"
        outside_debate.mkdir()
        (nested_link_root / "openai" / "debate_sim").symlink_to(
            outside_debate, target_is_directory=True
        )
        checks.append(
            (
                "nested writable-family symlink is rejected before extension",
                _expect_value_error(
                    lambda: runner.DebateRun(
                        "openai",
                        1,
                        "test-openai-key",
                        "",
                        resume_from=nested_link_root,
                    ),
                    "unsafe saved run",
                ),
            )
        )

        results_link_root = base / "results-link-results"
        _write_family_metadata(results_link_root)
        outside_results = base / "outside-results"
        outside_results.mkdir()
        (results_link_root / "medical_results").symlink_to(
            outside_results, target_is_directory=True
        )
        checks.append(
            (
                "symlinked derived-results tree is rejected before writes",
                _expect_value_error(
                    lambda: runner.DebateRun(
                        "openai",
                        1,
                        "test-openai-key",
                        "",
                        resume_from=results_link_root,
                    ),
                    "unsafe saved run",
                ),
            )
        )

        outside_fresh = base / "outside-fresh"
        outside_fresh.mkdir()
        symlinked_fresh_base = base / "fresh-base-link"
        symlinked_fresh_base.symlink_to(outside_fresh, target_is_directory=True)
        checks.append(
            (
                "fresh top-level results base may not be a symlink",
                _expect_value_error(
                    lambda: runner.create_results_root(symlinked_fresh_base),
                    "unsafe results base path",
                )
                and not any(outside_fresh.iterdir()),
            )
        )
        symlinked_parent = base / "fresh-parent-link"
        symlinked_parent.symlink_to(outside_fresh, target_is_directory=True)
        checks.append(
            (
                "fresh results base may not pass through a symlink ancestor",
                _expect_value_error(
                    lambda: runner.create_results_root(symlinked_parent / "exp"),
                    "unsafe results base path",
                ),
            )
        )

        sibling_root = base / "sibling-link-results"
        sibling_root.mkdir()
        sibling_target = base / "sibling-target"
        sibling_target.mkdir()
        (sibling_root / "openai").symlink_to(sibling_target, target_is_directory=True)
        sibling_probe = object.__new__(runner.DebateRun)
        sibling_probe.exp_dir = sibling_root
        sibling_probe.family = "anthropic"
        checks.append(
            (
                "combined analysis rejects a symlinked sibling family tree",
                _expect_runtime_error(
                    sibling_probe._combined_analysis_inputs,
                    "cannot safely inspect sibling openai",
                ),
            )
        )

        exp_dir = base / "file-handler-exp"
        exp_dir.mkdir()
        experiment = Experiment(exp_dir, Method.baseline, BaselineType.blind)
        checks.append(
            (
                "judge path traversal is rejected by file handler",
                _expect_value_error(
                    lambda: experiment.get_judge_filename("../../outside"),
                    "unsafe judge name path component",
                ),
            )
        )
        outside_judge = base / "outside-judge"
        outside_judge.mkdir()
        judge_root = exp_dir / "baseline_blind"
        judge_root.mkdir()
        (judge_root / "safe-name").symlink_to(outside_judge, target_is_directory=True)
        checks.append(
            (
                "file handler refuses an existing symlinked judge directory",
                _expect_value_error(
                    lambda: experiment.get_judge_filename("safe-name"),
                    "contains a symlink",
                ),
            )
        )

        real_create_results_root = runner.create_results_root

        # Constructor failures must happen before the session tempdir (and
        # therefore before the API-key file) is created.
        metadata_secret_dir = base / "metadata-failure-session"
        metadata_temp_calls: list[str] = []

        def metadata_failure_mkdtemp(*_args, **_kwargs) -> str:
            metadata_temp_calls.append("called")
            metadata_secret_dir.mkdir()
            return str(metadata_secret_dir)

        metadata_error_raised = False
        with (
            patch.object(
                runner,
                "create_results_root",
                side_effect=lambda: real_create_results_root(
                    base / "metadata-failure-results"
                ),
            ),
            patch.object(
                runner.DebateRun,
                "_write_run_metadata",
                side_effect=OSError("simulated metadata write failure"),
            ),
            patch.object(
                runner.tempfile,
                "mkdtemp",
                side_effect=metadata_failure_mkdtemp,
            ),
        ):
            try:
                runner.DebateRun(
                    "openai",
                    1,
                    "test-openai-key",
                    "",
                    use_canonical_baselines=False,
                )
            except OSError as exc:
                metadata_error_raised = "metadata write failure" in str(exc)
        checks.append(
            (
                "metadata write failure occurs before any API-key tempdir exists",
                metadata_error_raised
                and not metadata_temp_calls
                and not metadata_secret_dir.exists(),
            )
        )

        malformed_resume = base / "malformed-resume"
        malformed_family = malformed_resume / "openai"
        malformed_family.mkdir(parents=True)
        malformed_record = _record("openai")
        malformed_record["baselines_dir"] = str(
            malformed_resume / "baselines" / "openai"
        )
        (malformed_family / "run_metadata.json").write_text(
            json.dumps(malformed_record) + "\n", encoding="utf-8"
        )
        malformed_working = malformed_family / "debate_sim" / "data0.csv"
        malformed_working.parent.mkdir()
        malformed_ids = runner.pilot_case_ids(2)
        pd.DataFrame(
            {"id": [malformed_ids[1], malformed_ids[0]], "complete": [False, False]}
        ).to_csv(malformed_working, index=False)
        resume_secret_dir = base / "resume-failure-session"
        resume_temp_calls: list[str] = []

        def resume_failure_mkdtemp(*_args, **_kwargs) -> str:
            resume_temp_calls.append("called")
            resume_secret_dir.mkdir()
            return str(resume_secret_dir)

        resume_error_raised = False
        with patch.object(
            runner.tempfile, "mkdtemp", side_effect=resume_failure_mkdtemp
        ):
            try:
                runner.DebateRun(
                    "openai",
                    2,
                    "test-openai-key",
                    "",
                    resume_from=malformed_resume,
                )
            except ValueError as exc:
                resume_error_raised = "exact ordered prefix" in str(exc)
        checks.append(
            (
                "malformed resume fails before any API-key tempdir exists",
                resume_error_raised
                and not resume_temp_calls
                and not resume_secret_dir.exists(),
            )
        )

        interrupted_secret_dir = base / "interrupted-secret-session"

        def interrupted_mkdtemp(*_args, **_kwargs) -> str:
            interrupted_secret_dir.mkdir(mode=0o700)
            return str(interrupted_secret_dir)

        secret_interrupt_raised = False
        with (
            patch.object(
                runner,
                "create_results_root",
                side_effect=lambda: real_create_results_root(
                    base / "interrupted-secret-results"
                ),
            ),
            patch.object(runner.tempfile, "mkdtemp", side_effect=interrupted_mkdtemp),
            patch.object(
                runner.os,
                "fdopen",
                side_effect=KeyboardInterrupt("simulated cancellation"),
            ),
        ):
            try:
                runner.DebateRun(
                    "openai",
                    1,
                    "test-openai-key",
                    "",
                    use_canonical_baselines=False,
                )
            except KeyboardInterrupt:
                secret_interrupt_raised = True
        checks.append(
            (
                "BaseException during secret creation removes the private tempdir",
                secret_interrupt_raised and not interrupted_secret_dir.exists(),
            )
        )

        fresh_base = base / "fresh"
        with patch.object(
            runner,
            "create_results_root",
            side_effect=lambda: real_create_results_root(fresh_base),
        ):
            app_run = runner.DebateRun(
                "openai",
                1,
                "test-openai-key",
                "",
                use_canonical_baselines=False,
            )
        captured_env: dict[str, str] = {}
        secret_ready = (
            app_run._secrets_path.exists()
            and app_run._secrets_path.stat().st_mode & 0o777 == 0o600
            and "API_KEY=test-openai-key"
            in app_run._secrets_path.read_text(encoding="utf-8")
        )

        def fake_popen(_cmd, **kwargs):
            captured_env.update(kwargs["env"])
            return _SuccessfulProcess()

        try:
            with patch("app.runner.subprocess.Popen", side_effect=fake_popen):
                rc = app_run._run_stage("guard-marker", [sys.executable, "-V"])
            checks.append(
                (
                    "every app subprocess receives the paid-stage guard marker",
                    rc == 0 and captured_env.get("MEDICAL_DEBATE_APP_RUNNER") == "1",
                )
            )
            prompt_cache = app_run._tempdir / "prompt_history"
            prompt_cache.mkdir()
            (prompt_cache / "provider-prompt.txt").write_text(
                "temporary", encoding="utf-8"
            )
            with patch.object(
                app_run,
                "_run",
                side_effect=lambda: app_run._finalise(runner.Phase.DONE, "Finished."),
            ):
                app_run.start()
                app_run._thread.join(timeout=2)
            checks.append(
                (
                    "normal constructor and start retain then clean session secrets",
                    secret_ready
                    and not app_run._thread.is_alive()
                    and app_run.snapshot().phase is runner.Phase.DONE
                    and not app_run._secrets_path.exists()
                    and not prompt_cache.exists(),
                )
            )
        finally:
            app_run.cleanup()

        def make_fresh_run(path: Path) -> runner.DebateRun:
            with patch.object(
                runner,
                "create_results_root",
                side_effect=lambda: real_create_results_root(path),
            ):
                return runner.DebateRun(
                    "openai",
                    1,
                    "test-openai-key",
                    "",
                    use_canonical_baselines=False,
                )

        thread_failure_run = make_fresh_run(base / "thread-start-failure")
        thread_failure_tempdir = thread_failure_run._tempdir
        thread_failure_secret = thread_failure_run._secrets_path
        thread_failure_raised = False
        with patch.object(
            runner.threading.Thread,
            "start",
            side_effect=RuntimeError("can't start new thread"),
        ):
            try:
                thread_failure_run.start()
            except RuntimeError as exc:
                thread_failure_raised = "can't start new thread" in str(exc)
        checks.append(
            (
                "thread-start failure removes secrets before returning to the UI",
                thread_failure_raised
                and thread_failure_run._thread is None
                and not thread_failure_secret.exists()
                and not thread_failure_tempdir.exists(),
            )
        )

        already_stopped = make_fresh_run(base / "already-stopped")
        try:
            already_stopped.stop()
            with patch("app.runner.subprocess.Popen") as popen:
                stopped_rc = already_stopped._run_stage(
                    "must-not-launch", [sys.executable, "-V"]
                )
            checks.append(
                (
                    "a stage never launches after Stop was already requested",
                    stopped_rc != 0 and not popen.called,
                )
            )
        finally:
            already_stopped.cleanup()

        raced_stop = make_fresh_run(base / "raced-stop")
        raced_process = _TerminableProcess()

        def racing_popen(_cmd, **_kwargs):
            # Reproduce Stop landing after the final precheck but before the
            # new process has been published as ``self._proc``.
            raced_stop.stop()
            return raced_process

        try:
            with patch("app.runner.subprocess.Popen", side_effect=racing_popen):
                raced_rc = raced_stop._run_stage("raced-stop", [sys.executable, "-V"])
            checks.append(
                (
                    "a process launched during the Stop race is terminated",
                    raced_rc != 0 and raced_process.terminated,
                )
            )
        finally:
            raced_stop.cleanup()

        checks.append(
            (
                "live arm completion requires both explicit flags and A/B answers",
                runner.arm_judgement_is_complete(
                    {
                        "orig": "Answer: A",
                        "swap": "Answer: B",
                        "orig_complete": True,
                        "swap_complete": True,
                    }
                )
                and not runner.arm_judgement_is_complete(
                    {
                        "orig": "Error occurred while judging",
                        "swap": "Answer: B",
                        "orig_complete": True,
                        "swap_complete": True,
                    }
                )
                and not runner.arm_judgement_is_complete(
                    {
                        "orig": "Answer: A",
                        "swap": "Answer: B",
                        "orig_complete": True,
                        "swap_complete": False,
                    }
                ),
            )
        )

        live_root = base / "live-case-status"
        live_family = live_root / "openai"
        live_case_id = runner.pilot_case_ids(1)[0]
        status_probe = object.__new__(runner.DebateRun)
        status_probe.family_dir = live_family
        status_probe.frontier_model = runner.FAMILY_MODELS["openai"]["frontier"]
        status_probe.weaker_model = runner.FAMILY_MODELS["openai"]["weaker"]
        status_probe._initial_cases = [
            runner.CaseStatus(
                case_id=live_case_id,
                question="question",
                correct="correct",
                distractor="distractor",
            )
        ]

        def write_live_judgement(
            condition: str,
            judge: str,
            filename: str,
            answer: str,
            complete: bool,
        ) -> None:
            path = live_family / "debate_sim" / f"{condition}_{judge}" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "id": [live_case_id],
                    "answer_judge": [answer],
                    "complete_judge": [complete],
                }
            ).to_csv(path, index=False)

        write_live_judgement(
            "e1_info_asymmetry",
            status_probe.frontier_model,
            "data0_judgement.csv",
            "Answer: A",
            False,
        )
        write_live_judgement(
            "e1_info_asymmetry",
            status_probe.frontier_model,
            "data0_swap_judgement.csv",
            "Answer: B",
            True,
        )
        write_live_judgement(
            "e2_double_asymmetry",
            status_probe.weaker_model,
            "data0_judgement.csv",
            "Error occurred while judging",
            True,
        )
        write_live_judgement(
            "e2_double_asymmetry",
            status_probe.weaker_model,
            "data0_swap_judgement.csv",
            "Answer: B",
            True,
        )
        write_live_judgement(
            "e3_capability_asymmetry",
            status_probe.weaker_model,
            "data0_judgement.csv",
            "Answer: A",
            True,
        )
        write_live_judgement(
            "e3_capability_asymmetry",
            status_probe.weaker_model,
            "data0_swap_judgement.csv",
            "Answer: B",
            True,
        )
        # A stale judge directory must not get mixed into the configured arm.
        write_live_judgement(
            "e4_full_symmetry",
            "stale-model",
            "data0_judgement.csv",
            "Answer: A",
            True,
        )
        live_answers = status_probe._read_cases()[0].arm_answers
        checks.append(
            (
                "live status reads configured judge only and retains completion flags",
                not runner.arm_judgement_is_complete(live_answers["e1_info_asymmetry"])
                and not runner.arm_judgement_is_complete(
                    live_answers["e2_double_asymmetry"]
                )
                and runner.arm_judgement_is_complete(
                    live_answers["e3_capability_asymmetry"]
                )
                and "e4_full_symmetry" not in live_answers,
            )
        )

    failed = False
    for description, condition in checks:
        print(f"{'ok   ' if condition else 'FAIL '} [{description}]")
        failed = failed or not condition
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
