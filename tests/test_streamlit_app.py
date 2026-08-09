"""No-network Streamlit confirmation and family-resume regressions."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import runner  # noqa: E402


class _LaunchForbidden:
    def __init__(self, *args, **kwargs):
        raise AssertionError("AppTest must never construct DebateRun")


class _SetupOSError:
    def __init__(self, *args, **kwargs):
        raise OSError("simulated metadata write failure")


class _CompletedStatusRun:
    def snapshot(self):
        return runner.RunSnapshot(
            phase=runner.Phase.DONE,
            message="done",
            cases=[
                runner.CaseStatus(
                    case_id="case-1",
                    question="question",
                    correct="correct",
                    distractor="distractor",
                    debate_complete=True,
                    arm_answers={
                        "e1_info_asymmetry": {
                            "orig": "Answer: A",
                            "swap": "Answer: B",
                            "orig_complete": False,
                            "swap_complete": True,
                        },
                        "e2_double_asymmetry": {
                            "orig": "Error occurred while judging",
                            "swap": "Answer: B",
                            "orig_complete": True,
                            "swap_complete": True,
                        },
                        "e3_capability_asymmetry": {
                            "orig": "Answer: A",
                            "swap": "Answer: B",
                            "orig_complete": True,
                            "swap_complete": True,
                        },
                    },
                )
            ],
            log_tail=[],
            started_at=1.0,
        )

    def stop(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


def _button(at: AppTest, *prefixes: str):
    return next(button for button in at.button if button.label.startswith(prefixes))


def _captions(at: AppTest) -> list[str]:
    return [caption.value for caption in at.caption]


class StreamlitAppTests(unittest.TestCase):
    def assert_clean(self, at: AppTest) -> None:
        self.assertFalse(at.exception, [item.value for item in at.exception])
        self.assertIsNone(at.session_state["run"])

    def test_family_resume_and_confirmation_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical_root = root / "canonical"
            openai_run = root / "2026-01-01_00-00-00_results"
            anthropic_run = root / "2026-01-02_00-00-00_results"
            for path in (canonical_root / "openai", canonical_root / "anthropic"):
                path.mkdir(parents=True)
            anthropic_baselines = anthropic_run / "baselines" / "anthropic"
            anthropic_baselines.mkdir(parents=True)
            openai_run.mkdir()

            latest = {
                "openai": {
                    "exp_dir": openai_run,
                    "family": "openai",
                    "n_cases": 100,
                    "frontier_model": "saved-openai-frontier",
                    "weaker_model": "saved-openai-weaker",
                    "concession_model": "saved-openai-qc",
                    "recorded_at_utc": "2026-01-01T00:00:00Z",
                    "last_updated_at_utc": "2026-01-01T00:00:00Z",
                    "baselines_dir": canonical_root / "openai",
                    "using_canonical_baselines": True,
                    "baselines_writable": False,
                },
                "anthropic": {
                    "exp_dir": anthropic_run,
                    "family": "anthropic",
                    "n_cases": 2,
                    "frontier_model": "saved-anthropic-frontier",
                    "weaker_model": "saved-anthropic-weaker",
                    "concession_model": "saved-anthropic-qc",
                    "recorded_at_utc": "2026-01-02T00:00:00Z",
                    "last_updated_at_utc": "2026-01-02T00:00:00Z",
                    "baselines_dir": anthropic_baselines,
                    "using_canonical_baselines": False,
                    "baselines_writable": True,
                },
            }

            def fake_latest(base_dir=runner.REPO_ROOT / "exp", family=None):
                return latest.get(family)

            def fake_canonical(family, *args, **kwargs):
                return canonical_root / family

            with (
                patch.object(runner, "find_latest_run", side_effect=fake_latest),
                patch.object(
                    runner, "canonical_baselines_path", side_effect=fake_canonical
                ),
                patch.object(
                    runner,
                    "pilot_case_ids",
                    side_effect=lambda n: tuple(f"case-{i}" for i in range(n)),
                ),
                patch.object(
                    runner,
                    "estimate_cost_usd",
                    side_effect=lambda family, n, include_baseline_recompute=False: (
                        (3.0 if include_baseline_recompute else 2.5) * n
                    ),
                ),
                patch.object(runner, "DebateRun", _LaunchForbidden),
            ):
                at = AppTest.from_file(
                    str(REPO_ROOT / "app" / "streamlit_app.py"),
                    default_timeout=10,
                ).run()
                self.assert_clean(at)
                self.assertEqual(at.selectbox[0].value, "openai")
                self.assertTrue(
                    any("saved-openai-frontier" in text for text in _captions(at))
                )
                self.assertFalse(
                    any(
                        slider.label == "Total cases after this run"
                        for slider in at.slider
                    )
                )

                at.selectbox[0].set_value("anthropic").run()
                self.assert_clean(at)
                self.assertTrue(
                    any("saved-anthropic-frontier" in text for text in _captions(at))
                )
                self.assertTrue(
                    any(
                        slider.label == "Total cases after this run"
                        for slider in at.slider
                    )
                )

                # Fresh first click arms only; recompute change cancels it.
                self.assertIn("~$7.50", _button(at, "▶ Run").label)
                _button(at, "▶ Run").click().run()
                self.assert_clean(at)
                self.assertEqual(
                    at.session_state["confirm_signature"],
                    ("anthropic", 3, False),
                )
                _button(at, "✅ Confirm: spend ~")
                next(
                    toggle
                    for toggle in at.toggle
                    if toggle.label.startswith("Recompute baselines")
                ).set_value(True).run()
                self.assert_clean(at)
                self.assertFalse(at.session_state["confirm_armed"])
                self.assertIsNone(at.session_state["confirm_signature"])
                self.assertIn("~$9.00", _button(at, "▶ Run").label)

                # Resume first click arms only and binds saved models/source.
                self.assertIn("~$6.00", _button(at, "↻ Resume", "➕ Extend").label)
                _button(at, "↻ Resume", "➕ Extend").click().run()
                self.assert_clean(at)
                signature = at.session_state["resume_confirm_signature"]
                self.assertTrue(at.session_state["resume_confirm_armed"])
                self.assertEqual(signature[0], "anthropic")
                self.assertEqual(signature[1], str(anthropic_run.resolve()))
                self.assertEqual(signature[2], 2)
                self.assertEqual(
                    signature[4:7],
                    (
                        "saved-anthropic-frontier",
                        "saved-anthropic-weaker",
                        "saved-anthropic-qc",
                    ),
                )
                self.assertEqual(signature[7], str(anthropic_baselines))
                self.assertTrue(signature[8])
                self.assertIn("~$6.00", _button(at, "✅ Confirm: budget ~").label)

                # Changing target cancels; re-arm then changing family cancels.
                next(
                    slider
                    for slider in at.slider
                    if slider.label == "Total cases after this run"
                ).set_value(3).run()
                self.assert_clean(at)
                self.assertFalse(at.session_state["resume_confirm_armed"])
                self.assertIsNone(at.session_state["resume_confirm_signature"])

                _button(at, "➕ Extend").click().run()
                self.assert_clean(at)
                self.assertEqual(at.session_state["resume_confirm_signature"][2], 3)
                at.selectbox[0].set_value("openai").run()
                self.assert_clean(at)
                self.assertFalse(at.session_state["resume_confirm_armed"])
                self.assertIsNone(at.session_state["resume_confirm_signature"])

    def test_clear_run_clears_both_password_widgets(self) -> None:
        with (
            patch.object(runner, "find_latest_run", return_value=None),
            patch.object(runner, "canonical_baselines_path", return_value=None),
            patch.object(runner, "DebateRun", _LaunchForbidden),
        ):
            at = AppTest.from_file(
                str(REPO_ROOT / "app" / "streamlit_app.py"),
                default_timeout=10,
            ).run()
            next(field for field in at.text_input if field.label == "OpenAI key").input(
                "test-openai-key"
            ).run()
            next(
                field for field in at.text_input if field.label == "Anthropic key"
            ).input("test-anthropic-key").run()

            self.assertEqual(at.session_state["openai_api_key"], "test-openai-key")
            self.assertEqual(
                at.session_state["anthropic_api_key"], "test-anthropic-key"
            )
            _button(at, "Clear run").click().run()

        self.assert_clean(at)
        self.assertEqual(at.session_state["openai_api_key"], "")
        self.assertEqual(at.session_state["anthropic_api_key"], "")
        self.assertEqual(
            next(field for field in at.text_input if field.label == "OpenAI key").value,
            "",
        )
        self.assertEqual(
            next(
                field for field in at.text_input if field.label == "Anthropic key"
            ).value,
            "",
        )

    def test_setup_io_error_is_shown_without_crashing_the_app(self) -> None:
        with (
            patch.object(runner, "find_latest_run", return_value=None),
            patch.object(runner, "canonical_baselines_path", return_value=None),
            patch.object(runner, "DebateRun", _SetupOSError),
        ):
            at = AppTest.from_file(
                str(REPO_ROOT / "app" / "streamlit_app.py"),
                default_timeout=10,
            ).run()
            next(field for field in at.text_input if field.label == "OpenAI key").input(
                "test-openai-key"
            ).run()
            _button(at, "▶ Run").click().run()
            _button(at, "✅ Confirm: spend ~").click().run()

        self.assert_clean(at)
        self.assertTrue(at.error)
        self.assertIn("Could not start the run", at.error[0].value)
        self.assertIn("simulated metadata write failure", at.error[0].value)

    def test_live_arm_counts_and_verdicts_require_completed_parseable_pairs(
        self,
    ) -> None:
        with (
            patch.object(runner, "find_latest_run", return_value=None),
            patch.object(runner, "canonical_baselines_path", return_value=None),
            patch.object(runner, "DebateRun", _LaunchForbidden),
        ):
            at = AppTest.from_file(
                str(REPO_ROOT / "app" / "streamlit_app.py"),
                default_timeout=10,
            ).run()
            at.session_state["run"] = _CompletedStatusRun()
            at.session_state["selected_case"] = "case-1"
            at.run()

        self.assertFalse(at.exception, [item.value for item in at.exception])
        self.assertEqual(at.dataframe[0].value.loc[0, "arms judged"], "1/4")
        detail = at.dataframe[1].value
        self.assertEqual(
            tuple(detail.loc[0, ["orig pick", "swap pick", "result"]]),
            ("—", "B", "partial"),
        )
        self.assertEqual(
            tuple(detail.loc[1, ["orig pick", "swap pick", "result"]]),
            ("?", "B", "unparseable"),
        )
        self.assertEqual(detail.loc[2, "result"], "✓ correct")


if __name__ == "__main__":
    unittest.main()
