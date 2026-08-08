"""Regression tests for terminal debate and judge stage failures."""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import debate as debate_module  # noqa: E402
from core import judge as judge_module  # noqa: E402


class _FakeExperiment:
    def __init__(self, data_file: Path, judgement_file: Path | None = None):
        self.data_file = data_file
        self.judgement_file = judgement_file
        self.debate_swaps = []
        self.judge_swaps = []

    def get_debate_filename(self, *, seed, swap):
        self.debate_swaps.append(swap)
        return self.data_file

    def get_judge_filename(self, judge_name, *, seed, swap, exp_suffix):
        self.judge_swaps.append(swap)
        if self.judgement_file is None:
            raise AssertionError("A judgement path was not configured")
        return self.judgement_file


class EntrypointFailureTests(unittest.TestCase):
    def test_incomplete_debate_is_a_task_failure(self):
        cfg = object()
        async_main = AsyncMock(return_value=False)

        with patch.object(debate_module, "async_main", async_main):
            with self.assertRaisesRegex(RuntimeError, "Debate stage did not complete"):
                debate_module.main.__wrapped__(cfg)

        async_main.assert_awaited_once_with(cfg)

    def test_incomplete_judge_is_a_task_failure(self):
        cfg = object()
        async_main = AsyncMock(return_value=False)

        with patch.object(judge_module, "async_main", async_main):
            with self.assertRaisesRegex(RuntimeError, "Judge stage did not complete"):
                judge_module.main.__wrapped__(cfg)

        async_main.assert_awaited_once_with(cfg)

    def test_successful_entrypoints_still_return_true(self):
        for module in (debate_module, judge_module):
            with self.subTest(module=module.__name__):
                async_main = AsyncMock(return_value=True)
                with patch.object(module, "async_main", async_main):
                    self.assertIs(module.main.__wrapped__(object()), True)


class SwapShortCircuitTests(unittest.TestCase):
    def test_failed_original_debate_does_not_start_swapped_debate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_file = root / "data.csv"
            data_file.write_text("complete,transcript\nTrue,done\n", encoding="utf-8")
            experiment = _FakeExperiment(data_file)
            retry = AsyncMock(return_value=False)
            cfg = SimpleNamespace(
                logging="info",
                anthropic_tag="ANTHROPIC_API_KEY",
                openai_tag="API_KEY",
                anthropic_num_threads=5,
                openai_fraction_rate_limit=1.0,
                print_prompt_and_response=False,
                organization=None,
                exp_dir=str(root),
                method="debate",
                method_type="sim",
                use_intermediary=False,
                seed=0,
                limit=1,
                correct_debater=SimpleNamespace(BoN=4),
                incorrect_debater=SimpleNamespace(BoN=4),
            )

            with (
                patch.object(debate_module, "setup_environment"),
                patch.object(debate_module, "delete_old_prompt_files"),
                patch.object(debate_module, "ModelAPI", return_value=object()),
                patch.object(debate_module, "Experiment", return_value=experiment),
                patch.object(debate_module, "setup_debate", return_value=Mock()),
                patch.object(debate_module, "async_function_with_retry", retry),
            ):
                complete = asyncio.run(debate_module.async_main(cfg))

        self.assertFalse(complete)
        self.assertEqual(experiment.debate_swaps, [False])
        retry.assert_awaited_once()

    def test_failed_original_judge_does_not_start_swapped_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_file = root / "data.csv"
            judgement_file = root / "judge" / "data_judgement.csv"
            data_file.write_text("complete,transcript\nTrue,done\n", encoding="utf-8")
            experiment = _FakeExperiment(data_file, judgement_file)
            retry = AsyncMock(return_value=False)
            cfg = SimpleNamespace(
                logging="info",
                anthropic_tag="ANTHROPIC_API_KEY",
                openai_tag="API_KEY",
                seed=0,
                anthropic_num_threads=5,
                openai_fraction_rate_limit=1.0,
                print_prompt_and_response=False,
                organization=None,
                exp_dir=str(root),
                method="debate",
                method_type="sim",
                use_intermediary=False,
                judge_type=judge_module.JudgeType.default,
                judge=SimpleNamespace(n_vote=0),
                judge_name="test-judge",
                round_limit=None,
                limit=1,
            )

            with (
                patch.object(judge_module, "setup_environment"),
                patch.object(judge_module, "delete_old_prompt_files"),
                patch.object(judge_module, "ModelAPI", return_value=object()),
                patch.object(judge_module, "Experiment", return_value=experiment),
                patch.object(judge_module, "setup_judge", return_value=Mock()),
                patch.object(judge_module, "async_function_with_retry", retry),
            ):
                complete = asyncio.run(judge_module.async_main(cfg))

        self.assertFalse(complete)
        self.assertEqual(experiment.debate_swaps, [False])
        self.assertEqual(experiment.judge_swaps, [False])
        retry.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
