"""Regression tests for terminal debate and judge stage failures."""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd

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

        with (
            patch.dict(
                debate_module.os.environ,
                {debate_module.APP_RUNNER_ENV: "1"},
            ),
            patch.object(debate_module, "async_main", async_main),
        ):
            with self.assertRaisesRegex(RuntimeError, "Debate stage did not complete"):
                debate_module.main.__wrapped__(cfg)

        async_main.assert_awaited_once_with(cfg)

    def test_incomplete_judge_is_a_task_failure(self):
        cfg = object()
        async_main = AsyncMock(return_value=False)

        with (
            patch.dict(
                judge_module.os.environ,
                {judge_module.APP_RUNNER_ENV: "1"},
            ),
            patch.object(judge_module, "async_main", async_main),
        ):
            with self.assertRaisesRegex(RuntimeError, "Judge stage did not complete"):
                judge_module.main.__wrapped__(cfg)

        async_main.assert_awaited_once_with(cfg)

    def test_successful_entrypoints_still_return_true(self):
        for module in (debate_module, judge_module):
            with self.subTest(module=module.__name__):
                async_main = AsyncMock(return_value=True)
                with (
                    patch.dict(module.os.environ, {module.APP_RUNNER_ENV: "1"}),
                    patch.object(module, "async_main", async_main),
                ):
                    self.assertIs(module.main.__wrapped__(object()), True)

    def test_paid_entrypoints_reject_direct_execution(self):
        for module in (debate_module, judge_module):
            with self.subTest(module=module.__name__):
                with patch.dict(module.os.environ, {}, clear=True):
                    with self.assertRaisesRegex(RuntimeError, "Streamlit app"):
                        module.main.__wrapped__(object())


class IncrementalPersistenceTests(unittest.TestCase):
    @staticmethod
    async def _wait_for_csv_value(
        path: Path,
        column: str,
        index: int,
        expected: str,
    ) -> None:
        for _ in range(200):
            saved = pd.read_csv(path, keep_default_na=False)
            if column in saved and str(saved.loc[index, column]).lower() == expected:
                return
            await asyncio.sleep(0.005)
        raise AssertionError(f"{column}[{index}] never became {expected}")

    def test_debate_writes_each_finished_case_before_batch_finishes(self):
        async def scenario(path: Path) -> None:
            release_second = asyncio.Event()

            class Rollout:
                async def run(self, index, row, swap=False):
                    if index == 1:
                        await release_second.wait()
                    return {"transcript": f"done-{index}", "complete": True}

            stage = asyncio.create_task(
                debate_module.run_debates(Rollout(), path, num_debate_threads=2)
            )
            await self._wait_for_csv_value(path, "complete", 0, "true")
            self.assertFalse(
                stage.done(), "the second API call should still be pending"
            )

            saved_while_running = pd.read_csv(path, keep_default_na=False)
            self.assertEqual(saved_while_running.loc[0, "transcript"], "done-0")

            release_second.set()
            self.assertEqual(await stage, (True, 2))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            pd.DataFrame(
                [
                    {"id": "one", "complete": False, "transcript": ""},
                    {"id": "two", "complete": False, "transcript": ""},
                ]
            ).to_csv(path, index=False)
            asyncio.run(scenario(path))

    def test_judge_interruption_keeps_finished_case_and_resume_skips_it(self):
        async def scenario(path: Path) -> None:
            never_release = asyncio.Event()

            class FirstJudge:
                async def make_decision(self, index, row, swap=False, round_limit=None):
                    if index == 1:
                        await never_release.wait()
                    return {
                        "judgement": f"Answer: {'A' if index == 0 else 'B'}",
                        "complete": True,
                        "transcript": row["transcript"],
                    }

            interrupted = asyncio.create_task(
                judge_module.run_judge(FirstJudge(), path, anthropic_num_threads=2)
            )
            await self._wait_for_csv_value(path, "complete_judge", 0, "true")
            interrupted.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await interrupted

            calls: list[int] = []

            class ResumeJudge:
                async def make_decision(self, index, row, swap=False, round_limit=None):
                    calls.append(index)
                    return {
                        "judgement": "Answer: B",
                        "complete": True,
                        "transcript": row["transcript"],
                    }

            outcome = await judge_module.run_judge(
                ResumeJudge(), path, anthropic_num_threads=2
            )
            self.assertEqual(outcome, (True, 1))
            self.assertEqual(calls, [1], "the persisted paid result must not rerun")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judgements.csv"
            pd.DataFrame(
                [
                    {"id": "one", "complete": True, "transcript": "t0"},
                    {"id": "two", "complete": True, "transcript": "t1"},
                ]
            ).to_csv(path, index=False)
            asyncio.run(scenario(path))


class JudgeAnswerValidationTests(unittest.TestCase):
    def test_unparseable_answer_stays_incomplete_and_resume_retries_it(self):
        async def scenario(path: Path) -> None:
            calls: list[int] = []

            class TruncatedJudge:
                async def make_decision(self, index, row, swap=False, round_limit=None):
                    calls.append(index)
                    return {
                        "judgement": "The evidence favours the first diagnosis, but",
                        "complete": True,
                        "transcript": row["transcript"],
                    }

            first_outcome = await judge_module.run_judge(
                TruncatedJudge(), path, anthropic_num_threads=1
            )
            self.assertEqual(first_outcome, (False, 0))
            first_saved = pd.read_csv(path, keep_default_na=False)
            self.assertFalse(bool(first_saved.loc[0, "complete_judge"]))

            class CompleteJudge:
                async def make_decision(self, index, row, swap=False, round_limit=None):
                    calls.append(index)
                    return {
                        "judgement": "Reasoning complete.\nAnswer: A",
                        "complete": True,
                        "transcript": row["transcript"],
                    }

            resumed_outcome = await judge_module.run_judge(
                CompleteJudge(), path, anthropic_num_threads=1
            )
            self.assertEqual(resumed_outcome, (True, 1))
            self.assertEqual(calls, [0, 0], "the incomplete answer must be retried")
            resumed = pd.read_csv(path, keep_default_na=False)
            self.assertTrue(bool(resumed.loc[0, "complete_judge"]))
            self.assertEqual(
                resumed.loc[0, "answer_judge"],
                "Reasoning complete.\nAnswer: A",
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judgements.csv"
            pd.DataFrame([{"id": "one", "complete": True, "transcript": "t0"}]).to_csv(
                path, index=False
            )
            asyncio.run(scenario(path))

    def test_ambiguous_legacy_answer_is_not_complete(self):
        response = (
            "The correct answer is most likely to be (A), but elsewhere the "
            "correct answer is most likely to be (B)."
        )
        self.assertFalse(judge_module._parseable_ab_judgement(response))

    def test_resume_repairs_legacy_complete_flag_and_retries_bad_answer(self):
        async def scenario(path: Path) -> None:
            calls: list[int] = []

            class CompleteJudge:
                async def make_decision(self, index, row, swap=False, round_limit=None):
                    calls.append(index)
                    return {
                        "judgement": "Answer: B",
                        "complete": True,
                        "transcript": row["transcript"],
                    }

            outcome = await judge_module.run_judge(
                CompleteJudge(), path, anthropic_num_threads=1
            )
            self.assertEqual(outcome, (True, 1))
            self.assertEqual(calls, [0])
            saved = pd.read_csv(path, keep_default_na=False)
            self.assertTrue(bool(saved.loc[0, "complete_judge"]))
            self.assertEqual(saved.loc[0, "answer_judge"], "Answer: B")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            pd.DataFrame(
                [
                    {
                        "id": "one",
                        "complete": True,
                        "transcript": "t0",
                        "complete_judge": True,
                        "answer_judge": "truncated before the final answer",
                    }
                ]
            ).to_csv(path, index=False)
            asyncio.run(scenario(path))

    def test_concession_y_n_is_not_subject_to_answer_parser(self):
        async def scenario(path: Path) -> None:
            class ConcessionJudge:
                async def make_decision(self, index, row, swap=False, round_limit=None):
                    return {
                        "judgement": "Y",
                        "complete": True,
                        "transcript": row["transcript"],
                    }

            outcome = await judge_module.run_judge(
                ConcessionJudge(),
                path,
                anthropic_num_threads=1,
                judge_type=judge_module.JudgeType.concession,
            )
            self.assertEqual(outcome, (True, 1))
            saved = pd.read_csv(path, keep_default_na=False)
            self.assertTrue(bool(saved.loc[0, "complete_concession"]))
            self.assertEqual(saved.loc[0, "answer_concession"], "Y")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concession.csv"
            pd.DataFrame([{"id": "one", "complete": True, "transcript": "t0"}]).to_csv(
                path, index=False
            )
            asyncio.run(scenario(path))


class JudgementReconciliationTests(unittest.TestCase):
    def test_reordered_rows_are_reconciled_by_id_and_missing_rows_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.csv"
            judgement_path = root / "judgement.csv"
            pd.DataFrame(
                [
                    {
                        "id": "case-a",
                        "question": "new question a",
                        "complete": True,
                        "transcript": "new transcript a",
                    },
                    {
                        "id": "case-b",
                        "question": "new question b",
                        "complete": True,
                        "transcript": "new transcript b",
                    },
                    {
                        "id": "case-c",
                        "question": "new question c",
                        "complete": True,
                        "transcript": "new transcript c",
                    },
                ]
            ).to_csv(data_path, index=False)
            pd.DataFrame(
                [
                    {
                        "id": "case-b",
                        "question": "seed question b",
                        "complete": False,
                        "transcript": "old transcript b",
                        "complete_judge": True,
                        "answer_judge": "Answer: B",
                    },
                    {
                        "id": "case-a",
                        "question": "seed question a",
                        "complete": False,
                        "transcript": "old transcript a",
                        "complete_judge": False,
                        "answer_judge": "partial-a",
                    },
                ]
            ).to_csv(judgement_path, index=False)

            judge_module._reconcile_judgement_with_data(data_path, judgement_path)

            saved = pd.read_csv(judgement_path, keep_default_na=False)
            self.assertEqual(saved["id"].tolist(), ["case-a", "case-b", "case-c"])
            by_id = saved.set_index("id")
            self.assertEqual(by_id.loc["case-a", "answer_judge"], "partial-a")
            self.assertFalse(bool(by_id.loc["case-a", "complete_judge"]))
            self.assertEqual(by_id.loc["case-b", "answer_judge"], "Answer: B")
            self.assertTrue(bool(by_id.loc["case-b", "complete_judge"]))
            self.assertEqual(by_id.loc["case-a", "transcript"], "new transcript a")
            self.assertEqual(by_id.loc["case-b", "transcript"], "new transcript b")
            self.assertEqual(by_id.loc["case-c", "transcript"], "new transcript c")
            self.assertTrue(saved["complete"].astype(str).str.lower().eq("true").all())
            self.assertEqual(by_id.loc["case-c", "answer_judge"], "")
            self.assertFalse(bool(by_id.loc["case-c", "complete_judge"]))
            self.assertEqual(by_id.loc["case-c", "question"], "new question c")

    def test_malformed_or_foreign_ids_fail_without_touching_judgement(self):
        malformed_cases = {
            "duplicate data ID": (
                ["case-a", "case-a"],
                ["case-a"],
            ),
            "duplicate judgement ID": (
                ["case-a", "case-b"],
                ["case-a", "case-a"],
            ),
            "blank judgement ID": (
                ["case-a", "case-b"],
                ["case-a", ""],
            ),
            "foreign judgement ID": (
                ["case-a", "case-b"],
                ["case-a", "case-x"],
            ),
        }
        for label, (data_ids, judgement_ids) in malformed_cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_path = root / "data.csv"
                judgement_path = root / "judgement.csv"
                pd.DataFrame(
                    {
                        "id": data_ids,
                        "complete": [True] * len(data_ids),
                        "transcript": [f"data-{i}" for i in range(len(data_ids))],
                    }
                ).to_csv(data_path, index=False)
                pd.DataFrame(
                    {
                        "id": judgement_ids,
                        "complete": [True] * len(judgement_ids),
                        "transcript": [f"judge-{i}" for i in range(len(judgement_ids))],
                        "complete_judge": [True] * len(judgement_ids),
                        "answer_judge": ["Answer: A"] * len(judgement_ids),
                    }
                ).to_csv(judgement_path, index=False)
                original = judgement_path.read_bytes()

                with self.assertRaises(ValueError):
                    judge_module._reconcile_judgement_with_data(
                        data_path, judgement_path
                    )

                self.assertEqual(judgement_path.read_bytes(), original)

    def test_missing_id_column_fails_without_touching_judgement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.csv"
            judgement_path = root / "judgement.csv"
            pd.DataFrame(
                [{"id": "case-a", "complete": True, "transcript": "data"}]
            ).to_csv(data_path, index=False)
            pd.DataFrame(
                [{"complete": True, "transcript": "judge", "answer_judge": "A"}]
            ).to_csv(judgement_path, index=False)
            original = judgement_path.read_bytes()

            with self.assertRaisesRegex(ValueError, "missing an id/case_id"):
                judge_module._reconcile_judgement_with_data(data_path, judgement_path)

            self.assertEqual(judgement_path.read_bytes(), original)

    def test_failed_atomic_replace_leaves_original_judgement_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.csv"
            judgement_path = root / "judgement.csv"
            pd.DataFrame(
                [
                    {"id": "case-a", "complete": True, "transcript": "new-a"},
                    {"id": "case-b", "complete": True, "transcript": "new-b"},
                ]
            ).to_csv(data_path, index=False)
            pd.DataFrame(
                [
                    {
                        "id": "case-a",
                        "complete": True,
                        "transcript": "old-a",
                        "complete_judge": True,
                        "answer_judge": "Answer: A",
                    }
                ]
            ).to_csv(judgement_path, index=False)
            original = judgement_path.read_bytes()

            with patch.object(Path, "replace", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    judge_module._reconcile_judgement_with_data(
                        data_path, judgement_path
                    )

            self.assertEqual(judgement_path.read_bytes(), original)
            self.assertEqual(
                list(root.glob(f".{judgement_path.name}.*.tmp")),
                [],
                "failed writes must not leave checkpoint temp files behind",
            )


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
