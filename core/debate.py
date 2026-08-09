import asyncio
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from core.create_agents import setup_debate
from core.file_handler import Experiment
from core.llm_api.llm import ModelAPI
from core.load.medical import main as medical_loader
from core.rollouts.rollout_base import RolloutBase
from core.utils import (
    async_function_with_retry,
    delete_old_prompt_files,
    setup_environment,
)

LOGGER = logging.getLogger(__name__)
APP_RUNNER_ENV = "MEDICAL_DEBATE_APP_RUNNER"


def _require_app_runner() -> None:
    """Refuse direct paid-stage execution outside the Streamlit runner.

    This is a product guardrail, not an authentication boundary: the marker is
    deliberately just an environment variable set by ``app.runner`` for the
    subprocesses it owns.  It prevents the old, unsupported ``python -m``
    workflow from accidentally bypassing the app's cost confirmation and
    resume checks.
    """
    if os.environ.get(APP_RUNNER_ENV) != "1":
        raise RuntimeError(
            "Direct debate execution is unsupported. Start the Streamlit app "
            "and run the experiment there."
        )


def _atomic_write_csv(df: pd.DataFrame, filename: Path) -> None:
    """Replace ``filename`` atomically with a complete CSV snapshot."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{filename.name}.",
            suffix=".tmp",
            dir=filename.parent,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            df.to_csv(tmp, index=False)
        tmp_path.replace(filename)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _completed_mask(series: pd.Series) -> pd.Series:
    """Normalise booleans reloaded from partially populated CSV columns."""
    return series.astype(str).str.strip().str.lower().eq("true")


async def load_dataset(cfg: DictConfig, filename: Path):
    if cfg.dataset != "medical":
        raise ValueError(
            f"Unsupported dataset {cfg.dataset!r}. This fork only supports the medical DDXPlus pipeline."
        )

    medical_loader(
        filename,
        source_path=cfg.medical_source_path,
        limit=cfg.limit,
    )


async def run_debates(
    rollout: RolloutBase,
    filename: Path,
    swap: bool = False,
    limit: int = None,
    num_debate_threads: int = 20,
):
    # Load the whole working dataset. ``limit`` selects the target cohort but
    # must never truncate rows already present in a resumable CSV.
    full_df = pd.read_csv(filename, encoding="utf-8", keep_default_na=False)
    if "complete" not in full_df.columns:
        full_df["complete"] = False
    if "transcript" not in full_df.columns:
        full_df["transcript"] = ""
    target_df = full_df.head(int(limit)) if limit is not None else full_df
    target_indices = target_df.index
    df = target_df[~_completed_mask(target_df["complete"])]

    start_time = datetime.now()
    LOGGER.info(f"Processing {filename}")
    LOGGER.info(f"Time is {start_time}")
    LOGGER.info(f"Processing {len(df)} rows")

    async def bounded_run(sem, index, row, swap):
        async with sem:
            try:
                result = await rollout.run(index, row, swap=swap)
                # There is deliberately no ``await`` between receiving the
                # paid response and replacing the CSV. All coroutines share
                # this event-loop thread, so these synchronous updates form a
                # single-writer sequence without competing dataframe writes.
                full_df.at[index, "transcript"] = result["transcript"]
                full_df.at[index, "complete"] = bool(result["complete"])
                _atomic_write_csv(full_df, filename)
                return bool(result["complete"])
            except asyncio.CancelledError:
                raise
            except Exception:
                # One malformed/erroring case must not discard successfully
                # persisted siblings from the same asynchronous batch.
                LOGGER.exception("Debate case %s failed", index)
                return False

    sem = asyncio.Semaphore(num_debate_threads)
    tasks = [bounded_run(sem, index, row, swap) for index, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    correct = sum(results)
    LOGGER.info(f"Processed {len(results)} rows. {correct}  were complete.")
    LOGGER.info(f"Time taken: {datetime.now() - start_time}")

    # `correct` is how many of this attempt's incomplete rows finished. The
    # retry loop uses it to tell a partial failure worth retrying from a total
    # one that would just re-bill the same failing calls.
    target_complete = _completed_mask(full_df.loc[target_indices, "complete"]).all()
    return bool(target_complete), int(correct)


@hydra.main(version_base=None, config_path="config/", config_name="config")
def main(cfg: DictConfig):
    _require_app_runner()
    complete = asyncio.run(async_main(cfg))
    if not complete:
        raise RuntimeError(
            "Debate stage did not complete; see the preceding errors for details."
        )
    return True


async def async_main(cfg: DictConfig):
    setup_environment(
        logger_level=cfg.logging,
        anthropic_tag=cfg.anthropic_tag,
        openai_tag=cfg.openai_tag,
    )
    delete_old_prompt_files()
    api_handler = ModelAPI(
        anthropic_num_threads=cfg.anthropic_num_threads,
        openai_fraction_rate_limit=cfg.openai_fraction_rate_limit,
        print_prompt_and_response=cfg.print_prompt_and_response,
        organization=cfg.organization,
    )
    exp_dir = Path(cfg.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    experiment = Experiment(
        exp_dir,
        cfg.method,
        cfg.method_type,
        cfg.use_intermediary,
    )
    if cfg.method == "baseline":
        assert cfg.rollout.num_steps == 0, "Baseline requires num_steps to be 0"
    if cfg.method_type == "seq":
        assert (
            cfg.rollout.rollout_type == "quality_seq"
        ), "You didn't set the right rollout config for sequential!"

    # Load dataset
    filename = experiment.get_debate_filename(seed=cfg.seed, swap=False)
    filename.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = filename.parent / f"cache_{filename.stem}"
    if not filename.exists():
        print(f"No dataset found. Calling loader to {filename}")
        await load_dataset(cfg, filename)

    rollout = setup_debate(
        cfg,
        cache_dir,
        api_handler,
    )

    # Heuristic to work out num threads. Each case fans out into BoN candidate
    # generations, so divide the thread budget by BoN to get concurrent cases.
    #
    # This must stay an int >= 1. True division returned a float, and
    # `asyncio.Semaphore(1.25)` does not limit concurrency to one — it lets
    # every waiting task straight through, so a stage given a thread budget of
    # 5 with BoN=4 issued unbounded concurrent requests instead of the single
    # case the heuristic intended.
    _BoN_correct = cfg.correct_debater.BoN
    _BoN_incorrect = cfg.incorrect_debater.BoN
    num_debate_threads = max(
        1,
        int(cfg.anthropic_num_threads) // max(1, _BoN_correct, _BoN_incorrect),
    )

    LOGGER.info(f"Running debates with {num_debate_threads} threads")

    # Run with retry
    complete = await async_function_with_retry(
        run_debates,
        rollout,
        filename,
        swap=False,
        limit=cfg.limit,
        num_debate_threads=num_debate_threads,
    )
    if not complete:
        return False

    # for sequential debate you must run swap at rollout time since it is not possible to do before judging
    if cfg.method_type == "seq" or (cfg.method == "debate" and cfg.use_intermediary):
        filename = experiment.get_debate_filename(seed=cfg.seed, swap=True)
        cache_dir = filename.parent / f"cache_{filename.stem}"
        if not filename.exists():
            await load_dataset(cfg, filename)

        rollout.cache_dir = cache_dir
        complete_swap = await async_function_with_retry(
            run_debates,
            rollout,
            filename,
            swap=True,
            limit=cfg.limit,
            num_debate_threads=num_debate_threads,
        )
    else:
        complete_swap = True
    return complete and complete_swap


if __name__ == "__main__":
    main()
