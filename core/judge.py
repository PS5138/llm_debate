import asyncio
import logging
import os
import random
import tempfile
from enum import StrEnum
from pathlib import Path
from shutil import copyfile

import hydra
import pandas as pd
from omegaconf import DictConfig

from core.agents.judge_base import JudgeBase
from core.create_agents import setup_judge
from core.file_handler import Experiment
from core.llm_api.llm import ModelAPI
from core.scoring.accuracy import find_answer
from core.utils import (
    async_function_with_retry,
    delete_old_prompt_files,
    setup_environment,
)

LOGGER = logging.getLogger(__name__)
APP_RUNNER_ENV = "MEDICAL_DEBATE_APP_RUNNER"


def _require_app_runner() -> None:
    """Refuse direct paid-stage execution outside the Streamlit runner."""
    if os.environ.get(APP_RUNNER_ENV) != "1":
        raise RuntimeError(
            "Direct judge execution is unsupported. Start the Streamlit app "
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


def _parseable_ab_judgement(value: object) -> bool:
    """Whether the production answer parser finds exactly one final A/B.

    Final answer judges must not checkpoint a truncated or ambiguous response
    as complete: resume relies on ``complete_judge=False`` to know that the API
    call needs retrying. Concession judgements use Y/N and are deliberately
    exempted by ``run_judge`` rather than being passed to this helper.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    found_a = bool(find_answer(value, "A"))
    found_b = bool(find_answer(value, "B"))
    return found_a != found_b


class JudgeType(StrEnum):
    default = "judge"
    concession = "concession"


def _reconcile_judgement_with_data(data_csv: Path, judgement_csv: Path) -> None:
    """Bring a pre-existing judgement CSV back into agreement with the
    debate-stage `data*.csv` it was originally seeded from.

    Reconciliation is strictly case-ID keyed. The data CSV owns row order,
    ``complete``, and ``transcript``; the judgement CSV owns judge answers and
    judge completion flags. Existing judge outputs follow their case ID even
    if the judgement CSV was reordered, while data IDs missing from the
    judgement CSV are appended in data order with empty judge state.

    `answer_judge`, `complete_judge`, and any other judge-stage output
    columns are preserved on rows that already had them — this is the
    important property that makes resume/extend safe.
    """
    if not data_csv.exists() or not judgement_csv.exists():
        return
    # IDs are opaque strings. Reading the whole CSV as text prevents pandas
    # from silently collapsing distinct-looking identifiers such as ``001``
    # and ``1`` before we validate or join them.
    data_df = pd.read_csv(data_csv, keep_default_na=False, dtype=str)
    j_df = pd.read_csv(judgement_csv, keep_default_na=False, dtype=str)

    def _validated_ids(df: pd.DataFrame, label: str) -> tuple[str, pd.Series]:
        id_column = (
            "id"
            if "id" in df.columns
            else "case_id"
            if "case_id" in df.columns
            else None
        )
        if id_column is None:
            raise ValueError(f"{label} CSV is missing an id/case_id column")
        raw = df[id_column].astype(str)
        ids = raw.str.strip()
        if ids.eq("").any():
            raise ValueError(f"{label} CSV contains a blank case ID")
        if raw.ne(ids).any():
            raise ValueError(
                f"{label} CSV contains a case ID with surrounding whitespace"
            )
        duplicates = sorted(set(ids[ids.duplicated(keep=False)]))
        if duplicates:
            preview = ", ".join(duplicates[:3])
            raise ValueError(f"{label} CSV contains duplicate case ID(s): {preview}")
        return id_column, ids

    data_id_column, data_ids = _validated_ids(data_df, "data")
    judgement_id_column, judgement_ids = _validated_ids(j_df, "judgement")
    data_id_set = set(data_ids)
    foreign_ids = [case_id for case_id in judgement_ids if case_id not in data_id_set]
    if foreign_ids:
        preview = ", ".join(foreign_ids[:3])
        raise ValueError(
            "judgement CSV contains case ID(s) absent from its data CSV: " + preview
        )

    # Keep the judgement schema, adding data-owned columns if a legacy file is
    # missing them. Judge-owned columns are never copied from the data CSV on
    # an existing row.
    output_columns = list(j_df.columns)
    for column in ("complete", "transcript"):
        if column in data_df.columns and column not in output_columns:
            output_columns.append(column)

    data_by_id = {
        case_id: data_df.iloc[position] for position, case_id in enumerate(data_ids)
    }
    judgement_by_id = {
        case_id: j_df.iloc[position] for position, case_id in enumerate(judgement_ids)
    }
    rows: list[dict] = []
    for case_id in data_ids:
        data_row = data_by_id[case_id]
        existing = judgement_by_id.get(case_id)
        if existing is None:
            row = {}
            for column in output_columns:
                if column == judgement_id_column:
                    row[column] = case_id
                elif column.startswith("complete"):
                    row[column] = False
                elif column.startswith("answer_"):
                    row[column] = ""
                elif column in data_df.columns:
                    row[column] = data_row[column]
                else:
                    row[column] = ""
        else:
            row = {column: existing.get(column, "") for column in output_columns}
            row[judgement_id_column] = case_id

        # These fields belong to the debate/data stage and must follow the ID,
        # not the judgement file's previous row position.
        for column in ("complete", "transcript"):
            if column in data_df.columns:
                row[column] = data_row[column]
        rows.append(row)

    reconciled = pd.DataFrame(rows, columns=output_columns)
    # A differently named data ID column is not judge-owned; if it is present
    # in the judgement schema, keep it aligned too.
    if data_id_column in reconciled.columns:
        reconciled[data_id_column] = data_ids.values
    _atomic_write_csv(reconciled, judgement_csv)


async def run_judge(
    judge: JudgeBase,
    filename: Path,
    anthropic_num_threads: int = 5,
    round_limit: int = None,
    limit: int = None,
    swap: bool = False,
    n_vote: int = 0,
    judge_type: JudgeType = JudgeType.default,
):
    # Load the whole working dataset. ``limit`` selects the target cohort but
    # must never truncate rows already present in a resumable CSV.
    full_df = pd.read_csv(filename, keep_default_na=False)
    target_df = full_df.head(int(limit)) if limit is not None else full_df
    target_indices = target_df.index
    assert _completed_mask(
        target_df["complete"]
    ).all(), "Not all values in the column 'complete' are True, rerun debate"

    # majority voting setup
    n_vote_str = n_vote if n_vote > 0 else ""
    complete_column = f"complete_{judge_type}{n_vote_str}"
    judge_column = f"answer_{judge_type}{n_vote_str}"
    if complete_column not in full_df.columns:
        full_df[complete_column] = False
    if judge_column not in full_df.columns:
        full_df[judge_column] = ""

    # Repair legacy checkpoints that marked an API call successful even though
    # its text never produced a usable final answer. Without this migration,
    # resume would skip exactly the rows it now needs to retry.
    if judge_type == JudgeType.default:
        target_completion = _completed_mask(
            full_df.loc[target_indices, complete_column]
        )
        target_parseable = full_df.loc[target_indices, judge_column].map(
            _parseable_ab_judgement
        )
        invalid_complete_indices = target_completion[
            target_completion & ~target_parseable
        ].index
        if len(invalid_complete_indices) > 0:
            LOGGER.warning(
                "Resetting %d cached judge row(s) with no unambiguous final "
                "A/B so resume can retry them",
                len(invalid_complete_indices),
            )
            full_df.loc[invalid_complete_indices, complete_column] = False
            _atomic_write_csv(full_df, filename)

    # filter out rows that have already been completed
    target_df = full_df.loc[target_indices]
    df = target_df[~_completed_mask(target_df[complete_column])]

    LOGGER.info(f"Using column {complete_column}")
    LOGGER.info(f"Using file {filename}")
    LOGGER.info(f"Processing {len(df)} rows")
    if round_limit is not None:
        LOGGER.info(f"Using round limit {round_limit}")

    async def bounded_run(sem, index, row, swap, round_limit):
        async with sem:
            try:
                result = await judge.make_decision(index, row, swap, round_limit)
                result_complete = bool(result["complete"])
                if (
                    result_complete
                    and judge_type == JudgeType.default
                    and not _parseable_ab_judgement(result["judgement"])
                ):
                    LOGGER.warning(
                        "Judge case %s returned no unambiguous final A/B; "
                        "leaving it incomplete so resume can retry it",
                        index,
                    )
                    result_complete = False
                # Persist immediately after the paid response, before this
                # coroutine can yield again. Coroutines run on one event-loop
                # thread, so each atomic replacement is a single-writer step.
                full_df.at[index, judge_column] = result["judgement"]
                full_df.at[index, complete_column] = result_complete
                if "transcript" in result:
                    full_df.at[index, "transcript"] = result["transcript"]
                _atomic_write_csv(full_df, filename)
                return result_complete
            except asyncio.CancelledError:
                raise
            except Exception:
                # A failed case must not discard completed, already-persisted
                # siblings from the same asynchronous batch.
                LOGGER.exception("Judge case %s failed", index)
                return False

    sem = asyncio.Semaphore(int(anthropic_num_threads))
    tasks = [
        bounded_run(sem, index, row, swap, round_limit) for index, row in df.iterrows()
    ]
    results = await asyncio.gather(*tasks)

    correct = sum(results)
    LOGGER.info(f"Processed {len(results)} rows. {correct} were complete.")
    # See core/debate.py: the second element is how many rows this attempt
    # newly completed, which the retry loop uses as its no-progress guard.
    target_complete = _completed_mask(
        full_df.loc[target_indices, complete_column]
    ).all()
    return bool(target_complete), int(correct)


@hydra.main(version_base=None, config_path="config/", config_name="config")
def main(cfg: DictConfig):
    _require_app_runner()
    complete = asyncio.run(async_main(cfg))
    if not complete:
        raise RuntimeError(
            "Judge stage did not complete; see the preceding errors for details."
        )
    return True


async def async_main(cfg: DictConfig):
    setup_environment(
        logger_level=cfg.logging,
        anthropic_tag=cfg.anthropic_tag,
        openai_tag=cfg.openai_tag,
    )
    random.seed(cfg.seed)

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
    if cfg.judge_type == JudgeType.concession:
        cfg.judge = cfg.concession_judge

    judge = setup_judge(
        cfg,
        cfg.judge,
        api_handler,
    )
    if cfg.judge_name is None:
        cfg.judge_name = judge.config.language_model.model
        LOGGER.info(f"Judge name not specified, using {cfg.judge_name}")

    # using exp_suffix to easily run differing number of rounds with judge
    exp_suffix = f"_{cfg.round_limit}rounds" if cfg.round_limit is not None else ""

    # Run swap = False
    filename = experiment.get_debate_filename(seed=cfg.seed, swap=False)
    filename_judgement = experiment.get_judge_filename(
        cfg.judge_name, seed=cfg.seed, swap=False, exp_suffix=exp_suffix
    )

    assert filename.exists(), f"{filename} does not exist"
    filename_judgement.parent.mkdir(exist_ok=True, parents=True)
    if not filename_judgement.exists():
        copyfile(filename, filename_judgement)
    else:
        _reconcile_judgement_with_data(filename, filename_judgement)

    complete = await async_function_with_retry(
        run_judge,
        judge,
        filename_judgement,
        anthropic_num_threads=cfg.anthropic_num_threads,
        round_limit=cfg.round_limit,
        limit=cfg.limit,
        swap=False,
        n_vote=cfg.judge.n_vote,
        judge_type=cfg.judge_type,
    )
    if not complete:
        return False

    # Run swap = True
    filename_swap = experiment.get_debate_filename(seed=cfg.seed, swap=True)
    filename_swap_judgement = experiment.get_judge_filename(
        cfg.judge_name, seed=cfg.seed, swap=True, exp_suffix=exp_suffix
    )
    if not filename_swap_judgement.exists():
        if cfg.method_type == "seq" or (
            cfg.method == "debate" and cfg.use_intermediary
        ):
            assert (
                filename_swap.exists()
            ), f"{filename_swap} does not exist and is required for sequential (or intermediary debate)"
            copyfile(filename_swap, filename_swap_judgement)
        else:
            # If debate is seq, we insist that a swapped debate file exists. In other cases, we should still prefer using a swapped debate file if it's available.
            if filename_swap.exists():
                copyfile(filename_swap, filename_swap_judgement)
            else:
                copyfile(filename, filename_swap_judgement)
    else:
        # Sim mode reuses data0.csv for the swap judgement, so reconcile
        # against whichever source the judge was originally seeded from.
        swap_source = filename_swap if filename_swap.exists() else filename
        _reconcile_judgement_with_data(swap_source, filename_swap_judgement)

    complete_swap = await async_function_with_retry(
        run_judge,
        judge,
        filename_swap_judgement,
        anthropic_num_threads=cfg.anthropic_num_threads,
        round_limit=cfg.round_limit,
        limit=cfg.limit,
        swap=True,
        n_vote=cfg.judge.n_vote,
        judge_type=cfg.judge_type,
    )
    return complete and complete_swap


if __name__ == "__main__":
    main()
