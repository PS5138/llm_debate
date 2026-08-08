import asyncio
import json
import logging
import os
import time
from functools import lru_cache, wraps
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import typer
import yaml

from core.rollouts.utils import TranscriptConfig

typer.main.get_command_name = lambda name: name

LOGGER = logging.getLogger(__name__)
SEPARATOR = "---------------------------------------------\n\n"
SEPARATOR_CONVERSATIONAL_TURNS = "=============================================\n\n"
PROMPT_HISTORY = "prompt_history"

LOGGING_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def setup_environment(
    anthropic_tag: str = "ANTHROPIC_API_KEY",
    logger_level: str = "info",
    openai_tag: str = "API_KEY",
    organization: str = None,
):
    setup_logging(logger_level)
    secrets_path = os.environ.get("MEDICAL_DEBATE_SECRETS_PATH", "SECRETS")
    secrets = load_secrets(secrets_path)
    openai.api_key = secrets[openai_tag]
    os.environ["ANTHROPIC_API_KEY"] = secrets[anthropic_tag]
    if organization is not None:
        openai.organization = secrets[organization]


def prompt_history_dir() -> str:
    return os.environ.get("MEDICAL_DEBATE_PROMPT_HISTORY_DIR", PROMPT_HISTORY)


def setup_logging(level_str):
    level = LOGGING_LEVELS.get(
        level_str.lower(), logging.INFO
    )  # default to INFO if level_str is not found
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Disable logging from openai
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)


def load_secrets(file_path):
    secrets = {}
    with open(file_path) as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            secrets[key] = value
    return secrets


def load_yaml(file_path):
    with open(file_path) as f:
        content = yaml.safe_load(f)
    return content


@lru_cache(maxsize=8)
def load_yaml_cached(file_path):
    with open(file_path) as f:
        content = yaml.safe_load(f)
    return content


def save_yaml(file_path, data):
    with open(file_path, "w") as f:
        yaml.dump(data, f)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data


def save_jsonl(file_path, data):
    with open(file_path, "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")


def delete_old_prompt_files(
    path: str | None = None, max_age_minutes: int = 60, keep_recent: int = 50
):
    """
    Delete all files in the folder that:
    - Are more than max_age_minutes old
    - AND are not one of the keep_recent most recent files
    """
    if path is None:
        path = prompt_history_dir()
    if not os.path.exists(path):
        return

    # Get all files in the folder with their full paths and creation times
    files = [
        {
            "path": os.path.join(path, filename),
            "ctime": os.path.getctime(os.path.join(path, filename)),
        }
        for filename in os.listdir(path)
        if os.path.isfile(os.path.join(path, filename))
    ]

    # Sort files by creation time
    files.sort(key=lambda f: f["ctime"], reverse=True)

    # Current time in seconds since epoch
    now = time.time()

    deleted_count = 0
    for index, file_info in enumerate(files):
        # File age in minutes
        age_minutes = (now - file_info["ctime"]) / 60

        # If file is older than x_minutes and is not one of the y_most_recent files, delete it
        if age_minutes > max_age_minutes and index >= keep_recent:
            os.remove(file_info["path"])
            deleted_count += 1

    if deleted_count > 0:
        print(f"Deleted {deleted_count} old prompt files")


def create_filtered_csv(
    csv_path: Path, output_path: Path, filter_func: Callable[[TranscriptConfig], bool]
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    def filter_transcript(row: pd.Series):
        transcript = TranscriptConfig(**json.loads(row["transcript"]))
        return filter_func(transcript)

    filtered_df = df[df.apply(filter_transcript, axis=1)]
    filtered_df.to_csv(output_path)


def typer_async(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # No event loop running
            loop = None

        if loop is None:
            return asyncio.run(f(*args, **kwargs))
        else:
            return f(*args, **kwargs)  # Return coroutine to be awaited

    return wrapper


# Stage-level retry budget. A stage reports (complete, newly_completed); the
# retry loop stops as soon as an attempt completes nothing new, because a
# stage that made zero progress will not make any on the next identical
# attempt either. That distinction matters: a transient rate limit leaves some
# rows finished and is worth retrying, while a bad key or an unknown model
# fails every row, and blindly retrying re-pays for the partial work each
# time. Override the ceiling with MEDICAL_MAX_ATTEMPTS.
MAX_STAGE_ATTEMPTS = int(os.environ.get("MEDICAL_MAX_ATTEMPTS", "16"))


def _unpack_stage_result(result) -> tuple[bool, int | None]:
    """Normalise a stage's return value to (complete, newly_completed).

    Stages that report progress return a `(bool, int)` tuple. Older stages
    return a bare bool, for which progress is unknown and the no-progress
    guard cannot apply.
    """
    if isinstance(result, tuple) and len(result) == 2:
        complete, progress = result
        return bool(complete) is True and complete is not None, int(progress)
    return result is True, None


def _log_attempt(attempt: int, max_attempts: int, progress: int | None) -> bool:
    """Log one failed attempt. Returns True if retrying is worthwhile."""
    if progress is None:
        LOGGER.info(
            f"Attempt {attempt}/{max_attempts} did not complete the stage; retrying."
        )
        return True
    if progress > 0:
        LOGGER.info(
            f"Attempt {attempt}/{max_attempts} completed {progress} new row(s) "
            f"but the stage is not finished; retrying."
        )
        return True
    LOGGER.error(
        f"Attempt {attempt}/{max_attempts} completed 0 new rows. Retrying would "
        f"re-issue the same failing calls and be billed again, so the stage is "
        f"stopping here. Check the errors above — a bad API key, an unknown "
        f"model name, or an exhausted quota all look like this."
    )
    return False


def function_with_retry(function, *args, max_attempts: int = None, **kwargs):
    max_attempts = max_attempts or MAX_STAGE_ATTEMPTS
    for attempt in range(1, max_attempts + 1):
        complete, progress = _unpack_stage_result(function(*args, **kwargs))
        if complete:
            return True
        if attempt == max_attempts or not _log_attempt(attempt, max_attempts, progress):
            return False
    return False


async def async_function_with_retry(
    function, *args, max_attempts: int = None, **kwargs
):
    max_attempts = max_attempts or MAX_STAGE_ATTEMPTS
    for attempt in range(1, max_attempts + 1):
        complete, progress = _unpack_stage_result(await function(*args, **kwargs))
        if complete:
            return True
        if attempt == max_attempts or not _log_attempt(attempt, max_attempts, progress):
            return False
    return False


def log_model_timings(api_handler, save_location="./model_timings.png"):
    if len(api_handler.model_timings) > 0:
        plt.figure(figsize=(10, 6))
        for model in api_handler.model_timings:
            timings = np.array(api_handler.model_timings[model])
            wait_times = np.array(api_handler.model_wait_times[model])
            LOGGER.info(
                f"{model}: response {timings.mean():.3f}, waiting {wait_times.mean():.3f} (max {wait_times.max():.3f}, min {wait_times.min():.3f})"
            )
            plt.plot(
                timings, label=f"{model} - Response Time", linestyle="-", linewidth=2
            )
            plt.plot(
                wait_times, label=f"{model} - Waiting Time", linestyle="--", linewidth=2
            )
        plt.legend()
        plt.title("Model Performance: Response and Waiting Times")
        plt.xlabel("Sample Number")
        plt.ylabel("Time (seconds)")
        plt.savefig(save_location, dpi=300)
        plt.close()


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
