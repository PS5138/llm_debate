"""Regression checks for the end-to-end medical shell wrapper."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run_wrapper(
    frontier: str | None, weaker: str | None, family: str = "openai"
) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "scripts").mkdir()
        (work / "data/ddxplus").mkdir(parents=True)
        (work / "data/ddxplus/ddxplus_debate_pilot_100.csv").write_text(
            "id\n1\n", encoding="utf-8"
        )
        (work / "SECRETS").write_text(
            "API_KEY=test\nANTHROPIC_API_KEY=test\nDEFAULT_ORG=\n", encoding="utf-8"
        )
        shutil.copy2(ROOT / "scripts/run_medical_full.sh", work / "scripts")

        record = work / "calls.txt"
        _write_executable(
            work / "scripts/run_medical_baselines.sh",
            '#!/usr/bin/env bash\nprintf \'baseline|%s|%s|%s|%s\\n\' "$1" "$2" "$3" "$4" >> "$CALL_RECORD"\n',
        )
        _write_executable(
            work / "scripts/run_medical_debate.sh",
            '#!/usr/bin/env bash\nprintf \'debate|%s|%s|%s|%s|%s\\n\' "$1" "$2" "$3" "$FRONTIER" "$WEAKER" >> "$CALL_RECORD"\n',
        )

        env = os.environ.copy()
        env.pop("FRONTIER", None)
        env.pop("WEAKER", None)
        env.update(
            {
                "BASELINES_DIR": "external/baselines",
                "CALL_RECORD": str(record),
                "PREPARE_DATA": "0",
                "PYTHON": "/usr/bin/true",
                "RUN_TESTS": "0",
            }
        )
        if frontier is not None:
            env["FRONTIER"] = frontier
        if weaker is not None:
            env["WEAKER"] = weaker
        subprocess.run(
            ["bash", "scripts/run_medical_full.sh", "7", family, "run-root"],
            cwd=work,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        return record.read_text(encoding="utf-8").splitlines()


def main() -> int:
    failures = []

    calls = _run_wrapper("frontier-custom", "weaker-custom")
    expected = [
        "baseline|7|external/baselines|openai|weaker-custom,frontier-custom",
        "debate|7|run-root|openai|frontier-custom|weaker-custom",
    ]
    if calls != expected:
        failures.append(("custom models reach both stages", calls, expected))

    same_calls = _run_wrapper("same-model", "same-model")
    same_expected = [
        "baseline|7|external/baselines|openai|same-model",
        "debate|7|run-root|openai|same-model|same-model",
    ]
    if same_calls != same_expected:
        failures.append(
            ("duplicate baseline model is de-duplicated", same_calls, same_expected)
        )

    openai_default_calls = _run_wrapper(None, None)
    openai_default_expected = [
        "baseline|7|external/baselines|openai|gpt-5.4-mini,gpt-5.5",
        "debate|7|run-root|openai|gpt-5.5|gpt-5.4-mini",
    ]
    if openai_default_calls != openai_default_expected:
        failures.append(
            (
                "OpenAI defaults match both stages",
                openai_default_calls,
                openai_default_expected,
            )
        )

    anthropic_default_calls = _run_wrapper(None, None, family="anthropic")
    anthropic_default_expected = [
        "baseline|7|external/baselines|anthropic|claude-sonnet-4-6,claude-opus-4-6",
        "debate|7|run-root|anthropic|claude-opus-4-6|claude-sonnet-4-6",
    ]
    if anthropic_default_calls != anthropic_default_expected:
        failures.append(
            (
                "Anthropic defaults match both stages",
                anthropic_default_calls,
                anthropic_default_expected,
            )
        )

    for description, got, expected_value in failures:
        print(f"FAIL  [{description}] got={got!r} expected={expected_value!r}")
    if failures:
        return 1

    print("4 shell workflow cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
