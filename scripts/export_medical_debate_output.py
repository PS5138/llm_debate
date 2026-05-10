import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.rollouts.utils import TranscriptConfig


CONDITIONS = [
    "e1_info_asymmetry",
    "e2_double_asymmetry",
    "e3_capability_asymmetry",
    "e4_full_symmetry",
]


def read_first_row(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, keep_default_na=False)
    if df.empty:
        return None
    return df.iloc[0]


def transcript_to_markdown(row: pd.Series) -> str:
    transcript = TranscriptConfig(**json.loads(row["transcript"]))
    lines = [
        f"Question: {transcript.question}",
        "",
        f"A: {transcript.answers.correct}",
        f"B: {transcript.answers.incorrect}",
        "",
        "Patient Evidence:",
        transcript.story or "",
        "",
        "Transcript:",
    ]
    for i, round_ in enumerate(transcript.rounds, start=1):
        lines.extend([f"", f"Round {i}:"])
        if round_.correct:
            lines.extend(["", f"Debater A: {round_.correct}"])
        if round_.incorrect:
            lines.extend(["", f"Debater B: {round_.incorrect}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--family", default="openai")
    args = parser.parse_args()

    debate_file = args.exp_dir / "debate_sim" / "data0.csv"
    debate_row = read_first_row(debate_file)
    if debate_row is None:
        raise FileNotFoundError(f"No debate row found at {debate_file}")

    lines = [
        "# Medical Debate Smoke Output",
        "",
        f"Family: {args.family}",
        f"Experiment directory: {args.exp_dir}",
        f"Requested limit: {args.limit}",
        "",
        "## Debate",
        "",
        transcript_to_markdown(debate_row),
        "",
        "## Judgements",
    ]

    debate_root = args.exp_dir / "debate_sim"
    for condition in CONDITIONS:
        matches = sorted(debate_root.glob(f"{condition}_*/data0_judgement.csv"))
        swap_matches = sorted(
            debate_root.glob(f"{condition}_*/data0_swap_judgement.csv")
        )
        lines.extend(["", f"### {condition}", ""])
        if not matches:
            lines.append("Missing non-swap judgement.")
            continue
        row = read_first_row(matches[0])
        if row is not None:
            lines.extend(
                [
                    f"Source: {matches[0]}",
                    "",
                    "Non-swap judgement:",
                    "",
                    str(row.get("answer_judge", "")),
                    "",
                ]
            )
        if swap_matches:
            swap_row = read_first_row(swap_matches[0])
            if swap_row is not None:
                lines.extend(
                    [
                        f"Swap source: {swap_matches[0]}",
                        "",
                        "Swap judgement:",
                        "",
                        str(swap_row.get("answer_judge", "")),
                        "",
                    ]
                )

    output_path = args.exp_dir / "one_debate_outputs.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
