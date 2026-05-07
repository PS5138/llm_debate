import argparse
import csv
from pathlib import Path
from typing import Optional


REQUIRED_COLUMNS = {
    "case_id",
    "question_stem",
    "evidence",
    "pathology",
    "top_differential",
}


def validate_source(fieldnames: list[str], source_path: Path) -> None:
    missing = REQUIRED_COLUMNS - set(fieldnames)
    if missing:
        raise ValueError(
            f"{source_path} is missing required medical dataset columns: {sorted(missing)}"
        )


def write_questions(rows: list[dict], filepath: Path, limit: Optional[int] = None) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if limit is not None:
        rows = rows[: int(limit)]

    with filepath.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "id",
                "question",
                "correct answer",
                "negative answer",
                "complete",
                "transcript",
                "answer",
                "prompt",
                "cot prompt",
                "story",
                "story_title",
                "question_set_id",
                "story tokens",
            ]
        )

        for row in rows:
            evidence = row["evidence"]
            writer.writerow(
                [
                    row["case_id"],
                    row["question_stem"],
                    row["pathology"],
                    row["top_differential"],
                    False,
                    "",
                    "",
                    "",
                    "",
                    evidence,
                    row["case_id"],
                    row["case_id"],
                    len(str(evidence).split()),
                ]
            )


def main(
    filepath: Path,
    source_path: Path = Path("data/ddxplus/ddxplus_debate_pilot_100.csv"),
    limit: Optional[int] = None,
    write_to_file: bool = True,
    **_,
) -> list[dict]:
    source_path = Path(source_path)
    with source_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        validate_source(reader.fieldnames or [], source_path)
        rows = list(reader)

    if write_to_file:
        print(f"Writing {min(len(rows), int(limit)) if limit is not None else len(rows)} medical questions to {filepath}")
        write_questions(rows, Path(filepath), limit=limit)

    return rows[: int(limit)] if limit is not None else rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=Path)
    parser.add_argument("--source_path", type=Path, default=Path("data/ddxplus/ddxplus_debate_pilot_100.csv"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.filepath, source_path=args.source_path, limit=args.limit)
