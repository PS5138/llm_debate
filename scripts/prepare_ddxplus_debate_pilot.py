import ast
import csv
import json
import random
from pathlib import Path


DATA_DIR = Path("data/ddxplus")
PATIENTS_PATH = DATA_DIR / "release_test_patients"
EVIDENCES_PATH = DATA_DIR / "release_evidences.json"
OUTPUT_PATH = DATA_DIR / "ddxplus_debate_pilot_100.csv"

VALUE_CLEANUPS = {
    "palace": "palate",
    "heartbreaking": "tearing",
    "haunting": "throbbing",
    "tedious": "aching",
    "a knife stroke": "stabbing",
    "N": "no",
    "Y": "yes",
}

INITIAL_COMPLAINT_OVERRIDES = {
    "E_45": "chest pain",
    "E_50": "increased sweating",
    "E_51": "diarrhea or increased stool frequency",
    "E_53": "pain related to the reason for consultation",
    "E_66": "significant shortness of breath",
    "E_75": "choking or suffocating sensation",
    "E_77": "productive cough",
    "E_82": "lightheadedness or near-fainting",
    "E_88": "severe fatigue limiting usual activities",
    "E_89": "fatigue or non-restorative sleep",
    "E_76": "dizziness or lightheadedness",
    "E_91": "fever",
    "E_97": "muscle aches",
    "E_103": "loss of smell",
    "E_129": "skin lesions or rash",
    "E_140": "black stools",
    "E_144": "coughing up blood",
    "E_148": "nausea or vomiting",
    "E_150": "inability to pass stools or gas",
    "E_155": "palpitations",
    "E_161": "loss of appetite or early satiety",
    "E_174": "unintentional weight loss or loss of appetite",
    "E_179": "pale skin",
    "E_181": "runny or congested nose",
    "E_190": "increased salivation",
    "E_201": "cough",
    "E_203": "pain after trauma",
    "E_211": "repeated vomiting or retching",
    "E_212": "hoarse or softer voice",
    "E_215": "symptoms worse after eating",
    "E_217": "symptoms worse lying down and relieved by sitting up",
    "E_218": "exertional symptoms relieved by rest",
    "E_220": "rib pain",
}


def sex_word(sex: str) -> str:
    if sex == "F":
        return "female"
    if sex == "M":
        return "male"
    return sex


def decode_evidence_item(item: str, evidence_defs: dict) -> tuple[str, str, str]:
    if "_@_" in item:
        evidence_id, value_id = item.split("_@_", 1)
    else:
        evidence_id, value_id = item, None

    evidence = evidence_defs[evidence_id]
    question = evidence["question_en"].strip()

    if value_id is None:
        value = "yes"
    else:
        value_meaning = evidence.get("value_meaning") or {}
        value = value_meaning.get(value_id, {}).get("en", str(value_id))
        value = VALUE_CLEANUPS.get(value, value)

    category = "antecedent/risk factor" if evidence["is_antecedent"] else "symptom/current evidence"
    return evidence_id, category, f"{question} Answer: {value}."


def decode_initial_evidence(initial: str, evidence_defs: dict) -> str:
    _, _, decoded = decode_evidence_item(initial, evidence_defs)
    return decoded


def initial_complaint(initial: str, decoded_initial: str) -> str:
    evidence_id = initial.split("_@_", 1)[0]
    if evidence_id in INITIAL_COMPLAINT_OVERRIDES:
        return INITIAL_COMPLAINT_OVERRIDES[evidence_id]

    complaint = decoded_initial
    complaint = complaint.replace(" Answer: yes.", "")
    complaint = complaint.replace(" Answer: no.", "")
    complaint = complaint.strip()
    if complaint.endswith("?"):
        complaint = complaint[:-1]
    complaint = complaint[:1].lower() + complaint[1:]
    return complaint


def decode_record(age: str, sex: str, items: list[str], evidence_defs: dict) -> str:
    lines = [f"Patient: {age}-year-old {sex_word(sex)}."]
    for item in items:
        evidence_id, category, decoded = decode_evidence_item(item, evidence_defs)
        lines.append(f"{category}: {decoded} [source: {item}; base: {evidence_id}]")
    return "\n".join(lines)


def load_rows() -> list[dict]:
    evidence_defs = json.loads(EVIDENCES_PATH.read_text())
    rows = []

    with PATIENTS_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for source_row_id, row in enumerate(reader):
            differentials = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
            evidences = ast.literal_eval(row["EVIDENCES"])
            pathology = row["PATHOLOGY"]
            if int(row["AGE"]) < 18:
                continue

            pathology_prob = next(prob for diagnosis, prob in differentials if diagnosis == pathology)
            distractor = next(
                ((diagnosis, prob) for diagnosis, prob in differentials if diagnosis != pathology),
                None,
            )
            if distractor is None:
                continue

            distractor_name, distractor_prob = distractor
            if len(evidences) < 20:
                continue
            if pathology_prob < 0.05 or distractor_prob < 0.05:
                continue
            if abs(distractor_prob - pathology_prob) > 0.10:
                continue

            rows.append(
                {
                    "source_row_id": source_row_id,
                    "age": row["AGE"],
                    "sex": row["SEX"],
                    "pathology": pathology,
                    "distractor": distractor_name,
                    "pathology_probability": pathology_prob,
                    "distractor_probability": distractor_prob,
                    "evidence_count": len(evidences),
                    "initial_evidence": row["INITIAL_EVIDENCE"],
                    "decoded_initial_evidence": decode_initial_evidence(
                        row["INITIAL_EVIDENCE"], evidence_defs
                    ),
                    "evidence": decode_record(row["AGE"], row["SEX"], evidences, evidence_defs),
                }
            )

    rows.sort(
        key=lambda row: (
            abs(row["distractor_probability"] - row["pathology_probability"]),
            -row["evidence_count"],
            row["source_row_id"],
        )
    )
    return rows


def choose_pilot_rows(rows: list[dict], n: int = 100) -> list[dict]:
    by_pathology = {}
    for row in rows:
        by_pathology.setdefault(row["pathology"], []).append(row)

    selected = []
    while len(selected) < n:
        added = False
        for pathology in sorted(by_pathology):
            bucket = by_pathology[pathology]
            if bucket:
                selected.append(bucket.pop(0))
                added = True
                if len(selected) == n:
                    break
        if not added:
            break
    return selected


def main() -> None:
    rng = random.Random(20260504)
    selected = choose_pilot_rows(load_rows(), n=100)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as f:
        fieldnames = [
            "case_id",
            "source_row_id",
            "question_stem",
            "diagnosis_a",
            "diagnosis_b",
            "evidence",
            "correct_answer",
            "pathology",
            "top_differential",
            "pathology_probability",
            "top_differential_probability",
            "evidence_count",
            "initial_evidence",
            "decoded_initial_evidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case_number, row in enumerate(selected, start=1):
            correct_is_a = rng.choice([True, False])
            diagnosis_a = row["pathology"] if correct_is_a else row["distractor"]
            diagnosis_b = row["distractor"] if correct_is_a else row["pathology"]
            correct_answer = "diagnosis_a" if correct_is_a else "diagnosis_b"
            question_stem = (
                f"A {row['age']}-year-old {sex_word(row['sex'])} patient presents with "
                f"{initial_complaint(row['initial_evidence'], row['decoded_initial_evidence'])}. "
                f"What is the most likely diagnosis?"
            )

            writer.writerow(
                {
                    "case_id": f"ddxplus_pilot_{case_number:03d}",
                    "source_row_id": row["source_row_id"],
                    "question_stem": question_stem,
                    "diagnosis_a": diagnosis_a,
                    "diagnosis_b": diagnosis_b,
                    "evidence": row["evidence"],
                    "correct_answer": correct_answer,
                    "pathology": row["pathology"],
                    "top_differential": row["distractor"],
                    "pathology_probability": row["pathology_probability"],
                    "top_differential_probability": row["distractor_probability"],
                    "evidence_count": row["evidence_count"],
                    "initial_evidence": row["initial_evidence"],
                    "decoded_initial_evidence": row["decoded_initial_evidence"],
                }
            )

    print(f"Wrote {len(selected)} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
