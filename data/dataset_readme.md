# Medical Debate Dataset

This fork uses a 100-case pilot derived from **DDXPlus**, a public synthetic
differential-diagnosis dataset. It no longer ships the original QuALITY human
feedback or LLM debate datasets from the upstream repository.

> **Research use only.** DDXPlus contains synthetic patients and its authors
> state that it is intended for research. This pilot is not clinical data,
> medical advice, or a diagnostic tool and must not be used for real-patient
> decisions.

## Committed Files

- `data/ddxplus/ddxplus_debate_pilot_100.csv`
  - The prepared 100-case pilot used by the medical debate pipeline.
- `data/ddxplus/release_evidences.json`
  - DDXPlus evidence definitions used to decode evidence IDs into readable
    symptom and antecedent text.
- `data/ddxplus/release_conditions.json`
  - DDXPlus condition metadata retained with the raw release files.

## Local-Only Files

These files may exist locally when rebuilding the pilot, but they are not
tracked in Git:

- `data/ddxplus/release_test_patients`
  - Raw DDXPlus test-patient CSV used to rebuild the pilot.
- `data/ddxplus/release_test_patients.zip`
  - Compressed copy of the raw DDXPlus test-patient file.

## Source

Download the official release from
[DDXPlus on Figshare](https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374).
The dataset is released under the
[Creative Commons Attribution 4.0 licence](https://creativecommons.org/licenses/by/4.0/),
so reuse requires appropriate attribution. The accompanying paper and dataset
documentation should be cited:

```bibtex
@inproceedings{tchango2022ddxplus,
  title={DDXPlus: A New Dataset For Automatic Medical Diagnosis},
  author={Tchango, Arsene Fansi and Goel, Rishab and Wen, Zhi and Martel, Julien and Ghosn, Joumana},
  booktitle={Advances in Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```

To rebuild the committed pilot locally:

1. Download and extract the DDXPlus release from Figshare.
2. Place the test-patient file at
   `data/ddxplus/release_test_patients`. The committed evidence and condition
   definition files must remain at
   `data/ddxplus/release_evidences.json` and
   `data/ddxplus/release_conditions.json`.
3. From the repository root, run the no-API preparation utility:

```bash
python scripts/prepare_ddxplus_debate_pilot.py
```

This command only prepares data; it does not launch provider calls. Normal app
users do not need the raw release because the prepared 100-case CSV is already
committed.

## How The 100 Cases Are Chosen

The pilot is deterministic, not a simple random sample. The preparation script:

1. Reads the DDXPlus test-patient file and evidence definitions.
2. Keeps adult cases only (`AGE >= 18`).
3. Keeps cases whose differential diagnosis list contains at least one
   non-true diagnosis to use as a distractor.
4. Keeps evidence-rich cases with at least 20 evidence items.
5. Keeps cases where both the true pathology and distractor have probability
   at least `0.05`.
6. Keeps cases where the true pathology probability and distractor probability
   are close: absolute difference no greater than `0.10`.
7. Sorts candidates by:
   - closest true-vs-distractor probability gap,
   - then more evidence items,
   - then source row order.
8. Selects cases in a round-robin over pathologies so the 100-case pilot is not
   dominated by one condition.
9. Writes auxiliary randomized `diagnosis_a`, `diagnosis_b`, and
   `correct_answer` columns using fixed seed `20260504`.

This means the pilot intentionally emphasizes difficult two-choice cases where
the distractor is plausible according to DDXPlus, rather than easy random cases.

## Prepared CSV Schema

`ddxplus_debate_pilot_100.csv` contains one row per case:

- `case_id`: Stable pilot identifier such as `ddxplus_pilot_001`.
- `source_row_id`: Row index in the raw DDXPlus test-patient file.
- `question_stem`: Short patient-facing question used in prompts.
- `diagnosis_a`, `diagnosis_b`: Auxiliary answer choices in seeded randomized
  order. The current experiment loader does not consume these columns.
- `evidence`: Decoded structured patient evidence.
- `correct_answer`: Auxiliary label for the randomized prepared-CSV columns;
  not consumed by the current experiment loader.
- `pathology`: The true DDXPlus pathology.
- `top_differential`: The selected plausible distractor.
- `pathology_probability`, `top_differential_probability`: DDXPlus
  differential probabilities for the true and distractor diagnoses.
- `evidence_count`: Number of raw evidence items.
- `initial_evidence`, `decoded_initial_evidence`: Presenting complaint source
  and decoded text.

## Experiment CSVs

The pipeline does not run directly on `ddxplus_debate_pilot_100.csv`. Instead,
`core/load/medical.py` converts it into the internal experiment schema expected
by the inherited debate machinery. It reconstructs internal A as the true
`pathology` and internal B as `top_differential`; position control comes from
the later original and A/B-swapped judgement passes, not the auxiliary
randomized columns above.

When baseline recomputation is selected in the Streamlit app, a blind arm can
write:

```text
exp/YYYY-MM-DD_HH-MM-SS_results/baselines/openai/baseline_blind/data0.csv
```

That file is an experiment working file, not a final result table. It contains
the question, internal correct/negative answer columns, patient evidence in the
`story` column, an initially empty `transcript` column, and a `complete` flag.
The app's internal debate stage fills `data0.csv`; its judgement stage writes
CSV files under the judge-model subdirectory, for example:

```text
exp/YYYY-MM-DD_HH-MM-SS_results/baselines/openai/baseline_blind/gpt-5.4-mini/data0_judgement.csv
```

The aggregate scored baseline results are appended to:

```text
exp/YYYY-MM-DD_HH-MM-SS_results/baselines/openai/results.csv
```
