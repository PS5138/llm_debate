# Medical Debate Dataset

This fork uses a 100-case pilot derived from **DDXPlus**, a public synthetic
differential-diagnosis dataset. It no longer ships the original QuALITY human
feedback or LLM debate datasets from the upstream repository.

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

The DDXPlus files come from the public dataset released with:

```bibtex
@inproceedings{tchango2022ddxplus,
  title={DDXPlus: A New Dataset For Automatic Medical Diagnosis},
  author={Tchango, Arsene Fansi and Goel, Rishab and Wen, Zhi and Martel, Julien and Ghosn, Joumana},
  booktitle={Advances in Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```

The prepared pilot can be rebuilt with:

```bash
python scripts/prepare_ddxplus_debate_pilot.py
```

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
9. Randomizes whether the true diagnosis appears as diagnosis A or diagnosis B
   using a fixed seed (`20260504`), so the output is reproducible.

This means the pilot intentionally emphasizes difficult two-choice cases where
the distractor is plausible according to DDXPlus, rather than easy random cases.

## Prepared CSV Schema

`ddxplus_debate_pilot_100.csv` contains one row per case:

- `case_id`: Stable pilot identifier such as `ddxplus_pilot_001`.
- `source_row_id`: Row index in the raw DDXPlus test-patient file.
- `question_stem`: Short patient-facing question used in prompts.
- `diagnosis_a`, `diagnosis_b`: The two answer choices in randomized order.
- `evidence`: Decoded structured patient evidence.
- `correct_answer`: `diagnosis_a` or `diagnosis_b`.
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
by the inherited debate machinery. For example, a blind baseline run writes:

```text
exp/medical_debate_n100/baselines/baseline_blind/data0.csv
```

That file is an experiment working file, not a final result table. It contains
the question, internal correct/negative answer columns, patient evidence in the
`story` column, an initially empty `transcript` column, and a `complete` flag.
After `core.debate` runs, `data0.csv` has completed baseline/debate transcripts.
After `core.judge` runs, judgement CSVs are written under the judge-model
subdirectory, for example:

```text
exp/medical_debate_n100/baselines/baseline_blind/gpt-5.4-mini/data0_judgement.csv
```

The aggregate scored baseline results are appended to:

```text
exp/medical_debate_n100/baselines/results.csv
```
