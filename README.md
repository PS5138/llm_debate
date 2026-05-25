# Medical Debate

**Can AI debate help a weaker judge choose the right medical diagnosis?**

![Live debate viewer](assets/debate_pic.png)

## Introduction

This repository is a work-in-progress research fork of
[`ucl-dark/llm_debate`](https://github.com/ucl-dark/llm_debate), the codebase
released with Khan et al. (2024), *Debating with More Persuasive LLMs Leads to
More Truthful Answers*. The original paper studied debate on hard reading
comprehension questions. This fork adapts the same broad oversight idea to
synthetic medical differential diagnosis.

The core question is whether debate can make hidden evidence easier to judge.
Each case has two candidate diagnoses: the true DDXPlus pathology and a
plausible distractor from that case's differential diagnosis list. Two debaters
argue opposite sides while seeing the structured patient evidence. A judge then
chooses between the diagnoses, sometimes without seeing the evidence directly.

The motivation is future-facing. A clinician reviewing an AI-assisted diagnosis
may be surrounded by more information than they can realistically verify: a long
record, a differential diagnosis, prior history, medications, test results, and
model-generated reasoning arriving under time pressure. In that world, the
supervisor may not be weak in an absolute sense; they may simply be weaker than
the system on that task, or too overloaded to audit every detail from scratch.
This project uses a weaker model judge as a proxy for that future human
supervisor: capable enough to evaluate arguments, but not assumed to have the
time, context, or raw capability to reconstruct the answer alone.

The project is being developed as part of the BlueDot Impact Technical AI
Safety Sprint, with API spend supported by a BlueDot grant.

## Methodology

The experiment asks whether AI debate improves blind judging in synthetic
differential diagnosis from structured symptom and antecedent evidence. The
headline oversight setting is **E2: double asymmetry** — stronger debaters
see the patient evidence and argue both sides, while a weaker judge
evaluates the transcript without seeing the evidence. If E2 improves on a
blind baseline, debate is recovering some of the value of hidden evidence
without requiring the judge to inspect the full record.

The pilot uses **DDXPlus**, a public synthetic differential-diagnosis
dataset. The committed `data/ddxplus/ddxplus_debate_pilot_100.csv` is a
deliberately hard 100-case slice: adult cases, ≥20 evidence items each,
both the true pathology and the distractor with probability ≥ 0.05 and a
true-vs-distractor gap ≤ 0.10, picked round-robin across pathologies with
seed `20260504`. Full filtering rules and the regeneration command live in
[`docs/methodology.md`](docs/methodology.md).

### Experimental arms

| Arm | Description | Debaters | Judge | Judge sees evidence? |
|---|---|---|---|---|
| B1 | Blind baseline | None | Weaker judge | No |
| B2 | Oracle baseline | None | Weaker judge | Yes |
| E1 | Information asymmetry | Frontier model | Same frontier model | No |
| E2 | Double asymmetry (headline) | Frontier model | Weaker model | No |
| E3 | Capability asymmetry | Frontier model | Weaker model | Yes |
| E4 | Full symmetry | Frontier model | Same frontier model | Yes |

The baselines define the floor and ceiling for **Performance Gap Recovered**:

```text
PGR = (debate_accuracy - blind_accuracy) / (oracle_accuracy - blind_accuracy)
```

### Model families

| Family | Frontier debaters / frontier judge | Weaker judge and baselines |
|---|---|---|
| OpenAI | `gpt-5.5` | `gpt-5.4-mini` |
| Anthropic | `claude-opus-4-6` | `claude-sonnet-4-6` |

E1 and E4 use the frontier model as final judge. E2 and E3 use the weaker
model as final judge. The concession quality-control judge uses `gpt-4o-mini`
for both families.

### Pipeline

```mermaid
flowchart TD
    A["DDXPlus raw release files"] --> B["prepare_ddxplus_debate_pilot.py"]
    B --> C["ddxplus_debate_pilot_100.csv"]
    C --> D["core/load/medical.py"]
    D --> E1["baseline_blind/data0.csv"]
    D --> E2["baseline_oracle/data0.csv"]
    D --> F["debate_sim/data0.csv"]

    E1 --> G1["Blind judge calls<br/>original order and swapped order"]
    E2 --> G2["Oracle judge calls<br/>original order and swapped order"]
    G1 --> H["baselines/results.csv"]
    G2 --> H

    F --> I["Round 1"]
    I --> J["Round 2"]
    J --> K["Round 3"]
    K --> L["Cached debate transcript<br/>debate_sim/data0.csv"]

    I -. "each debater turn" .-> I1["Generate 4 candidate arguments"]
    I1 -. "each candidate" .-> I2["Preference judge rates candidate"]
    I2 -. "best candidate" .-> I3["Selected argument enters transcript"]

    L --> M["E1-E4 final answer judges<br/>original order and swapped order"]
    L --> N["Concession judge<br/>quality-control pass"]
    M --> O["debate_sim/*/data0_judgement.csv"]
    N --> P["analysis/concession.csv"]
    O --> Q["aggregate_medical_results.py"]
    H --> Q
    P --> R["analyze_medical_debate.py"]
    Q --> S["medical_results/*.csv and plots"]
    R --> T["analysis/*.csv and plots"]
```

Baselines are conceptually fixed reference constants once
`(judge_model, judge_prompt, dataset)` are pinned. Canonical 100-case
baselines for both families are committed at `data/canonical_baselines/`
and re-used across debate runs by default; flip "Recompute baselines" in
the Streamlit sidebar to force a fresh per-run pass.

### Debate generation

![Debate transcript viewer in the Streamlit app](assets/streamlit-demo-final.png)

Each debate has three rounds. Both debaters see the structured patient
evidence; one is assigned the correct diagnosis and the other is assigned
the plausible distractor. Each turn uses **BoN=4**: the debater generates
four candidate arguments, a preference judge rates each against the
in-progress transcript, and the highest-rated candidate is written into the
transcript. The final answer judges run after the full transcript has been
generated.

### Bias controls

- **Swap-averaged accuracy.** Every case is judged in original and swapped
  A/B order; the two passes collapse into a per-case score of 0.0, 0.5,
  or 1.0. Position-biased cases score 0.5 by construction.
- **Position-bias breakdown.** `analysis/position_bias.csv` decomposes each
  arm into four `(orig_pick, swap_pick)` buckets and reports the resulting
  bias percentage.
- **Deterministic quote verification.** Cited spans are checked as exact
  normalised substrings of the patient evidence before judges see them;
  unverified spans are flagged as `<u_quote>`.
- **Quality-control judge.** A separate `gpt-4o-mini` pass flags debates
  where either side conceded or argued for the opposing diagnosis.
- **Verbosity, concession, and outcome diagnostics** are emitted alongside
  every run under `analysis/`.

For the detailed treatment of judge roles, sampling settings, and
output-directory schemas, see
[`docs/methodology.md`](docs/methodology.md).

## Results

_Pending._ Headline numbers and plots will be filled in here once the
full OpenAI 100-case run completes and is curated. The Anthropic
100-case run (Opus 4.6 + Sonnet 4.6) is the planned cross-family
comparison; canonical Anthropic baselines are already committed at
`data/canonical_baselines/anthropic/`.

## How to run

### Requirements

- Python 3.11.
- A Unix-like shell.
- OpenAI API access (required even for Anthropic-only runs, because the
  concession judge uses `gpt-4o-mini`).
- Anthropic API access if running the Anthropic family.
- The committed DDXPlus pilot CSV, or local raw DDXPlus release files if
  you want to rebuild it.

The OpenAI adapter is pinned to the legacy `openai==0.28.0` SDK in
`requirements.txt`; do not upgrade to `openai>=1.x` until
`core/llm_api/openai_llm.py` has been migrated.

Secrets live in a top-level `SECRETS` file (gitignored):

```text
API_KEY=<openai-key>
ANTHROPIC_API_KEY=<anthropic-key>
DEFAULT_ORG=
```

Keep the `DEFAULT_ORG=` and `ANTHROPIC_API_KEY=` lines even if empty —
the shared secrets loader expects both keys to exist.

### Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
make hooks   # optional pre-commit hooks
make test    # local no-API parser regression test
```

### Two ways to run

**Interactive frontend** — live row-by-row debate viewer, BYO key,
session-scoped tempfiles, suited to exploration and chunked
review-as-you-go workflows.

```bash
streamlit run app/streamlit_app.py
```

**One-command CLI** — the full
`baselines → debate → E1-E4 → concession → analysis → aggregation`
pipeline:

```bash
./scripts/run_medical_full.sh 1   openai      # one-case smoke
./scripts/run_medical_full.sh 100 openai      # planned headline OpenAI run
./scripts/run_medical_full.sh 100 anthropic   # planned Anthropic run
```

Both write outputs into a fresh `exp/YYYY-MM-DD_HH-MM-SS_results/`
folder. The deep walkthrough — manual stage-by-stage runs, no-API
re-analysis, frontend buttons and resume/extend semantics, the full
script map — lives in [`docs/operating.md`](docs/operating.md).

## Relationship to upstream

Focused research fork of `ucl-dark/llm_debate`. Upstream provides the
debate framework, rollouts, judge/debater abstractions, and scoring
patterns. This fork removes the QuALITY-era human-trial and tournament
tooling and adds the medical dataset loader, DDXPlus pilot preparation,
medical prompts, model-family configuration, OpenAI/Anthropic adapter
updates, swap judging, quote verification, concession analysis, result
aggregation, the Streamlit frontend, and the canonical-baselines layer.
Upstream paper: [`arXiv:2402.06782`](https://arxiv.org/abs/2402.06782).

## Acknowledgements

This work is part of the
[BlueDot Impact Technical AI Safety Sprint](https://bluedot.org/), a six-week
programme in which participants extend recent AI safety work into a small
original project. I am grateful to BlueDot for the programme structure,
mentorship, and grant funding for API credits.

## Citation

Base codebase and original debate result:

```bibtex
@misc{khan2024debating,
  title={Debating with More Persuasive LLMs Leads to More Truthful Answers},
  author={Akbir Khan and John Hughes and Dan Valentine and Laura Ruis and Kshitij Sachan and Ansh Radhakrishnan and Edward Grefenstette and Samuel R. Bowman and Tim Rocktäschel and Ethan Perez},
  year={2024},
  eprint={2402.06782},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

DDXPlus dataset:

```bibtex
@inproceedings{tchango2022ddxplus,
  title={DDXPlus: A New Dataset For Automatic Medical Diagnosis},
  author={Tchango, Arsene Fansi and Goel, Rishab and Wen, Zhi and Martel, Julien and Ghosn, Joumana},
  booktitle={Advances in Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```

This fork is licensed under the same MIT licence as upstream. See
[`LICENSE`](LICENSE).
