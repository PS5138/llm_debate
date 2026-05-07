# Medical Debate

> **Can two strong AIs arguing opposing diagnoses help a less-informed AI judge pick the right one?**
>
> A six-week BlueDot Impact Technical AI Safety Sprint, extending [Khan et al. (2024)](https://arxiv.org/abs/2402.06782) into clinical differential diagnosis.
>
> Forked from [`ucl-dark/llm_debate`](https://github.com/ucl-dark/llm_debate). Original Khan et al. README at the bottom of this file.

## Why this matters: the AI oversight problem

As AI systems get more capable, they will increasingly produce outputs that the people supervising them cannot easily check. A frontier AI can read fifty pages of clinical notes in seconds and produce a confident diagnosis; the doctor reviewing it might have ten minutes. If the AI is wrong, how does the doctor find out? This is the **AI oversight problem** in miniature, and it's one of the central practical concerns of AI safety today.

One promising idea is **AI debate**. The intuition: lying about something checkable is harder than telling the truth — especially when an opposing AI is actively looking for the lie. If two AIs argue opposing answers, and one is wrong, the other should be able to expose specific weaknesses that a less-informed reviewer can recognise even without re-checking the source material. The reviewer's job becomes *evaluating arguments*, not generating answers from scratch — a cognitively easier task.

## What Khan et al. (2024) showed

Khan and colleagues at University College London tested this on hard reading-comprehension questions about short stories. They set up:

- **Two debaters** — copies of the same AI model, instructed to argue opposite answers. Both have read the story.
- **A judge** — another AI, which has *not* read the story. It only sees the two-debater transcript and picks a winner.
- **Three rounds**, with a strict word limit per round, and an automatic tool that verifies any direct quotes the debaters use.

The headline result was striking. Compared to a baseline judge guessing without any debate (~48% accuracy on a hard two-option task — barely above chance) or watching a single AI argue one side (~54%), a judge watching a full **debate reached ~76% accuracy** — recovering most of the way to the 92.5% ceiling achieved by an AI that read the story directly. Critically, the debaters and the judge were the *same model*. The format itself was doing the work.

Khan et al. validated that result in one regime — where the debaters had read the story and the judge had not — and explicitly invited follow-up work in two harder regimes they couldn't fully test:

1. **Harder, more realistic domains** than story comprehension.
2. **Cases where the judge is a noticeably weaker model than the debaters** — the realistic oversight scenario, where a cheaper or smaller AI (or a busy human) is checking work produced by a more capable one.

This sprint accepts that invitation, in clinical medicine.

## What this fork tests

Two AIs are given the same patient's structured clinical evidence and asked to argue for competing diagnoses. A judge AI — which has *not* seen the evidence — reads the debate and picks a diagnosis. We measure how often the judge gets it right, across four conditions:

| Condition | Debaters | Judge | Judge sees evidence? | What this tests |
|---|---|---|---|---|
| Information asymmetry | Frontier model | Same frontier model | No | Replication of Khan et al.'s positive result, in medicine. |
| Double asymmetry | Frontier model | **Weaker model** | No | **The novel claim.** Does debate still help when the judge is also weaker than the debaters? This is the realistic clinical-oversight scenario. |
| Capability asymmetry | Frontier model | Weaker model | Yes | Re-test of Khan et al.'s negative finding — when the weak judge has direct access to the evidence, does debate add anything? |
| Full symmetry | Frontier model | Same frontier model | Yes | Re-test of Khan et al.'s other negative finding — same model on both sides, no information gap. |

We run this on two model families in parallel so the result doesn't hinge on any single provider:

- **Anthropic family** — frontier debaters: Claude Opus 4.7. Weaker judge: Claude Sonnet.
- **OpenAI family** — frontier debaters: GPT-5.5. Weaker judge: a smaller GPT model.

Each case is also run through two **baselines** — judge sees the question only ("blind"), and judge sees the question plus the patient's evidence ("oracle") — for each judge model. These give the floor and ceiling against which debate accuracy is measured.

## The dataset, and an honest caveat

The cases come from **DDXPlus**, a publicly released synthetic differential-diagnosis dataset. Each case has a presenting question stem (e.g. *"A 39-year-old female patient presents with hoarse voice. What is the most likely diagnosis?"*), two plausible diagnoses (one true, one a real probabilistic alternative from DDXPlus's differential), and a structured list of symptom and antecedent evidence (~20–35 items per case).

DDXPlus is **not** real long-form chart review. Records are synthetic, structured, and a few hundred words after decoding — there are no real labs, imaging, examination findings, or narrative clinical notes. So the framing of this sprint is:

> *Does AI debate improve blind judging in synthetic differential diagnosis from structured symptom and antecedent evidence?*

A clean positive result here is a precondition for, not a substitute for, a future study on long real clinical records. We commit upfront to writing up the result honestly either way — including a clean negative result, which would itself be informative for the AI safety literature.

## Methodology, in brief

Three rounds of debate, 150 words per round, both debaters reading the same evidence and arguing opposing diagnoses. Debaters can quote directly from the evidence using `<quote>` tags, and quotes are automatically verified as exact matches before the judge sees the transcript. The "incorrect" debater always argues for a real probabilistic alternative from DDXPlus's own differential — never a wrong-on-its-face diagnosis — which keeps the debate honest and avoids the safety-tuning of frontier models that otherwise causes them to apologise mid-argument and concede.

We use **best-of-N candidate sampling at N=4** at each debater turn (the setting Khan et al. used for their headline result): the model generates four candidate arguments per turn, and a separate preference model — the same frontier model as the debater — picks the most persuasive one to enter the visible transcript.

We control for two well-documented LLM biases. **Answer-letter bias** — large language models prefer picking "A" over "B" — is controlled by running every judging step twice with the answer letters swapped, and averaging. **Verbosity bias** — judges may favour longer arguments — is controlled by enforcing a strict per-round word limit and reporting any correlation between argument length and judge wins.

The primary metric is **judge accuracy** (with bootstrap 95% confidence intervals), reported per condition for each model family. We additionally report **Performance Gap Recovered** — the fraction of the gap between the blind baseline and the oracle baseline that debate closes — which is Khan et al.'s preferred summary statistic.

## What this fork adds on top of upstream

- `core/load/medical.py` — DDXPlus loader. Converts the prepared pilot CSV into the repo's internal question schema.
- `core/config/experiment/medical_blind.yaml`, `medical_oracle.yaml` — baseline experiment configs.
- `core/config/experiment/judge/baselines/medical_*.yaml` — judge prompts for the medical baselines.
- `scripts/prepare_ddxplus_debate_pilot.py` — builds the pilot CSV from the raw DDXPlus release.
- `scripts/run_medical_baselines.sh` — single entrypoint for the blind + oracle baseline pipeline.
- `data/ddxplus/ddxplus_debate_pilot_100.csv` — the prepared 100-case pilot. Raw DDXPlus release files are gitignored; rebuild locally with the script above.
- `EDA.ipynb` — exploratory analysis of the pilot dataset and the pipeline outputs.
- Minimal modifications to `core/debate.py`, `core/judge.py`, `core/main.py`, `core/llm_api/openai_llm.py`, and `core/config/config.yaml` — medical-pipeline plumbing (dataset selection, model registry entries).

## Status

Sprint in progress. The pilot dataset, baseline pipeline, and configs are wired up. Debate runs and full analysis are next.

## Quickstart

```bash
# 1. Set up the environment (Python 3.11; the project pins this version)
virtualenv --python python3.11 .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Add API keys to a top-level SECRETS file (not .env). At minimum:
#    API_KEY=<openai-key>
#    ANTHROPIC_API_KEY=<anthropic-key>
#    DEFAULT_ORG=

# 3. The 100-case pilot CSV is already committed. To rebuild from the raw DDXPlus release:
python scripts/prepare_ddxplus_debate_pilot.py

# 4. Run the baseline pipeline. First arg = number of cases; second arg (optional) = exp dir.
./scripts/run_medical_baselines.sh 20             # 20-case smoke  -> exp/medical_pilot_n20
./scripts/run_medical_baselines.sh 100            # full pilot     -> exp/medical_pilot_n100
```

Outputs land under `exp/<run-name>/baseline_blind/data0.csv` and `exp/<run-name>/baseline_oracle/data0.csv`. The "Pipeline-Backed Baseline" section of `EDA.ipynb` reads those CSVs and produces a comparison plot.

For the full debate runs, follow the same pattern as the baseline script with `+experiment=debate` and the appropriate judge override; the four debate conditions are produced by re-judging the cached transcripts with different judge models and prompt variants.

## Acknowledgements

This project is being carried out as part of the [BlueDot Impact Technical AI Safety Sprint](https://bluedot.org/), a six-week applied research programme in which participants take a recently published AI-safety paper and produce a small original extension of it. Thanks to the BlueDot Impact team for the programme structure, mentorship, and funding support that make this work possible.

## Credits and licence

Base codebase: [`ucl-dark/llm_debate`](https://github.com/ucl-dark/llm_debate), MIT licensed. Original authors: Akbir Khan, John Hughes, Dan Valentine, Laura Ruis, Kshitij Sachan, Ansh Radhakrishnan, Edward Grefenstette, Samuel R. Bowman, Tim Rocktäschel, Ethan Perez. Their paper is included as `paper.pdf`.

Cite the original work:

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

The DDXPlus dataset is described in:

```bibtex
@inproceedings{tchango2022ddxplus,
  title={DDXPlus: A New Dataset For Automatic Medical Diagnosis},
  author={Tchango, Arsene Fansi and Goel, Rishab and Wen, Zhi and Martel, Julien and Ghosn, Joumana},
  booktitle={Advances in Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```

This fork is licensed under the same MIT licence as upstream (see `LICENSE`).

---

## Original upstream README (Khan et al.)

Setup and reproduction instructions for the QuALITY experiments are unchanged from upstream. See the [original repository](https://github.com/ucl-dark/llm_debate) for the QuALITY-specific quickstart, the human-trial frontend, and tournament instructions.
