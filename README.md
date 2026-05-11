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

**Pilot dataset and loader**
- `core/load/medical.py` — DDXPlus loader. Converts the prepared pilot CSV into the repo's internal question schema.
- `data/ddxplus/ddxplus_debate_pilot_100.csv` — the 100-case pilot. Raw DDXPlus release files are gitignored; rebuild locally with `scripts/prepare_ddxplus_debate_pilot.py`.

**Experiment configs — medical baselines and four debate conditions**
- `core/config/experiment/medical_blind.yaml`, `medical_oracle.yaml` — baseline experiment configs (B1, B2).
- `core/config/experiment/medical_debate.yaml` — debate experiment config; defaults to E1.
- `core/config/experiment/medical_debate_e{2,3,4}_*.yaml` — per-condition variants for E2 (double asymmetry), E3 (capability asymmetry), E4 (full symmetry).
- `core/config/experiment/debaters/medical_v1.yaml` — debater config with medical-shaped prompts (replaces the upstream story-comprehension wording).
- `core/config/experiment/judge/baselines/medical_*.yaml` — baseline judge prompts.
- `core/config/experiment/judge/debate/medical_e{1,2,3,4}_*.yaml` — per-condition main-judge prompts.
- `core/config/experiment/judge/debate/preference_medical.yaml`, `concession_medical.yaml` — preference and concession judges, medical-shaped.

**Runner and analysis scripts**
- `scripts/prepare_ddxplus_debate_pilot.py` — builds the pilot CSV from the raw DDXPlus release.
- `scripts/run_medical_baselines.sh` — runs the blind + oracle baselines.
- `scripts/run_medical_debate.sh` — generates one debate transcript per case, re-judges across all four conditions (E1–E4), runs the concession judge, and triggers the bias-control analyses, aggregator, and cost summary.
- `scripts/analyze_medical_debate.py` — per-family bias-control diagnostics. Computes verbosity (per-side word counts), quote verification (verified vs unverified `<quote>` tag counts), and concession rate, and emits matching PNG plots. Pure file-system work; no API calls.
- `scripts/aggregate_medical_results.py` — cross-run aggregator. Reads cached judgement CSVs and writes `accuracy_by_condition.csv`, `pgr_by_condition.csv`, and `per_case_lift.csv` with bootstrap 95% CIs, plus the three headline plots (accuracy by condition, PGR by condition, per-case 2×2 lift heatmaps).
- `scripts/summarise_run_costs.py` — greps the Hydra logs for per-call cost lines and emits `cost_summary.csv` for spend tracking against the budget.
- `scripts/export_medical_debate_output.py` — produces a single readable markdown file with the full debate and all four judgements per case (useful for review).
- `EDA.ipynb` — exploratory analysis of the pilot dataset and pipeline outputs.

**Tests**
- `tests/test_accuracy.py` — 19 regression cases for `find_answer`, pinning the bug shapes that previously inverted scoring whenever a judge wrapped its answer in `"Final answer: Answer: <X>"`. Run with `python tests/test_accuracy.py` or `make test`.

**Internal modules**
- `core/transcript_parser.py` — `<quote>` verification helpers (`normalize_text`, `add_missing_quote_tags`, `verify`, `verify_strict`), extracted from the upstream `web/backend/services/parser.py` so the medical pipeline can run without the human-trial web stack.

**Modifications to upstream code**
- `core/scoring/accuracy.py` — `find_answer` rewrite that takes the last `Answer: <X>` occurrence as the judge's final pick, with a negative lookahead that prevents matching the literal `A` inside the next word "Answer". Pinned by `tests/test_accuracy.py`.
- `core/llm_api/openai_llm.py` — added the GPT-5.x family to the model registry and pricing table; added GPT-5-specific param handling (`max_completion_tokens` rename, dropped `temperature`/`top_p`); added a fast-fail path for OpenAI quota / billing exhaustion so the retry loop bails out immediately with a clear error rather than burning thousands of identical retries.
- `core/llm_api/anthropic_llm.py` — full rewrite from the upstream legacy Completion API to the Messages API. Registers the Claude 4 family (Opus 4.7, Sonnet 4.6, Haiku 4.5) with real per-token pricing; extracts `system` messages to the top-level param Anthropic expects; supports BoN via parallel calls; mirrors the OpenAI quota fast-fail; records real cost from `response.usage` (output tokens include Opus's hidden reasoning, so the figures match what Anthropic bills).
- `core/llm_api/llm.py` — chunks BoN candidate sampling at 8 per call to stay within OpenAI rate limits and avoid silent dropped completions.
- `core/agents/judge_quality.py` — falls back from logprob preference judging to plain completion for GPT-5-family models (their logprob API path doesn't apply).
- `core/debate.py`, `core/judge.py`, `core/main.py`, `core/config/config.yaml` — medical-pipeline plumbing (dataset selection, model registry entries).

**Removed from upstream**

To keep the fork focused on the medical experiments, the following were stripped from the codebase. The QuALITY codepath proper (loader, debater/judge classes, rollouts) is preserved so that infrastructure still resolves cleanly, but nothing in this fork drives it:

- `web/` — the original Khan et al. human-trial web frontend (FastAPI + React); QuALITY-only and not used by the medical experiments. The shared `TranscriptParser` it hosted now lives in `core/transcript_parser.py`.
- `scripts/human_trial_example/` — example code for the human-trial database.
- `scripts/reproduce_minimal.sh`, `scripts/run_figure{1,3+5,4}.sh`, `scripts/run_tournament.py`, `scripts/tournament_players/`, `scripts/plot_minimal.ipynb` — Khan QuALITY reproduction scripts and tournament-player configs.
- `core/tournament.py`, `core/swiss_tournament.py`, `core/scoring/ratings.py`, `core/scoring/trueskill.py` — cross-play tournament and Elo / TrueSkill rating code.
- `core/scoring/quotes.py` — Khan-era quote-visualisation driver.
- `core/load/human_questions.csv` — Khan human-trial question list.
- `core/config/experiment/blind.yaml`, `oracle.yaml`, `consultancy.yaml`, `consultancy_critique_story.yaml`, `debate.yaml`, `debate_critique_story.yaml`, `debate_interactive.yaml`, `debate_seq.yaml` — Khan-era top-level experiment YAMLs, superseded by the medical equivalents.
- `core/config/experiment/{consultants,critic,judge/consultancy}/` — Khan-era consultancy and critic configs.
- `core/config/experiment/{debaters/v1_interactive.yaml, judge/baselines/{blind,oracle}.yaml, judge/debate/{default,concession,preference_old}.yaml, rollout/{live,nyu,seq}.yaml}` — Khan sub-configs only referenced by the deleted top-level YAMLs.

## Status

End-to-end medical pipeline is wired up and ready to run on both model families: dataset loader, all baseline and debate configs, runner scripts, regression tests, the bias-control analyses (verbosity, quote-verification, concession), the cross-run aggregator (accuracy with bootstrap 95% CIs, PGR, per-case lift), and a cost-summary helper. The Anthropic adapter has been rewritten from the upstream legacy Completion API to the modern Messages API; both families now route cleanly. An initial 1-case OpenAI smoke confirmed the pipeline produces clean clinical debates (100% verified-quote rate on both sides, balanced word counts, no concessions). Full 100-case runs are queued and pending API credit.

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

# 4. Sanity-check the scoring code. No API calls.
python tests/test_accuracy.py        # or: make test

# 5. Run the baselines. First arg = number of cases; second arg (optional) = exp dir.
./scripts/run_medical_baselines.sh 20             # 20-case smoke
./scripts/run_medical_baselines.sh 100            # full pilot

# 6. Run the debate pipeline. Generates one debate per case, then judges across
#    four conditions (E1-E4), runs the concession judge, and writes bias-control
#    analyses. The four debate conditions all re-use the same cached transcript
#    per case — only the main judge changes — so the marginal cost of additional
#    conditions is small.
./scripts/run_medical_debate.sh 5 exp/medical_debate_n5 openai      # 5-case smoke
./scripts/run_medical_debate.sh 100 exp/medical_debate_n100 openai  # full pilot

# 7. Re-run the bias-control analyses and cross-run aggregator on cached
#    transcripts (no API spend). The aggregator also produces the headline
#    accuracy / PGR / per-case-lift plots.
python scripts/analyze_medical_debate.py exp/medical_debate_n5/openai
python scripts/aggregate_medical_results.py exp/medical_debate_n5/openai \
    --out-dir exp/medical_debate_n5/medical_results
python scripts/summarise_run_costs.py exp/medical_debate_n5/openai
```

Outputs land under `exp/<run-name>/`:

| Path | What it is |
|---|---|
| `baseline_blind/data0.csv`, `baseline_oracle/data0.csv` | Baseline judge results (B1, B2). |
| `<family>/debate_sim/data0.csv` | Debate transcripts, generated once per family. |
| `<family>/debate_sim/<condition>_<judge>/data0[_swap]_judgement.csv` | Per-condition judge decisions, in answer-letter swap pairs. |
| `<family>/debate_sim/concession_<model>/data0[_swap]_judgement.csv` | Concession-judge output. |
| `<family>/results.csv` | Accumulated accuracy per condition. |
| `<family>/cost_summary.csv` | Per-stage API spend (debate / judge / scoring / concession), grepped from Hydra logs. |
| `<family>/analysis/{verbosity,quote_verification,concession}.csv` | Bias-control summaries plus matching `_summary.txt` files. |
| `<family>/analysis/plots/04_verbosity.png`, `05_concession_rate.png`, `06_quote_verification.png` | Bias-control PNGs. |
| `<family>/one_debate_outputs.md` | Single-file markdown export of the debates and judgements (useful for review). |
| `medical_results/{accuracy_by_condition,pgr_by_condition,per_case_lift}.csv` | Cross-run summary CSVs with bootstrap 95% CIs. |
| `medical_results/plots/{01_accuracy_by_condition,02_pgr_by_condition,03_per_case_lift}.png` | Headline plots for the write-up. |

## Acknowledgements

This project is being carried out as part of the [BlueDot Impact Technical AI Safety Sprint](https://bluedot.org/), a six-week applied research programme in which participants take a recently published AI-safety paper and produce a small original extension of it. With thanks to the BlueDot Impact team for the programme structure, mentorship, and the grant that funds the API spend behind these experiments — none of this work would happen without that support.

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

## Note on the upstream codebase

This fork has been stripped of QuALITY-only tooling that wasn't used by the medical experiments — the human-trial web frontend, the figure-reproduction scripts, the cross-play tournament code, the Elo / TrueSkill ratings, and a substantial number of Khan-era experiment YAMLs. The QuALITY loader, debater / judge classes, and rollouts are kept in place so the underlying infrastructure still resolves cleanly, but nothing in this fork drives them end-to-end. If you need the human-trial UI, the figure-reproduction scripts, or the tournament code, see the [original repository](https://github.com/ucl-dark/llm_debate).
