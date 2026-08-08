# Methodology — deep dive

Companion document to the abstract-level [`README.md`](../README.md). This is
where the case-selection rules, judge definitions, sampling settings, quality
controls, and output schemas live so the README itself can stay short.

## Dataset preparation

The committed pilot is `data/ddxplus/ddxplus_debate_pilot_100.csv` and can be
rebuilt from local DDXPlus release files with:

```bash
python scripts/prepare_ddxplus_debate_pilot.py
```

The 100 cases are not a simple random sample. The preparation script:

1. Reads the DDXPlus test-patient file and evidence definitions.
2. Keeps adult cases only.
3. Requires a non-true diagnosis in the differential list to use as a plausible
   distractor.
4. Keeps evidence-rich cases with at least 20 evidence items.
5. Requires both the true pathology and the distractor to have probability at
   least `0.05`.
6. Requires the true-vs-distractor probability gap to be no larger than `0.10`.
7. Sorts by smallest probability gap, then higher evidence count, then original
   source row order.
8. Selects cases round-robin across pathologies so the pilot is not dominated
   by one condition.
9. Randomizes whether the true diagnosis appears as diagnosis A or diagnosis B
   with fixed seed `20260504`.

This deliberately emphasizes difficult two-choice cases where the distractor
is plausible according to DDXPlus. The raw DDXPlus patient release files are
not redistributed through this repository; generated experiment outputs are
also excluded from Git.

## Baseline runs in detail

Baselines are conceptually fixed reference constants once
`(judge_model, judge_prompt, dataset)` are pinned. The canonical numbers live
under `data/canonical_baselines/<family>/` and are re-used across debate runs
rather than recomputed each time. The frontend reads from this folder by
default; flip "Recompute baselines" in the sidebar to force a fresh per-run
pass over both judges. See
[`data/canonical_baselines/README.md`](../data/canonical_baselines/README.md)
for the regeneration command and when to use it.

A baseline run creates the following, **once per judge** (canonical folders
carry both the weaker and the frontier judge so every debate arm has a
same-judge floor and ceiling for PGR):

- `baseline_blind/data0.csv` and `baseline_oracle/data0.csv` — zero-round
  working rows for each arm.
- `baseline_blind/<judge>/data0_judgement.csv` and `..._swap_judgement.csv`
  — blind judgements in original and swapped A/B order.
- `baseline_oracle/<judge>/data0_judgement.csv` and `..._swap_judgement.csv`
  — same for oracle.
- `results.csv` (sibling of the arm folders) — scoring summaries appended by
  `core.scoring.accuracy`, one row per `(arm, swap, judge_model)`.
- `plots/<judge>/01_blind_vs_oracle.png` and `02_position_bias.png` — one
  set of diagnostic plots per judge, written by `scripts/plot_baselines.py`.

The `data0.csv` files are not final result tables; they are cached working
files. The judgement CSVs and aggregate outputs are what downstream analysis
reads.

### Same-judge PGR pairing

`scripts/aggregate_medical_results.py` pairs each debate arm with the
baselines for the *same* judge model: E2 and E3 (weaker judge) are anchored
to the weaker-judge blind/oracle baselines, and E1 and E4 (frontier judge)
are anchored to the frontier-judge blind/oracle baselines. This matches the
Khan et al. convention and keeps the PGR denominator interpretable.

## Debate generation in detail

Each debate has three rounds. Both debaters see the structured patient
evidence; one is assigned the correct diagnosis and the other is assigned the
plausible distractor. The debaters are prompted to argue for their assigned
answer and to respond to the transcript so far, so each turn includes
continuity from earlier turns.

By default, each debater turn uses `BoN=4`:

1. The debater generates four candidate arguments for its assigned side.
2. A preference judge temporarily inserts each candidate into the current
   transcript.
3. The preference judge says which answer, A or B, looks more likely correct.
4. The candidate receives rating `1.0` if the preference judge chooses that
   debater's side, `0.0` if it chooses the opposing side, and `0.5` if the
   output is unclear.
5. The highest-rated candidate is written into the transcript.

If multiple candidates tie for the top rating, the current implementation
selects the first tied candidate in candidate order. There is no second
tie-break judging pass.

The preference judge is only an internal argument-selection mechanism. The
final answer judges run after the full transcript has been generated.

## Judges

The experiment uses several judge roles:

- **Baseline answer judge:** chooses A or B in the blind and oracle baselines.
- **Preference judge:** rates candidate debater arguments during BoN
  selection.
- **Final answer judge:** chooses A or B after seeing the completed debate
  transcript under E1-E4.
- **Concession judge:** checks whether either debater broke the setup by
  conceding or arguing for the opposing side.

The concession judge is a quality-control judge, not a diagnosis judge. It
does not decide who won the debate. In the current scripts and frontend it
uses `gpt-4o-mini` regardless of whether the main model family is OpenAI or
Anthropic.

## Quote verification

Debaters are instructed to cite patient evidence with `<quote>...</quote>`
tags. Before a judge sees the transcript, deterministic code in
`core/transcript_parser.py` checks whether each quote is an exact normalized
substring of the patient evidence.

Matched quotes are rewritten as `<v_quote>...</v_quote>`; unmatched quotes
are rewritten as `<u_quote>...</u_quote>`. Judges are told to trust verified
quotes and distrust unverified quotes. This is not another model call; it is
deterministic quote checking. The analysis script then reports verified and
unverified quote rates.

## Bias controls and quality checks

The pipeline emits the following diagnostics per family run:

- **Answer-position control.** Final judging is run twice per case, in
  original and swapped A/B order.
- **Swap-averaged accuracy.** Aggregation collapses the two passes into a
  per-case accuracy of `0.0`, `0.5`, or `1.0`.
- **Position-bias breakdown.** `analysis/position_bias.csv` decomposes each
  arm into four `(orig_pick, swap_pick)` buckets — `(A,B)` correct,
  `(B,A)` wrong, `(A,A)` A-position bias, `(B,B)` B-position bias — and
  reports the resulting bias percentage and swap-averaged accuracy.
- **Quote verification.** Fabricated or unsupported quotes are marked
  `<u_quote>` before judging and summarized in
  `analysis/quote_verification.csv`.
- **Verbosity analysis.** `analysis/verbosity.csv` checks whether one side
  is systematically wordier.
- **Concession analysis.** `analysis/concession.csv` flags debates where a
  debater appears to concede or argue for the other answer.
- **Debate outcome counts.** `analysis/debate_outcomes.csv` reports
  usable / conceded / incomplete / unparsed debates.
- **Cost summaries.** `cost_summary.csv` extracts API spend from the Hydra
  logs, one row per log file. The adapters log a *cumulative* running total
  per process, so each file's spend is the maximum of its snapshots and the
  run total is the sum of those maxima; summing the raw log lines
  over-counts several-fold.

Each CSV is accompanied by a `*_summary.txt` with a short human-readable
explanation, and most have a matching PNG under `analysis/plots/`.

## Sampling and token settings

The YAML configs use the same intended sampling settings across model
families, but the API adapters normalize them where providers differ.

- **Debater argument generation:** `temperature=0.8`, `top_p=1.0`,
  `max_tokens=1000`, `min_words=70`, `max_words=150`, `BoN=4`. This is the
  creative part of the pipeline: each debater samples four candidate
  arguments, then a preference judge selects the best candidate for the
  visible transcript.
- **Preference judge for BoN selection:** `temperature=0.0`, `top_p=1.0`,
  `max_tokens=2` in config. GPT-5-family OpenAI calls are routed through text
  A/B preference judgements and the adapter enforces the provider's minimum
  completion-token floor. Anthropic calls also use text A/B preference
  judgements because Anthropic does not expose OpenAI-style logprobs here.
- **Blind/oracle baselines and E1-E4 final judges:** `temperature=0.0`,
  `top_p=1.0`, and `max_tokens=null`. OpenAI receives no explicit output cap
  when the cap is `null`; Anthropic receives the shared wrapper default of
  `2000` output tokens.
- **Concession judge:** `gpt-4o-mini`, `temperature=0.0`, `top_p=1.0`,
  `max_tokens=1`. This is only a Y/N quality-control pass, not a diagnosis
  judge.

Provider-specific differences:

- Anthropic Claude 4.x rejects requests that specify both `temperature` and
  `top_p`, so the Anthropic adapter keeps `temperature` and drops `top_p`.
- Starting with Opus 4.7, Anthropic rejects any non-default `temperature`,
  `top_p`, or `top_k`. The Anthropic adapter has a guard that strips them
  for Opus 4.7+ — the medical experiment defaults to Opus 4.6, which still
  accepts the same `temperature=0.8` used for OpenAI debaters.
- GPT-5-family OpenAI calls in this repo drop `temperature` and `top_p` in
  the adapter, because those models are handled through a stricter request
  shape in the current legacy OpenAI SDK path.

## Expected outputs

For a run such as `./scripts/run_medical_full.sh 100 openai`, the main
outputs are:

```text
exp/YYYY-MM-DD_HH-MM-SS_results/
  run_metadata.json
  baselines/openai/
    baseline_blind/
    baseline_oracle/
    results.csv
  openai/
    run_metadata.json
    debate_sim/
      data0.csv
      e1_info_asymmetry_<judge>/
      e2_double_asymmetry_<judge>/
      e3_capability_asymmetry_<judge>/
      e4_full_symmetry_<judge>/
      concession_<judge>/
    analysis/
      verbosity.csv
      quote_verification.csv
      concession.csv
      debate_outcomes.csv
      position_bias.csv
      plots/
    cost_summary.csv
    one_debate_outputs.md
  medical_results/
    accuracy_by_condition.csv
    per_case_lift.csv
    baselines_by_arm.csv
    headline_results.csv
    paired_tests.csv
    quality_control_summary.csv
    plots/
```

Generated outputs live under a fresh timestamped folder in `exp/`, which is
intentionally gitignored while experiments are in progress. For final
release, commit only the curated final results folder you want readers to
inspect. The headline OpenAI run is curated at
[`../results/openai_100/`](../results/openai_100/); a future Anthropic
run will be curated under `../results/anthropic_100/` in the same shape
(`headline_results.csv`, `paired_tests.csv`, `baselines_by_arm.csv`,
`analysis/`, and the plot set) so the two bundles share a column schema.

Every file in a curated bundle is script-generated, so a fork can reproduce
the published tables from cached judgement CSVs without paying for a rerun.

## Paired statistics

All significance testing is paired at the patient-case level: blind, debate,
and oracle are measured on the same cases, so the per-case scores are compared
case by case rather than as three independent percentages. The shared
implementation lives in `scripts/medical_stats.py`, used by both
`aggregate_medical_results.py` and `compute_paired_tests.py` so no two outputs
can disagree about the same quantity.

- **Accuracy and mean-difference CIs.** 20,000-replicate percentile
  bootstrap. Case positions are drawn once per replicate and every arm is read
  at those positions, preserving the correlation between arms.
- **PGR CIs.** The same case-paired bootstrap, recomputing PGR from the three
  resampled accuracies. PGR's numerator and denominator share the blind term,
  so resampling the arms independently would misstate the interval. Replicates
  whose blind→oracle gap collapses to zero give an undefined ratio and are
  dropped from the percentiles.
- **Paired t-test.** Reported for continuity, but the per-case differences only
  take values in {-1, -0.5, 0, 0.5, 1}, so normality does not hold and the
  t-test is an approximation even at n=100.
- **Sign-flip permutation test.** The distribution-free check that carries the
  claim: 200,000 replicates flipping the sign of each nonzero per-case
  difference, with the `(hits + 1) / (replicates + 1)` estimator. The p-value
  floor is therefore about 5e-06 rather than 0.
- **Holm correction.** Applied over all eight tests per family
  (`holm_p_all_8_tests`) and separately within each set of four
  (`holm_p_within_4_tests`). The all-eight column is the conservative one to
  cite.

Resample counts and the seed are fixed constants, so repeated runs reproduce
the published numbers exactly.
