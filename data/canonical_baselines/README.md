# Canonical Baselines

This directory holds **frozen blind + oracle baseline results** that every
debate run is compared against for PGR computation. By treating baselines as
fixed reference constants once `(judge_model, judge_prompt, dataset)` are
pinned, PGR is anchored to a stable floor/ceiling rather than re-rolling
slightly different baselines each experiment — which is how Khan et al. ran
QuALITY too.

## Layout

A fully-populated canonical folder carries **both the weaker and frontier
judge baselines** so every debate arm has a same-judge floor and ceiling for
PGR computation. E2/E3 use the weaker judge; E1/E4 use the frontier judge.

```
data/canonical_baselines/<family>/
  baseline_blind/
    data0.csv                                # working schema, not used downstream
    <weaker_judge>/data0_judgement.csv      # judged in original A/B order
    <weaker_judge>/data0_swap_judgement.csv # judged in swapped A/B order
    <frontier_judge>/data0_judgement.csv    # same pattern, frontier judge
    <frontier_judge>/data0_swap_judgement.csv
  baseline_oracle/
    ... same shape ...
  results.csv                                # per-(arm, swap, judge) accuracy summary
  plots/<weaker_judge>/01_blind_vs_oracle.png
  plots/<weaker_judge>/02_position_bias.png
  plots/<frontier_judge>/01_blind_vs_oracle.png
  plots/<frontier_judge>/02_position_bias.png
```

The aggregator (`scripts/aggregate_medical_results.py`) walks every judge
subdirectory under `baseline_blind/` and `baseline_oracle/`, so PGR is
computed for any debate arm whose judge has a matching baseline here. The
Streamlit frontend reads from this directory by default; flip "Recompute
baselines" in the sidebar to force a fresh per-run baseline pass for both
judges.

## When to regenerate

Regenerate canonical baselines whenever the baseline-relevant configuration
changes. Since E1/E4 now pair against frontier-judge baselines and E2/E3
against weaker-judge baselines, **either** judge model being swapped
invalidates the matching slice of this folder.

Regenerate when:

- The **weaker judge model** (`weaker` in `app.runner.FAMILY_MODELS`) is
  swapped — invalidates the weaker-judge baselines used by E2/E3.
- The **frontier judge model** (`frontier` in `app.runner.FAMILY_MODELS`)
  is swapped — invalidates the frontier-judge baselines used by E1/E4.
- The baseline prompts in
  `core/config/experiment/judge/baselines/medical_*.yaml` are edited.
- The DDXPlus pilot CSV (`data/ddxplus/ddxplus_debate_pilot_100.csv`) is
  rebuilt with different filtering.

In contrast, if only the **debate setup** changes — frontier-debater
prompts, BoN, rollout config, preference-judge — the baselines are still
valid (no judging-time change for the blind/oracle judges).

## How to regenerate

Both judges in one go (the default):

```bash
# OpenAI — runs gpt-5.4-mini AND gpt-5.5 baselines
./scripts/run_medical_baselines.sh 100 data/canonical_baselines/openai openai

# Anthropic — runs claude-sonnet-4-6 AND claude-opus-4-6 baselines
./scripts/run_medical_baselines.sh 100 data/canonical_baselines/anthropic anthropic
```

Just one judge (e.g. to add the frontier baselines without re-spending on
the weaker ones):

```bash
./scripts/run_medical_baselines.sh 100 data/canonical_baselines/openai openai gpt-5.5
./scripts/run_medical_baselines.sh 100 data/canonical_baselines/anthropic anthropic claude-opus-4-6
```

Each invocation runs blind + oracle across the 100 DDXPlus pilot cases for
each listed judge, in both original and swapped A/B orderings, and emits a
per-judge `plots/<judge>/*.png` set automatically via
`scripts/plot_baselines.py`. Rough cost per family at n=100: weaker judge
is ~$0.15; the frontier baseline is the dominant spend (~$1–2 for OpenAI
`gpt-5.5`; multiples of that for Anthropic `claude-opus-4-6` because
oracle calls carry the full patient evidence).

After regenerating, commit the new canonical baselines and note which
configuration they reflect so downstream comparisons stay honest.
