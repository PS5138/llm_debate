# Canonical Baselines

This directory holds **frozen blind + oracle baseline results** that every
debate run is compared against for PGR computation. By treating baselines as
fixed reference constants once `(judge_model, judge_prompt, dataset)` are
pinned, PGR is anchored to a stable floor/ceiling rather than re-rolling
slightly different baselines each experiment — which is how Khan et al. ran
QuALITY too.

## Layout

```
data/canonical_baselines/<family>/
  baseline_blind/
    data0.csv                                  # working schema, not used downstream
    <judge>/data0_judgement.csv               # judged in original A/B order
    <judge>/data0_swap_judgement.csv          # judged in swapped A/B order
  baseline_oracle/
    ... same shape ...
  results.csv                                  # per-arm accuracy summary
  plots/01_blind_vs_oracle.png
  plots/02_position_bias.png
```

The aggregator (`scripts/aggregate_medical_results.py`) reads the
`<judge>/data0[_swap]_judgement.csv` files. The Streamlit frontend reads from
this directory by default; you can flip "Recompute baselines" in the sidebar
to force a fresh per-run baseline pass.

## When to regenerate

Regenerate canonical baselines whenever the baseline-relevant configuration
changes:

- The weaker judge model (`weaker` in `app.runner.FAMILY_MODELS`) is swapped.
- The baseline prompts in `core/config/experiment/judge/baselines/medical_*.yaml`
  are edited.
- The DDXPlus pilot CSV (`data/ddxplus/ddxplus_debate_pilot_100.csv`) is
  rebuilt with different filtering.

If only the debate setup changes (frontier model, BoN, prompts), the
baselines are still valid.

## How to regenerate

For OpenAI (the headline weaker judge is `gpt-5.4-mini`):

```bash
./scripts/run_medical_baselines.sh 100 data/canonical_baselines/openai openai
```

For Anthropic (weaker judge is `claude-sonnet-4-6`):

```bash
./scripts/run_medical_baselines.sh 100 data/canonical_baselines/anthropic anthropic
```

Either command runs blind + oracle across all 100 DDXPlus pilot cases, in
both original and swapped A/B orderings, and emits the two `plots/*.png`
files automatically via `scripts/plot_baselines.py`. Cost is small (~$0.15
for the OpenAI family; broadly similar for Anthropic since Sonnet 4.6 has
no adaptive-thinking overhead at judge time).

After regenerating, commit the new canonical baselines and note which
configuration they reflect so downstream comparisons stay honest.
