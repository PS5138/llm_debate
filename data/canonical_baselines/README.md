# Canonical Baselines

This directory contains the **frozen blind and oracle reference results** used
for Performance Gap Recovered (PGR). Treating the baselines as constants once
`(judge model, judge prompt, dataset)` are fixed anchors each debate run to the
same floor and ceiling instead of rerolling them every time.

The Streamlit app validates and reuses these bundles read-only by default. It
never resumes into or regenerates a committed canonical directory in place.

The current committed bundles predate a provenance manifest. The repository
validates their model-named layout, exact 100-case pilot cohort, completion
flags, and parseable answers, but it cannot independently verify the exact
prompt/dataset revision, source run, or generation date that produced them.
Treat them as this project's published reference outputs; every future
replacement must include that provenance rather than trying to infer it from
file names or modification times.

## Layout

A complete family bundle carries both the weaker and frontier answer judges.
E2/E3 use the weaker-judge references; E1/E4 use the frontier-judge references.

```text
data/canonical_baselines/<family>/
  baseline_blind/
    data0.csv
    <weaker_judge>/data0_judgement.csv
    <weaker_judge>/data0_swap_judgement.csv
    <frontier_judge>/data0_judgement.csv
    <frontier_judge>/data0_swap_judgement.csv
  baseline_oracle/
    ...same judge layout...
  results.csv
  plots/<judge>/01_blind_vs_oracle.png
  plots/<judge>/02_position_bias.png
```

`data0.csv` is a working-schema file. Downstream PGR calculations read the
completed original/swap judgement CSVs and require an exact pilot case cohort,
parseable answers, and matching judge model.

## When a replacement is required

The matching canonical slice is invalidated when any baseline-relevant input
changes:

- the weaker judge model used by E2/E3;
- the frontier judge model used by E1/E4;
- either blind/oracle judge prompt under
  `core/config/experiment/judge/baselines/`;
- the contents or filtering of
  `data/ddxplus/ddxplus_debate_pilot_100.csv`; or
- answer parsing/scoring behavior that changes which judgement rows count.

Changes confined to debate generation—debater prompts, BoN, rollout settings,
or preference selection—do not invalidate blind/oracle measurements.

## Controlled replacement procedure

Canonical replacement is a maintainer release operation, not a public run mode.
Do not point a paid stage at this directory, and do not copy new inputs over an
old working folder before judging. Completed judgement rows are cached; reusing
that folder could silently preserve answers produced with the previous prompt,
dataset, or model.

Instead:

1. Start a **fresh** Streamlit run for the intended family at 100 cases.
2. Enable **Recompute baselines** in the sidebar and complete the app's cost
   confirmation. The app writes a new isolated baseline subtree under that
   timestamped `exp/` run.
3. Let both blind and oracle arms finish in original and swapped answer order
   for both the weaker and frontier judges.
4. Validate exact case IDs, unique rows, completion flags, parseable answers,
   model directory names, and expected row counts. Review the baseline accuracy
   and position-bias plots.
5. Commit a provenance manifest with the model IDs, prompt/dataset revision,
   source run, and generation date. Only after review, deliberately replace the
   family bundle as one unit and run the repository's no-API regression suite.

Never merge a new partial bundle into an existing canonical tree. Starting from
an empty isolated run is what guarantees that no old completed judgement is
mistaken for a regenerated result.

The app displays a planning allowance before any recomputation. The amount is
not guaranteed: oracle prompts include full patient evidence, provider token
usage varies, and the frontier judge is normally the dominant baseline cost.
