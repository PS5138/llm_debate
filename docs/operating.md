# Operating guide

Companion to the project [`README.md`](../README.md). Experiments in this fork
are launched and controlled through the local Streamlit app. The former shell
workflows have been removed: there is no supported command-line path for paid
baseline, debate, or judging runs.

> **Research use only.** This app uses synthetic patients to study AI
> oversight. It is not medical advice, a diagnostic system, or clinical
> decision support, and it must not be used for real-patient decisions.

## Launch the local app

From the repository root, after installing `requirements.txt` into `.venv`:

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py --server.address=127.0.0.1
```

Open the local URL printed by Streamlit, normally `http://127.0.0.1:8501`.
Binding to `127.0.0.1` keeps the unauthenticated research interface off the
local network. Do not expose it publicly without adding authentication, TLS,
and host isolation.

The Streamlit command only starts the web interface. All experiment choices,
validation, cost confirmation, provider calls, stopping, and resume/extension
are owned by the app. Direct invocation of paid `core.*` stages is unsupported
and blocked so it cannot bypass those controls.

## Start a fresh run

In the sidebar:

1. Pick `openai` or `anthropic` and choose between 1 and 100 cases from the
   committed DDXPlus pilot.
2. Leave the canonical baseline setting selected unless you deliberately need
   a new per-run baseline measurement. Recomputing baselines adds provider calls
   and cost.
3. Paste the required API key or keys into the password fields. Standard
   Anthropic-family runs need both an Anthropic key and an OpenAI key because
   concession quality control uses `gpt-4o-mini`; OpenAI-family runs need an
   OpenAI key.
4. Review the planning estimate. It covers debate and judging and, when
   selected, adds a separate allowance for baseline recomputation. It is a
   budget guide rather than a guaranteed invoice: token usage and retries vary.
5. Click **Run**, then click the confirmation button. No provider calls begin
   on the first click.

Start with 1–3 cases to verify account access and observe real spend before
scaling. The current app uses `gpt-5.5` for OpenAI debaters, preference judging,
and E1/E4 final judging; `gpt-5.4-mini` handles E2/E3. For Anthropic it uses
`claude-opus-4-6` in the frontier roles and `claude-sonnet-4-6` for E2/E3.

A fresh run creates
`exp/YYYY-MM-DD_HH-MM-SS_results/`. The app performs, in order:

1. validated canonical-baseline reuse, or fresh blind/oracle baselines when
   explicitly selected;
2. three-round, BoN=4 debate generation;
3. E1–E4 final judging in original and swapped answer order;
4. concession quality control;
5. analysis, paired statistics, plots, and consolidated result tables.

Completed debate and judgement cases are checkpointed as they finish. The case
table can therefore update during a stage, and a later stop or failure retains
already checkpointed work for resume. Calls that were still in flight at the
instant of a hard interruption can still be retried.

## Buttons and run state

- **Stop** requests cancellation and escalates from process-group termination
  if the current provider process does not exit promptly. Check the final status
  before assuming every in-flight request was cancelled.
- **Clear run** stops and removes the displayed run, resets both API-key fields
  and the confirmation state, and removes the run's temporary key,
  prompt-cache, and log files. It does not reset unrelated Streamlit settings
  or delete the persistent timestamped result folder.
- **Resume / extend previous _family_ run** appears for the newest valid saved
  run belonging to the selected model family. Changing the family changes the
  candidate; an OpenAI run is never offered while Anthropic is selected, or
  vice versa.

## Resume and extension

Resume requires a second confirmation click because it can make paid calls.
The app displays the saved run, its model roles, baseline source, current case
count, and a conservative full-rerun planning estimate.

- Leave the target at the saved count to retry incomplete stages. A row is
  reused only when that stage's explicit completion flag is true; incomplete or
  unparseable work is retried.
- Raise the target to extend the same run up to the 100-case pilot ceiling. The
  app grows the working CSVs by case ID and preserves prior completion state.
- The recorded frontier, weaker, and concession models are retained for the
  resumed family. Missing, malformed, or contradictory saved metadata fails
  closed instead of silently switching to current defaults.
- The original baseline source is retained. Canonical and complete external
  baselines are read-only. An external archive must cover every requested case
  and both recorded answer judges; otherwise extension is refused.
- Already complete work is skipped, so actual resume spend should normally be
  below the full-rerun estimate. Incomplete calls, failed responses, and
  provider retries can still cost money.

If both model families have valid cached results inside the same timestamped
run root, the shared result tables retain both families. A malformed sibling
family or mismatched case cohort is excluded or rejected rather than silently
mixed into the comparison.

## Baselines

Canonical 100-case blind and oracle references for both model families are
committed under [`../data/canonical_baselines/`](../data/canonical_baselines/).
The app validates the requested cases, both answer-judge models, completion
flags, and parseable answers before treating a bundle as usable. Valid
canonical bundles are reused read-only and add no new baseline API spend.

The **Recompute baselines** toggle creates a fresh baseline subtree inside the
new timestamped run. It never regenerates a committed canonical folder in
place. This distinction matters because judgement files are caches: writing a
new configuration over an old folder can otherwise reuse stale completed rows.

Canonical replacement is a maintainer release operation, not a user run mode.
A replacement must begin in a fresh isolated app run with recomputation enabled,
complete every case in both answer orders for both judges, pass validation, and
only then deliberately replace the committed bundle as a whole. See the
[canonical-baseline notes](../data/canonical_baselines/README.md).

## Outputs, keys, and privacy

The main persistent layout is:

```text
exp/YYYY-MM-DD_HH-MM-SS_results/
  run_metadata.json
  baselines/<family>/          # present only for per-run recomputation
  <family>/
    debate_sim/                # working transcripts and judgements
    analysis/                  # five QC CSVs, summaries, and plots
  medical_results/             # headline tables, paired tests, and plots
```

`exp/` is gitignored, but it intentionally persists so later app sessions can
resume it. It can contain complete model transcripts, judgements, provider logs,
and model-output caches; inspect it before sharing. API keys and the app's
prompt history are stored under a permission-restricted temporary directory,
not in the repository or persistent result folder, and are removed when the run
reaches a terminal app state or **Clear run** is clicked. Clear run also empties
both password fields. As with any local process, an ungraceful machine or server
kill can interrupt application cleanup; the files remain under the operating
system's temporary area rather than the repository, and should be removed before
sharing a machine image.

The app sends prompts only to the selected model providers. Streamlit's own
anonymous usage telemetry is disabled in the repository configuration. The app
does not provide arbitrary CSV uploads and always uses the committed pilot.

## Internal maintenance utilities

The following scripts remain because the app uses them after paid stages, or
because maintainers use them to prepare/inspect committed artifacts. They are
not alternative experiment launchers:

| File | Scope |
|---|---|
| `scripts/prepare_ddxplus_debate_pilot.py` | No-API dataset preparation from locally downloaded DDXPlus release files. |
| `scripts/analyze_medical_debate.py` | No-API QC tables, summaries, and plots from an existing run cache. |
| `scripts/aggregate_medical_results.py` | No-API accuracy, PGR, per-case tables, and plots from existing completed judgements. |
| `scripts/compute_paired_tests.py` | No-API headline, paired-test, and QC rollup tables. |
| `scripts/medical_stats.py` | Shared paired bootstrap, permutation, and Holm calculations; imported by the analysis tools. |
| `scripts/plot_baselines.py` | No-API plots for an already completed baseline bundle. |
| `scripts/summarise_run_costs.py` | No-API extraction of cost totals already recorded in run logs. |
| `scripts/export_medical_debate_output.py` | No-API readable export of an existing completed debate. |

After paid stages, the app automatically invokes
`analyze_medical_debate.py`, `aggregate_medical_results.py`, and
`compute_paired_tests.py`. The dataset, baseline-plot, cost-summary, and readable
export utilities are maintainer tools and are not run automatically. Their
inputs are raw run caches under `exp/`, which are not committed. Consequently a
fresh clone can verify and inspect the curated result bundle but cannot rebuild
every curated artifact from raw judgements. A new app run creates a new
stochastic replication and incurs provider cost; it is not a byte-for-byte
reconstruction of the published OpenAI source run.
