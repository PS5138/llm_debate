# Operating guide

Companion document to the abstract-level [`README.md`](../README.md). The
README's *How To Run* section covers the quickest paths (Streamlit demo and
the one-command wrapper). This document covers the manual stages, re-running
analysis without API calls, the full Streamlit walkthrough, and the script
map.

## Interactive frontend in detail

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Use `python -m streamlit` rather than a bare `streamlit`. The latter resolves
through `PATH` and commonly picks up an Anaconda or system Streamlit that has
none of this project's dependencies, failing with `ModuleNotFoundError: No
module named 'hydra'`.

If you moved the repo after creating `.venv`, the venv's console scripts
(`.venv/bin/streamlit`, `.venv/bin/pip`, ...) still point at the old absolute
path and will not run. `.venv/bin/python` itself keeps working, so
`python -m <tool>` is unaffected; to repair the scripts, recreate the
environment:

```bash
rm -rf .venv
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Once the page loads:

- Pick a model family (`openai` or `anthropic`) and a number of cases
  between **1 and 100** from the committed DDXPlus pilot.
- Paste your own API key(s) — they live only in the Streamlit session and
  are written to a session-scoped tempfile, never to your repo `SECRETS`.
- Hit **Run**. The app shells out to the same pipeline stages the CLI
  uses (baselines → debate → E1-E4 final judging → concession judge →
  analysis → aggregation) under a fresh
  `exp/YYYY-MM-DD_HH-MM-SS_results/` folder.
- The case table fills in row by row as debates complete. Pick any case
  to read the full transcript (Round 1, 2, 3 with both debaters
  labelled) and see which side each E1-E4 judge picked under the
  original and swapped A/B orderings.
- When the run finishes, the bottom of the page shows the aggregate
  accuracy/PGR tables and the generated plots from `medical_results/`.

Buttons:

- **Clear run** deletes only the temporary session SECRETS/log files
  and leaves the timestamped `exp/` output folder in place.
- **Stop** cancels the current run (SIGTERM → SIGKILL escalation if
  the subprocess doesn't exit within ~5s).
- **Resume / extend previous run** appears in the sidebar whenever a
  previous timestamped results folder is detected under `exp/`. It
  exposes a slider for the total case count after the run:
  - Leave the slider at the existing `n` to re-run the same pipeline
    against that folder. Stages with `complete=True` rows are skipped,
    so if a run failed after 66 of 100 cases finished debating, the
    resume only re-attempts cases 67–100.
  - Move the slider higher than the existing `n` to **extend** the run
    in place. The working `data0.csv` files (and any existing
    judgement CSVs) are grown to the new total, preserving completion
    state on earlier cases. The pipeline then only pays the API cost
    for the newly-added cases. This is the recommended way to do
    chunked review-as-you-go workflows ("run 10, look at results,
    extend to 20, look, extend to 100").

Things to know before clicking **Run**:

- **You pay for your run.** The sidebar shows a rough cost estimate
  based on family and case count; treat it as a ballpark, not an
  invoice. Start with 1–3 cases.
- **The OpenAI key is always required** because the concession
  (quality control) judge uses `gpt-4o-mini` regardless of family.
  The Anthropic key is only needed if you pick the `anthropic` family.
- Results are written under `exp/`, which is gitignored during
  development. There is no database, no upload, and no shared state
  between sessions. Every run gets a new local results folder and
  uses the user's own key.
- The app uses the same `data/ddxplus/ddxplus_debate_pilot_100.csv`
  that is committed to this repo. It does not accept arbitrary CSV
  uploads.

## One-command wrapper in detail

The README documents the basic invocation. A few additional details:

If you do not pass an output path, the wrapper creates a fresh folder
like `exp/2026-05-17_19-30-00_results/`. If you intentionally want to
reuse or resume a specific run folder, pass it as the third argument:

```bash
./scripts/run_medical_full.sh 100 openai exp/2026-05-17_19-30-00_results
```

To put both families in the same final results folder, create the
folder once and pass it to both wrapper calls:

```bash
RUN_ROOT="$(./scripts/create_results_dir.sh)"
./scripts/run_medical_full.sh 100 openai "$RUN_ROOT"
./scripts/run_medical_full.sh 100 anthropic "$RUN_ROOT"
```

The second call keeps the existing family outputs and regenerates
`$RUN_ROOT/medical_results/` using all completed family directories
it finds. Raw family outputs are separated under `$RUN_ROOT/openai/`,
`$RUN_ROOT/anthropic/`, `$RUN_ROOT/baselines/openai/`, and
`$RUN_ROOT/baselines/anthropic/`; only the shared aggregate
`$RUN_ROOT/medical_results/` is rewritten so it reflects the
currently available families.

The wrapper script:

1. Runs the local parser test unless `RUN_TESTS=0`.
2. Checks that the prepared DDXPlus pilot exists, rebuilding it if
   needed and raw files are available.
3. Runs blind and oracle baselines for **both** the weaker and frontier
   judges (or skips, if canonical baselines are in use).
4. Runs debate transcript generation.
5. Runs E1-E4 final judging.
6. Runs concession judging with `gpt-4o-mini` by default.
7. Runs bias-control analysis, aggregation, cost summary, and
   transcript export.

Useful overrides:

```bash
THREADS=2 ./scripts/run_medical_full.sh 100 openai
RUN_TESTS=0 ./scripts/run_medical_full.sh 100 openai
PREPARE_DATA=1 ./scripts/run_medical_full.sh 100 openai
FRONTIER=gpt-5.5 WEAKER=gpt-5.4-mini ./scripts/run_medical_full.sh 100 openai
```

Use a small smoke test first. The full run makes many API calls
because each debate turn uses BoN candidate generation and preference
judging.

## Manual run

If you prefer to run each stage yourself:

```bash
# 0. Create one fresh results folder and reuse it for all manual stages.
RUN_ROOT="$(./scripts/create_results_dir.sh)"

# 1. Optional: rebuild the prepared pilot from raw DDXPlus files.
python scripts/prepare_ddxplus_debate_pilot.py

# 2. Run blind and oracle baselines (defaults to running BOTH the weaker
#    and frontier judges, so every debate arm has a same-judge floor and
#    ceiling for PGR). Pass a comma-separated 4th argument to restrict.
./scripts/run_medical_baselines.sh \
  100 \
  "$RUN_ROOT/baselines/openai" \
  openai
# e.g. frontier only:
#   ./scripts/run_medical_baselines.sh 100 "$RUN_ROOT/baselines/openai" openai gpt-5.5

# 3. Run debate generation, E1-E4 judging, concession checks, analysis, and export.
./scripts/run_medical_debate.sh \
  100 \
  "$RUN_ROOT" \
  openai \
  "$RUN_ROOT/baselines/openai"
```

Replace `openai` with `anthropic` for the Anthropic model family (the
Anthropic defaults are `claude-sonnet-4-6,claude-opus-4-6`).

## Re-run analysis without API calls

Once the judgement CSVs exist, you can regenerate analysis without
repeating the expensive model calls:

```bash
python scripts/analyze_medical_debate.py \
  exp/YYYY-MM-DD_HH-MM-SS_results/openai \
  --baselines-dir data/canonical_baselines/openai

python scripts/aggregate_medical_results.py \
  exp/YYYY-MM-DD_HH-MM-SS_results/openai \
  --baselines-dir data/canonical_baselines/openai \
  --out-dir exp/YYYY-MM-DD_HH-MM-SS_results/medical_results

python scripts/compute_paired_tests.py \
  exp/YYYY-MM-DD_HH-MM-SS_results/openai \
  --baselines-dir data/canonical_baselines/openai \
  --out-dir exp/YYYY-MM-DD_HH-MM-SS_results/medical_results

# Pass the baselines dir too, or fresh baseline spend is left out of the total.
python scripts/summarise_run_costs.py \
  exp/YYYY-MM-DD_HH-MM-SS_results/openai \
  exp/YYYY-MM-DD_HH-MM-SS_results/baselines/openai

python scripts/export_medical_debate_output.py \
  exp/YYYY-MM-DD_HH-MM-SS_results/openai \
  --limit 100 \
  --family openai
```

If you ran fresh per-run baselines instead of using canonical, swap
`--baselines-dir` to point at
`exp/YYYY-MM-DD_HH-MM-SS_results/baselines/openai` for both commands
that take it.

## Script map

| File | Purpose |
|---|---|
| `app/streamlit_app.py` | Local frontend: live row-by-row debate viewer and aggregate dashboard. |
| `app/runner.py` | Pipeline runner used by the local frontend; writes runs under timestamped `exp/` folders. |
| `scripts/run_medical_full.sh` | One-command wrapper for baselines, debate, judging, analysis, and export. |
| `scripts/create_results_dir.sh` | Creates a fresh `exp/YYYY-MM-DD_HH-MM-SS_results/` folder for manual runs. |
| `scripts/write_run_metadata.py` | Records family/model settings in `run_metadata.json`. |
| `scripts/prepare_ddxplus_debate_pilot.py` | Rebuilds the deterministic 100-case DDXPlus pilot. |
| `scripts/run_medical_baselines.sh` | Runs blind and oracle baselines for one model family. |
| `scripts/run_medical_debate.sh` | Generates debate transcripts, runs E1-E4 final judges, concession checks, analysis, aggregation, cost summary, and export. |
| `scripts/analyze_medical_debate.py` | Produces verbosity, quote verification, concession, debate-outcome, and position-bias outputs. |
| `scripts/aggregate_medical_results.py` | Produces accuracy, PGR, per-case lift tables, and result plots from cached judgement CSVs. |
| `scripts/compute_paired_tests.py` | Produces `headline_results.csv`, `paired_tests.csv`, and `quality_control_summary.csv` — the curated-bundle statistics. |
| `scripts/resolve_baselines.py` | Picks the baselines directory for a run, falling back to the committed canonical set. Called before any paid stage. |
| `scripts/medical_stats.py` | Shared paired statistics (bootstrap, permutation, Holm) used by the two scripts above. Not a CLI. |
| `scripts/plot_baselines.py` | Renders blind-vs-oracle and position-bias plots from a baselines `results.csv`. |
| `scripts/summarise_run_costs.py` | Extracts API spend from Hydra logs into `cost_summary.csv`. Takes several directories; pass the family dir and its baselines dir to cover a whole run. |
| `scripts/export_medical_debate_output.py` | Writes a readable Markdown export of one debate and its judgements. |
| `tests/test_accuracy.py` | Regression tests for parsing judge answers. |
