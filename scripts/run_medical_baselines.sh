#!/usr/bin/env bash
# Run blind + oracle DDXPlus baselines for one or more judges in one go.
#
# Usage:
#   ./scripts/run_medical_baselines.sh [N] [EXP_DIR] [FAMILY] [JUDGE_MODELS]
#
# The 4th argument is a comma-separated list of judge model IDs. If omitted,
# both the weaker AND frontier judge for the chosen family are run, which
# is what we want when (re)building the canonical baselines folder so that
# every debate arm has a same-judge baseline available for PGR.
#
#   default JUDGE_MODELS for openai   : "gpt-5.4-mini,gpt-5.5"
#   default JUDGE_MODELS for anthropic: "claude-sonnet-4-6,claude-opus-4-6"
#
# Examples:
#   ./scripts/run_medical_baselines.sh 20                                       # both openai judges
#   ./scripts/run_medical_baselines.sh 100 data/canonical_baselines/openai openai
#   ./scripts/run_medical_baselines.sh 100 data/canonical_baselines/anthropic anthropic
#   ./scripts/run_medical_baselines.sh 100 data/canonical_baselines/openai openai gpt-5.5   # frontier only

set -euo pipefail

LIMIT="${1:-100}"
FAMILY="${3:-${FAMILY:-openai}}"
if [[ $# -ge 2 && -n "${2:-}" ]]; then
  EXP_DIR="$2"
else
  RUN_ROOT="${RUN_ROOT:-$(./scripts/create_results_dir.sh)}"
  EXP_DIR="${RUN_ROOT}/baselines/${FAMILY}"
fi
THREADS="${THREADS:-5}"
PYTHON="${PYTHON:-.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

if [[ "$FAMILY" == "anthropic" ]]; then
  DEFAULT_JUDGES="claude-sonnet-4-6,claude-opus-4-6"
elif [[ "$FAMILY" == "openai" ]]; then
  DEFAULT_JUDGES="gpt-5.4-mini,gpt-5.5"
else
  echo "unknown family: ${FAMILY}" >&2
  exit 1
fi

JUDGE_MODELS="${4:-${JUDGE_MODELS:-${DEFAULT_JUDGES}}}"
IFS=',' read -ra MODELS <<< "${JUDGE_MODELS}"

echo ">>> n=${LIMIT}  exp_dir=${EXP_DIR}  family=${FAMILY}  judges=${JUDGE_MODELS}"

for arm in medical_blind medical_oracle; do
  # core.debate writes the baseline data0.csv (zero rollout, no API
  # calls) and doesn't depend on the judge model — run it once per arm
  # using the first listed judge to satisfy Hydra's required config.
  "$PYTHON" -m core.debate \
    "exp_dir=${EXP_DIR}" "+experiment=${arm}" \
    "++limit=${LIMIT}" "++anthropic_num_threads=${THREADS}" \
    "++judge.language_model.model=${MODELS[0]}" \
    "++judge_name=${MODELS[0]}"

  for JUDGE_MODEL in "${MODELS[@]}"; do
    "$PYTHON" -m core.judge \
      "exp_dir=${EXP_DIR}" "+experiment=${arm}" \
      "++limit=${LIMIT}" "++anthropic_num_threads=${THREADS}" \
      "++judge.language_model.model=${JUDGE_MODEL}" \
      "++judge_name=${JUDGE_MODEL}"
    "$PYTHON" -m core.scoring.accuracy \
      "exp_dir=${EXP_DIR}" "+experiment=${arm}" \
      "++limit=${LIMIT}" "++anthropic_num_threads=${THREADS}" \
      "++judge.language_model.model=${JUDGE_MODEL}" \
      "++judge_name=${JUDGE_MODEL}"
  done
done

# Emit blind-vs-oracle plots from the resulting results.csv (no API calls).
if [[ -f "${EXP_DIR}/results.csv" ]]; then
  echo ">>> Plotting blind vs oracle (one plot per judge)"
  "$PYTHON" scripts/plot_baselines.py "${EXP_DIR}"
fi
