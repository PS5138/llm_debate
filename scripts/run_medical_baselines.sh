#!/usr/bin/env bash
# Run blind + oracle DDXPlus baselines.
#
# Usage:
#   ./scripts/run_medical_baselines.sh [N] [EXP_DIR] [FAMILY] [JUDGE_MODEL]
#
# Examples:
#   ./scripts/run_medical_baselines.sh 20
#   ./scripts/run_medical_baselines.sh 100 exp/medical_debate_n100/baselines/openai openai
#   ./scripts/run_medical_baselines.sh 100 exp/medical_debate_n100/baselines/anthropic anthropic

set -euo pipefail

LIMIT="${1:-100}"
EXP_DIR="${2:-exp/medical_pilot_n${LIMIT}}"
FAMILY="${3:-openai}"
THREADS="${THREADS:-5}"
PYTHON="${PYTHON:-.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

if [[ "$FAMILY" == "anthropic" ]]; then
  JUDGE_MODEL="${4:-${JUDGE_MODEL:-claude-sonnet-4-6}}"
elif [[ "$FAMILY" == "openai" ]]; then
  JUDGE_MODEL="${4:-${JUDGE_MODEL:-gpt-5.4-mini}}"
else
  echo "unknown family: ${FAMILY}" >&2
  exit 1
fi

JUDGE_NAME="${JUDGE_NAME:-${JUDGE_MODEL}}"

COMMON_ARGS=(
  "++limit=${LIMIT}"
  "++anthropic_num_threads=${THREADS}"
  "++judge.language_model.model=${JUDGE_MODEL}"
  "++judge_name=${JUDGE_NAME}"
)

echo ">>> n=${LIMIT}  exp_dir=${EXP_DIR}  family=${FAMILY}  judge=${JUDGE_MODEL}  judge_name=${JUDGE_NAME}"

for arm in medical_blind medical_oracle; do
  "$PYTHON" -m core.debate           "exp_dir=${EXP_DIR}" "+experiment=${arm}" "${COMMON_ARGS[@]}"
  "$PYTHON" -m core.judge            "exp_dir=${EXP_DIR}" "+experiment=${arm}" "${COMMON_ARGS[@]}"
  "$PYTHON" -m core.scoring.accuracy "exp_dir=${EXP_DIR}" "+experiment=${arm}" "${COMMON_ARGS[@]}"
done
