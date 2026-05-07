#!/usr/bin/env bash
# Run blind + oracle DDXPlus baselines.
#
# Usage:
#   ./scripts/run_medical_baselines.sh [N]            # N cases, auto-named exp dir
#   ./scripts/run_medical_baselines.sh [N] [EXP_DIR]  # explicit exp dir
#
# Examples:
#   ./scripts/run_medical_baselines.sh 20             # 20-case smoke -> exp/medical_pilot_n20
#   ./scripts/run_medical_baselines.sh 100            # 100-case pilot -> exp/medical_pilot_n100
#   ./scripts/run_medical_baselines.sh 5 exp/foo      # custom exp dir

set -euo pipefail

LIMIT="${1:-50}"
EXP_DIR="${2:-exp/medical_pilot_n${LIMIT}}"
THREADS="${THREADS:-5}"
JUDGE_NAME="${JUDGE_NAME:-gpt-4o-mini}"

COMMON_ARGS=(
  "++limit=${LIMIT}"
  "++anthropic_num_threads=${THREADS}"
  "++judge_name=${JUDGE_NAME}"
)

echo ">>> n=${LIMIT}  exp_dir=${EXP_DIR}  judge=${JUDGE_NAME}"

for arm in medical_blind medical_oracle; do
  python -m core.debate           "exp_dir=${EXP_DIR}" "+experiment=${arm}" "${COMMON_ARGS[@]}"
  python -m core.judge            "exp_dir=${EXP_DIR}" "+experiment=${arm}" "${COMMON_ARGS[@]}"
  python -m core.scoring.accuracy "exp_dir=${EXP_DIR}" "+experiment=${arm}" "${COMMON_ARGS[@]}"
done
