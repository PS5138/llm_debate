#!/usr/bin/env bash
# Run the medical debate pipeline for one model family.
#
# Usage:
#   ./scripts/run_medical_debate.sh [N] [EXP_DIR] [FAMILY] [BASELINES_DIR]
#
# Examples:
#   ./scripts/run_medical_debate.sh 1 exp/medical_debate_smoke openai
#   ./scripts/run_medical_debate.sh 20 exp/medical_debate_n20 openai
#   ./scripts/run_medical_debate.sh 100 exp/medical_debate_n100 openai exp/medical_debate_n100/baselines

set -euo pipefail

LIMIT="${1:-100}"
EXP_DIR="${2:-exp/medical_debate_n${LIMIT}}"
FAMILY="${3:-openai}"
BASELINES_DIR="${4:-${BASELINES_DIR:-${EXP_DIR}/baselines}}"
THREADS="${THREADS:-5}"
PYTHON="${PYTHON:-.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/medical-debate-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

if [[ "$FAMILY" == "anthropic" ]]; then
  FRONTIER="${FRONTIER:-claude-opus-4-7}"
  WEAKER="${WEAKER:-claude-sonnet-4-6}"
elif [[ "$FAMILY" == "openai" ]]; then
  FRONTIER="${FRONTIER:-gpt-5.5}"
  WEAKER="${WEAKER:-gpt-5.4-mini}"
else
  echo "unknown family: ${FAMILY}" >&2
  exit 1
fi

EXP="${EXP_DIR}/${FAMILY}"

DEBATE_OVERRIDES=(
  "++limit=${LIMIT}"
  "++anthropic_num_threads=${THREADS}"
  "++correct_debater.language_model.model=${FRONTIER}"
  "++incorrect_debater.language_model.model=${FRONTIER}"
  "++correct_preference.language_model.model=${FRONTIER}"
  "++incorrect_preference.language_model.model=${FRONTIER}"
  "++correct_debater.BoN=4"
  "++incorrect_debater.BoN=4"
  "++correct_debater.language_model.temperature=0.8"
  "++incorrect_debater.language_model.temperature=0.8"
)

echo ">>> Generating medical debate transcripts: n=${LIMIT} exp=${EXP} family=${FAMILY} frontier=${FRONTIER}"
"$PYTHON" -m core.debate "exp_dir=${EXP}" "+experiment=medical_debate" "${DEBATE_OVERRIDES[@]}"

declare -a CONDITIONS=(
  "e1_info_asymmetry medical_debate ${FRONTIER}"
  "e2_double_asymmetry medical_debate_e2_double_asymmetry ${WEAKER}"
  "e3_capability_asymmetry medical_debate_e3_capability_asymmetry ${WEAKER}"
  "e4_full_symmetry medical_debate_e4_full_symmetry ${FRONTIER}"
)

for item in "${CONDITIONS[@]}"; do
  read -r COND EXPERIMENT JUDGE_MODEL <<< "$item"
  JUDGE_NAME="${COND}_${JUDGE_MODEL}"
  COMMON_ARGS=(
    "exp_dir=${EXP}"
    "+experiment=${EXPERIMENT}"
    "++limit=${LIMIT}"
    "++anthropic_num_threads=${THREADS}"
    "++judge.language_model.model=${JUDGE_MODEL}"
    "++judge_name=${JUDGE_NAME}"
  )

  echo ">>> Judging ${COND}: judge=${JUDGE_MODEL}"
  "$PYTHON" -m core.judge "${COMMON_ARGS[@]}"
  "$PYTHON" -m core.scoring.accuracy "${COMMON_ARGS[@]}"
done

CONCESSION_MODEL="${CONCESSION_MODEL:-gpt-4o-mini}"
echo ">>> Running concession judge (${CONCESSION_MODEL}; Y/N only — keep on a cheap model)"
"$PYTHON" -m core.judge \
  "exp_dir=${EXP}" \
  "+experiment=medical_debate" \
  "++limit=${LIMIT}" \
  "++anthropic_num_threads=${THREADS}" \
  "++judge_type=concession" \
  "++concession_judge.language_model.model=${CONCESSION_MODEL}" \
  "++judge_name=concession_${CONCESSION_MODEL}"

echo ">>> Running bias-control analyses (verbosity / quote-rate / concession)"
"$PYTHON" scripts/analyze_medical_debate.py "${EXP}"

echo ">>> Aggregating accuracy + PGR + per-case lift (no API spend)"
"$PYTHON" scripts/aggregate_medical_results.py "${EXP}" \
  --baselines-dir "${BASELINES_DIR}" \
  --out-dir "${EXP_DIR}/medical_results"

echo ">>> Summarising API spend from logs"
"$PYTHON" scripts/summarise_run_costs.py "${EXP}"

"$PYTHON" scripts/export_medical_debate_output.py "${EXP}" --limit "${LIMIT}" --family "${FAMILY}"

echo ">>> Wrote consolidated output to ${EXP}/one_debate_outputs.md"
echo ">>> Wrote analyses to ${EXP}/analysis/"
echo ">>> Wrote aggregated results to ${EXP_DIR}/medical_results/"
echo ">>> Used baselines from ${BASELINES_DIR}/ if present"
echo ">>> Wrote cost summary to ${EXP}/cost_summary.csv"
