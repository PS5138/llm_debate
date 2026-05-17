#!/usr/bin/env bash
# Run the medical debate experiment end to end for one model family.
#
# Usage:
#   ./scripts/run_medical_full.sh [N] [FAMILY] [EXP_ROOT]
#
# Examples:
#   ./scripts/run_medical_full.sh 1 openai
#   ./scripts/run_medical_full.sh 100 openai
#   ./scripts/run_medical_full.sh 100 anthropic
#   ./scripts/run_medical_full.sh 100 openai exp/2026-05-17_19-30-00_results
#
# Useful environment overrides:
#   THREADS=2                      lower API concurrency
#   PYTHON=.venv/bin/python        choose interpreter
#   RUN_TESTS=0                    skip the local parser test
#   PREPARE_DATA=1                 rebuild the DDXPlus pilot before running
#   FRONTIER=gpt-5.5 WEAKER=...    override debate models
#   CONCESSION_MODEL=gpt-4o-mini   override concession judge model

set -euo pipefail

LIMIT="${1:-100}"
FAMILY="${2:-openai}"
if [[ $# -ge 3 && -n "${3:-}" ]]; then
  EXP_ROOT="$3"
elif [[ -n "${EXP_ROOT:-}" ]]; then
  EXP_ROOT="$EXP_ROOT"
elif [[ -n "${RUN_ROOT:-}" ]]; then
  EXP_ROOT="$RUN_ROOT"
else
  EXP_ROOT="$(./scripts/create_results_dir.sh)"
fi
BASELINES_DIR="${BASELINES_DIR:-${EXP_ROOT}/baselines/${FAMILY}}"
THREADS="${THREADS:-5}"
PYTHON="${PYTHON:-.venv/bin/python}"
RUN_TESTS="${RUN_TESTS:-1}"
PREPARE_DATA="${PREPARE_DATA:-auto}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

case "$FAMILY" in
  openai|anthropic) ;;
  *)
    echo "unknown family: ${FAMILY}; expected 'openai' or 'anthropic'" >&2
    exit 1
    ;;
esac

require_secret_key() {
  local key="$1"
  if [[ ! -f SECRETS ]] || ! grep -Eq "^${key}=.+" SECRETS; then
    echo "missing non-empty ${key}=... entry in top-level SECRETS" >&2
    exit 1
  fi
}

require_secret_line() {
  local key="$1"
  if [[ ! -f SECRETS ]] || ! grep -Eq "^${key}=" SECRETS; then
    echo "missing ${key}=... entry in top-level SECRETS" >&2
    exit 1
  fi
}

require_secret_key "API_KEY"
require_secret_line "ANTHROPIC_API_KEY"
require_secret_line "DEFAULT_ORG"

if [[ "$FAMILY" == "anthropic" ]]; then
  require_secret_key "ANTHROPIC_API_KEY"
fi

export THREADS
export PYTHON
export RUN_ROOT="$EXP_ROOT"

PILOT_CSV="data/ddxplus/ddxplus_debate_pilot_100.csv"
RAW_PATIENTS="data/ddxplus/release_test_patients"
EVIDENCE_DEFS="data/ddxplus/release_evidences.json"

echo ">>> Medical debate full run"
echo ">>> n=${LIMIT} family=${FAMILY} exp_root=${EXP_ROOT}"
echo ">>> baselines_dir=${BASELINES_DIR} threads=${THREADS} python=${PYTHON}"

if [[ "$RUN_TESTS" == "1" ]]; then
  echo ">>> Running local parser test"
  "$PYTHON" tests/test_accuracy.py
fi

if [[ "$PREPARE_DATA" == "1" || "$PREPARE_DATA" == "true" ]]; then
  echo ">>> Rebuilding prepared DDXPlus pilot"
  "$PYTHON" scripts/prepare_ddxplus_debate_pilot.py
elif [[ ! -f "$PILOT_CSV" ]]; then
  if [[ -f "$RAW_PATIENTS" && -f "$EVIDENCE_DEFS" ]]; then
    echo ">>> Prepared pilot missing; rebuilding from local raw DDXPlus files"
    "$PYTHON" scripts/prepare_ddxplus_debate_pilot.py
  else
    echo "missing ${PILOT_CSV}; add raw DDXPlus files or restore the prepared pilot" >&2
    exit 1
  fi
else
  echo ">>> Using prepared pilot at ${PILOT_CSV}"
fi

echo ">>> Running blind and oracle baselines"
./scripts/run_medical_baselines.sh "$LIMIT" "$BASELINES_DIR" "$FAMILY"

echo ">>> Running debate generation, final judging, analysis, and export"
./scripts/run_medical_debate.sh "$LIMIT" "$EXP_ROOT" "$FAMILY" "$BASELINES_DIR"

echo ">>> Done"
echo ">>> Debate outputs: ${EXP_ROOT}/${FAMILY}/"
echo ">>> Baseline outputs: ${BASELINES_DIR}/"
echo ">>> Aggregated results: ${EXP_ROOT}/medical_results/"
