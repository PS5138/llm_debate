#!/usr/bin/env bash
# Create a fresh timestamped results directory under exp/.
#
# Usage:
#   ./scripts/create_results_dir.sh
#   ./scripts/create_results_dir.sh exp

set -euo pipefail

BASE_DIR="${1:-${EXP_BASE:-exp}}"
STAMP="${RESULTS_TIMESTAMP:-$(date +"%Y-%m-%d_%H-%M-%S")}"
RESULTS_DIR="${BASE_DIR}/${STAMP}_results"

suffix=2
while [[ -e "$RESULTS_DIR" ]]; do
  RESULTS_DIR="${BASE_DIR}/${STAMP}_results_${suffix}"
  suffix=$((suffix + 1))
done

mkdir -p "$RESULTS_DIR"
printf '%s\n' "$RESULTS_DIR"
