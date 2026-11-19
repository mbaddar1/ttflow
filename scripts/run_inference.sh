#!/usr/bin/env bash
# Run inference_and_evaluation.py on one VT model artifact with each ODE solver
# (euler, rk4, dopri8), one after another. All console output goes to
# logs/inference_<TIMESTAMP>.log (and is shown live).
#
# Usage (from anywhere):
#   scripts/run_inference.sh <vt-artifact.pkl> [extra args passed to the script...]

set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <vt-artifact-file-path> [extra args for inference_and_evaluation.py]" >&2
    exit 1
fi

VT_ARTIFACT="$1"
shift
EXTRA_ARGS=("$@")

# Repo root = parent dir of this script's folder
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

if [[ ! -f "$VT_ARTIFACT" ]]; then
    echo "VT artifact not found: $VT_ARTIFACT (paths are relative to $REPO_ROOT)" >&2
    exit 1
fi

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
SOLVERS=(euler rk4 dopri8)
mkdir -p logs
LOG_FILE="logs/inference_$(date +%Y-%m-%dT%H-%M-%S).log"

# Send everything below (stdout + stderr) to the console and the log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Model:   $VT_ARTIFACT"
echo "Solvers: ${SOLVERS[*]}"
echo "Log:     $LOG_FILE"

declare -A STATUS
for solver in "${SOLVERS[@]}"; do
    echo
    echo "==================== $solver ===================="
    start=$(date +%s)
    PYTHONPATH="$REPO_ROOT" "$PYTHON" ttflow/inference/inference_and_evaluation.py \
        --vt-artifact-file-path "$VT_ARTIFACT" \
        --ode-solver-method "$solver" \
        "${EXTRA_ARGS[@]}"
    rc=$?
    STATUS[$solver]="exit=$rc time=$(( $(date +%s) - start ))s"
done

echo
echo "==================== summary ===================="
for solver in "${SOLVERS[@]}"; do
    echo "$solver: ${STATUS[$solver]}"
done
