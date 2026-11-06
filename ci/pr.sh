#!/usr/bin/env bash
# Usage: ci/pr.sh <base-sha> <head-sha>
# Env: BENCH_REGEX (default: \('fast')
set -euo pipefail

base=${1:?usage: ci/pr.sh <base-sha> <head-sha>}
head=${2:?usage: ci/pr.sh <base-sha> <head-sha>}
BENCH_REGEX=${BENCH_REGEX:-"\\('fast'"}
machine=github-ubuntu-latest

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 TQDM_DISABLE=1

asv machine --yes --machine "$machine"

hashfile=$(mktemp)
trap 'rm -f "$hashfile"' EXIT
printf '%s\n%s\n' "$base" "$head" > "$hashfile"

# `asv run` exits 2 when some benchmark failed; that is judged by ci.compare below.
run_status=0
asv run --interleave-rounds --show-stderr --machine "$machine" \
    -b "$BENCH_REGEX" "HASHFILE:$hashfile" || run_status=$?

compare_status=0
python -m ci.compare --base "$base" --head "$head" --factor 1.5 \
    --machine "$machine" || compare_status=$?

python -m ci.check_errors --mode pr --results-dir results --machine "$machine" \
    --base "$base" --head "$head" --bench-regex "$BENCH_REGEX"

if [ "$compare_status" -ne 0 ] || { [ "$run_status" -ne 0 ] && [ "$run_status" -ne 2 ]; }; then
    echo "pr.sh failed: asv run status=$run_status, compare status=$compare_status"
    exit 1
fi
