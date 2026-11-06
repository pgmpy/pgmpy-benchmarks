#!/usr/bin/env bash
# Usage: ci/nightly.sh <results-store-dir>
# Env: NIGHTLY_PUSH (default 1), BENCH_REGEX (default empty = all benchmarks)
set -euo pipefail

store=${1:?usage: ci/nightly.sh <results-store-dir>}
NIGHTLY_PUSH=${NIGHTLY_PUSH:-1}
BENCH_REGEX=${BENCH_REGEX:-}
machine=github-ubuntu-latest

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 TQDM_DISABLE=1

mkdir -p results
rsync -a --delete --exclude .git "$store"/ results/

HEAD=$(git -C ../pgmpy rev-parse HEAD)
PREV=$(python -m ci.last_benchmarked_commit --results-dir results --machine "$machine" --repo ../pgmpy)

if [ "$PREV" = "$HEAD" ]; then
    echo "dev unchanged since $HEAD; nothing to benchmark"
    exit 0
fi

asv machine --yes --machine "$machine"

bench_args=()
if [ -n "$BENCH_REGEX" ]; then
    bench_args=(-b "$BENCH_REGEX")
fi

hashfile=$(mktemp)
trap 'rm -f "$hashfile"' EXIT

# `asv run` exits 2 when some benchmark failed; that is judged by the checks below.
run_status=0
compare_status=0
if [ -z "$PREV" ]; then
    asv run --interleave-rounds --show-stderr --machine "$machine" \
        "${bench_args[@]}" "$HEAD^!" || run_status=$?
else
    printf '%s\n%s\n' "$PREV" "$HEAD" > "$hashfile"
    asv run --interleave-rounds --show-stderr --machine "$machine" \
        "${bench_args[@]}" "HASHFILE:$hashfile" || run_status=$?
    python -m ci.compare --base "$PREV" --head "$HEAD" --factor 1.5 \
        --machine "$machine" || compare_status=$?
fi

error_status=0
python -m ci.check_errors --mode nightly --results-dir results \
    --machine "$machine" --head "$HEAD" \
    ${BENCH_REGEX:+--bench-regex "$BENCH_REGEX"} || error_status=$?

# Publish regardless of the statuses above.
rsync -a --delete --exclude .git results/ "$store"/
git -C "$store" add -A
if ! git -C "$store" diff --cached --quiet; then
    git -C "$store" commit -m "Nightly results for ${HEAD:0:8}"
fi

if [ "$NIGHTLY_PUSH" = "1" ]; then
    git -C "$store" push
    asv gh-pages --rewrite
else
    asv publish
fi

if [ "$run_status" -ne 0 ] && [ "$run_status" -ne 2 ]; then
    run_failed=$run_status
else
    run_failed=0
fi
if [ "$run_failed" -ne 0 ] || [ "$compare_status" -ne 0 ] || [ "$error_status" -ne 0 ]; then
    echo "nightly.sh failed: asv run status=$run_status, compare status=$compare_status, check_errors status=$error_status"
    exit 1
fi
