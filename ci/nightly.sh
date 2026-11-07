#!/usr/bin/env bash
# Benchmark pgmpy dev (../pgmpy) against the newest dev commit that has results, save the results and publish the dashboard.
# Usage: ci/nightly.sh <results-store-dir> (a checkout of the results branch)
# Env: NIGHTLY_PUSH (1 push, 0 local only), BENCH_REGEX (asv -b), SUMMARY_FILE (changed-benchmarks table).
# Exits with asv's status (1: a >1.5x regression, 2: a benchmark error); results are saved and published either way.
set -euo pipefail

store=${1:?usage: ci/nightly.sh <results-store-dir>}
NIGHTLY_PUSH=${NIGHTLY_PUSH:-1}
SUMMARY_FILE=${SUMMARY_FILE:-/dev/null}
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

args=(--show-stderr --machine "$machine" ${BENCH_REGEX:+-b "$BENCH_REGEX"})

# asv exits 1 when a benchmark got slower and 2 when a benchmark errored.
status=0
: > "$SUMMARY_FILE"
if [ -z "$PREV" ]; then
    asv run "${args[@]}" "$HEAD^!" || status=$?
else
    asv continuous "${args[@]}" --interleave-rounds --factor 1.5 --split "$PREV" "$HEAD" ||
        status=$?
    # `asv continuous` prints no table when a benchmark errored, so build one from the saved results.
    [ "$status" -eq 0 ] || asv compare --machine "$machine" --factor 1.5 --split --only-changed "$PREV" "$HEAD" \
        > "$SUMMARY_FILE" || true
fi

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

exit "$status"
