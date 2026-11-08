#!/usr/bin/env bash
# Benchmark pgmpy dev against the newest dev commit that has results, save the results and publish the dashboard.
# Usage: ci/nightly.sh, from the repository root, with results/ a checkout of the results branch.
set -euo pipefail

[ -e results/.git ] || { echo "results/ must be a git checkout of the results branch" >&2; exit 1; }
NIGHTLY_PUSH=${NIGHTLY_PUSH:-1}
SUMMARY_FILE=${SUMMARY_FILE:-/dev/null}
machine=github-ubuntu-latest

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 TQDM_DISABLE=1

commits=$(python ci/commits.py)
read -r HEAD PREV <<< "$commits"

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

git -C results add -A
if ! git -C results diff --cached --quiet; then
    git -C results commit -m "Nightly results for ${HEAD:0:8}"
fi

if [ "$NIGHTLY_PUSH" = "1" ]; then
    git -C results push
    asv gh-pages --rewrite
else
    asv publish
fi

exit "$status"
