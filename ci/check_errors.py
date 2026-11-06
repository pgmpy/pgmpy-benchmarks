"""Report benchmarks that errored, as GitHub Actions annotations."""

import argparse
import itertools
import re

from asv import results as asv_results
from asv import util as asv_util


class MissingResults(Exception):
    """No result file for the commit (run crashed, or the hash is wrong/ambiguous)."""

    def __init__(self, commit):
        super().__init__(commit)
        self.commit = commit


def find_errors(results_dir, machine, commit, bench_regex=None):
    """Display names of benchmarks that failed (value None) at `commit`.

    NaN means skipped / not run and is not an error.
    """
    errors = set()
    try:
        found = list(asv_results.iter_results_for_machine_and_hash(results_dir, machine, commit))
    except asv_util.UserError as exc:  # unknown or ambiguous commit prefix
        raise MissingResults(commit) from exc
    if not found:
        raise MissingResults(commit)
    for res in found:
        for key in res.get_all_result_keys():
            params = res.get_result_params(key)
            value = res.get_result_value(key, params)
            # asv always returns one value per param combo (a single one when unparametrized).
            combos = [", ".join(c) for c in itertools.product(*params)]
            names = [f"{key}({c})" for c in combos] if params else [key]
            for name, v in zip(names, value):
                # Same matching as asv's -b filter.
                if v is None and (bench_regex is None or re.search(bench_regex, name)):
                    errors.add(name)
    return errors


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pr", "nightly"], required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--machine", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--base")
    p.add_argument("--bench-regex")
    args = p.parse_args(argv)

    if args.mode == "pr" and not args.base:
        p.error("--base is required in pr mode")

    def errors_or_none(commit):
        try:
            return find_errors(args.results_dir, args.machine, commit, args.bench_regex)
        except MissingResults:
            return None

    head = errors_or_none(args.head)
    if args.mode == "pr":
        base = errors_or_none(args.base)
        # `asv continuous` already fails the job on missing results; just annotate.
        for commit, errs in ((args.base, base), (args.head, head)):
            if errs is None:
                print(f"::warning title=No benchmark results::{commit}")
        if base is not None and head is not None:
            # New errors are already failed by `asv continuous`; only flag old ones.
            for name in sorted(base & head):
                print(f"::warning title=Benchmark errors on base and head::{name}")
        return 0

    if head is None:
        print(f"::error title=No benchmark results for dev::{args.head}")
        return 1
    for name in sorted(head):
        print(f"::error title=Benchmark errors on dev::{name}")
    return 1 if head else 0


if __name__ == "__main__":
    raise SystemExit(main())
