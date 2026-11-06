"""Print the newest commit on the first-parent history of HEAD that has benchmark results.

Ordering follows the history, not the commit dates: asv stores the author date, which is not
monotonic along dev's first-parent chain (rebased and squash-merged commits).
"""

import argparse
import os
import subprocess

from asv import results as asv_results


def latest_commit(results_dir, machine, repo):
    if not os.path.isdir(os.path.join(results_dir, machine)):
        return None
    done = {r.commit_hash for r in asv_results.iter_results_for_machine(results_dir, machine)}
    if not done:
        return None
    history = subprocess.run(
        ["git", "-C", repo, "rev-list", "--first-parent", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.split()  # fmt: skip
    return next((c for c in history if c in done), None)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--machine", required=True)
    p.add_argument("--repo", default="../pgmpy")
    args = p.parse_args(argv)
    print(latest_commit(args.results_dir, args.machine, args.repo) or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
