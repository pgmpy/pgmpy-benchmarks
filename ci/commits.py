"""Print the newest commit of pgmpy dev and the newest commit in its first-parent history that has results (if any)."""

import sys

from asv import results
from asv.config import Config
from asv.repo import get_repo

results_dir, machine = sys.argv[1:]
conf = Config.load()
repo = get_repo(conf)
repo.pull()
history = repo.get_branch_commits(conf.branches[0])
done = {r.commit_hash for r in results.iter_results_for_machine(results_dir, machine)}
print(history[0], next((c for c in history if c in done), ""))
