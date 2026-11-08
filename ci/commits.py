"""Print the newest commit of pgmpy dev and the newest commit in its first-parent history that has results (if any)."""

from asv import results
from asv.config import Config
from asv.repo import get_repo

conf = Config.load()
repo = get_repo(conf)
repo.pull()
history = repo.get_branch_commits(conf.branches[0])
done = set(results.get_existing_hashes(conf.results_dir))
print(history[0], next((c for c in history if c in done), ""))
