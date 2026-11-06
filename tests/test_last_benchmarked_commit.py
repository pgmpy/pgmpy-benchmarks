import subprocess

import pytest

from ci.last_benchmarked_commit import latest_commit
from tests.conftest import write_result_file


def _git(repo, *args, **env):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", *args],
        cwd=repo, check=True, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", **env},
    ).stdout.strip()  # fmt: skip


@pytest.fixture
def repo(tmp_path):
    """A linear history of three commits whose author dates are out of order."""
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    shas = []
    # The middle commit has the latest author date, like a rebased commit on dev.
    for i, date in enumerate(["2020-01-01T00:00:00", "2020-01-03T00:00:00", "2020-01-02T00:00:00"]):
        (path / "f").write_text(str(i))
        _git(path, "add", "f")
        _git(path, "commit", "-qm", str(i), GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
        shas.append(_git(path, "rev-parse", "HEAD"))
    return str(path), shas


def test_none_when_empty(tmp_path, repo):
    assert latest_commit(str(tmp_path), "m", repo[0]) is None


def test_newest_in_history_wins(tmp_path, repo):
    path, (first, second, third) = repo
    write_result_file(tmp_path, "m", first, 1000, {})
    write_result_file(tmp_path, "m", third, 500, {})
    assert latest_commit(str(tmp_path), "m", path) == third


def test_history_order_beats_author_date(tmp_path, repo):
    path, (first, second, third) = repo
    # `second` has the latest author date but `third` is the newer commit in the history.
    write_result_file(tmp_path, "m", second, 3000, {})
    write_result_file(tmp_path, "m", third, 2000, {})
    assert latest_commit(str(tmp_path), "m", path) == third


def test_commits_outside_history_ignored(tmp_path, repo):
    path, (first, *_) = repo
    write_result_file(tmp_path, "m", "f" * 40, 9000, {})
    write_result_file(tmp_path, "m", first, 1000, {})
    assert latest_commit(str(tmp_path), "m", path) == first


def test_other_machines_ignored(tmp_path, repo):
    path, (first, second, _) = repo
    write_result_file(tmp_path, "m", first, 1000, {})
    write_result_file(tmp_path, "other", second, 2000, {})
    assert latest_commit(str(tmp_path), "m", path) == first
