import json
import subprocess

import pytest

from ci.compare import main
from tests.conftest import write_result_file

KEY = "x.C.time_a"
PARAMS = [["'fast'"]]
BENCH = {
    KEY: {
        "code": "", "name": KEY, "param_names": ["tier"], "params": PARAMS,
        "type": "time", "unit": "seconds", "version": "1", "timeout": 60,
        "number": 1, "min_run_count": 2, "rounds": 2, "sample_time": 0.01,
        "warmup_time": -1,
    },
    "version": 2,
}  # fmt: skip


def _git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", *args],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip


@pytest.fixture
def env(tmp_path):
    """A real two-commit git repo plus an asv config pointing at it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    shas = []
    for i in range(2):
        (repo / "f").write_text(str(i))
        _git(repo, "add", "f")
        _git(repo, "commit", "-qm", str(i))
        shas.append(_git(repo, "rev-parse", "HEAD"))
    results = tmp_path / "results"
    results.mkdir()
    (results / "benchmarks.json").write_text(json.dumps(BENCH))
    conf = tmp_path / "asv.conf.json"
    conf.write_text(
        json.dumps({"version": 1, "project": "p", "repo": str(repo), "dvcs": "git",
                    "results_dir": str(results), "benchmark_dir": str(tmp_path)})
    )  # fmt: skip
    return results, conf, shas


def _run(env):
    _, conf, (base, head) = env
    return main(["--base", base, "--head", head, "--machine", "m", "--config", str(conf)])


def _write(env, base, head):
    results, _, shas = env
    for sha, date, spec in zip(shas, (1, 2), (base, head)):
        write_result_file(results, "m", sha, date, {KEY: spec})


def ok(center):
    return stats(center, center * 0.95, center * 1.05)


def stats(median, low, high):
    """Statistics without raw samples: what `asv run` saves, so the CI-overlap path is taken."""
    return [[median], PARAMS, {"ci": (low, high)}]


def test_two_times_slower_fails(env, capsys):
    _write(env, ok(0.1), ok(0.2))
    assert _run(env) == 1
    assert "| +" in capsys.readouterr().out


def test_unchanged_passes(env):
    _write(env, ok(0.1), ok(0.1))
    assert _run(env) == 0


def test_faster_passes(env):
    _write(env, ok(0.2), ok(0.1))
    assert _run(env) == 0


def test_errors_on_both_pass(env):
    _write(env, [[None], PARAMS], [[None], PARAMS])
    assert _run(env) == 0


def test_newly_failing_on_head_fails(env):
    _write(env, ok(0.1), [[None], PARAMS])
    assert _run(env) == 1


def test_missing_head_results_fail(env, capsys):
    results, _, shas = env
    write_result_file(results, "m", shas[0], 1, {KEY: ok(0.1)})
    assert _run(env) == 1
    assert f"::error title=No benchmark results::{shas[1]}" in capsys.readouterr().out


# `asv run` saves no samples, so these two tests cover the comparison that runs in CI.
def test_stats_only_disjoint_slowdown_fails(env, capsys):
    _write(env, stats(0.10, 0.09, 0.11), stats(0.20, 0.18, 0.22))
    assert _run(env) == 1
    assert "| +" in capsys.readouterr().out


def test_stats_only_overlapping_slowdown_passes(env):
    # Median ratio 1.6, but head's interval reaches down into base's.
    _write(env, stats(0.10, 0.08, 0.15), stats(0.16, 0.12, 0.30))
    assert _run(env) == 0
