import pytest

from ci.check_errors import MissingResults, find_errors, main
from tests.conftest import write_result_file

FAST = r"\('fast'"
PARAMS = [["'fast'", "'slow'"]]


def _run(tmp_path, mode, extra=()):
    return main(["--mode", mode, "--results-dir", str(tmp_path), "--machine", "m", *extra])


def test_pr_mode_warns_on_errors_in_both(tmp_path, capsys):
    for sha, date in (("a" * 40, 1), ("b" * 40, 2)):
        write_result_file(tmp_path, "m", sha, date, {"x.C.time_a": [[None, 1.0], PARAMS]})
    assert main(["--mode", "pr", "--results-dir", str(tmp_path), "--machine", "m",
                 "--base", "a" * 40, "--head", "b" * 40, "--bench-regex", FAST]) == 0
    assert "::warning title=Benchmark errors on base and head::x.C.time_a('fast')" in capsys.readouterr().out


def test_pr_mode_silent_on_new_failure(tmp_path, capsys):
    write_result_file(tmp_path, "m", "a" * 40, 1, {"x.C.time_a": [[0.1, 1.0], PARAMS]})
    write_result_file(tmp_path, "m", "b" * 40, 2, {"x.C.time_a": [[None, 1.0], PARAMS]})
    assert _run(tmp_path, "pr", ["--base", "a" * 40, "--head", "b" * 40]) == 0
    assert "::warning" not in capsys.readouterr().out


def test_nightly_mode_fails_on_any_head_error(tmp_path, capsys):
    write_result_file(tmp_path, "m", "b" * 40, 2, {"x.C.time_a": [[None, 1.0], PARAMS]})
    assert _run(tmp_path, "nightly", ["--head", "b" * 40]) == 1
    assert "::error title=Benchmark errors on dev::x.C.time_a('fast')" in capsys.readouterr().out


def test_skipped_params_are_not_errors(tmp_path):
    write_result_file(tmp_path, "m", "b" * 40, 2, {"x.C.time_a": [[0.1, float("nan")], PARAMS]})
    assert _run(tmp_path, "nightly", ["--head", "b" * 40]) == 0


def test_regex_limits_scope(tmp_path, capsys):
    for sha, date in (("a" * 40, 1), ("b" * 40, 2)):
        write_result_file(tmp_path, "m", sha, date, {"x.C.time_a": [[0.1, None], PARAMS]})
    assert _run(tmp_path, "pr", ["--base", "a" * 40, "--head", "b" * 40, "--bench-regex", FAST]) == 0
    assert "::warning" not in capsys.readouterr().out


def test_unparametrized_names_have_no_parens(tmp_path, capsys):
    write_result_file(tmp_path, "m", "b" * 40, 2, {"x.f.time_b": [None, []]})
    assert _run(tmp_path, "nightly", ["--head", "b" * 40]) == 1
    assert "::error title=Benchmark errors on dev::x.f.time_b\n" in capsys.readouterr().out


def test_nightly_mode_fails_when_head_results_missing(tmp_path, capsys):
    assert _run(tmp_path, "nightly", ["--head", "b" * 40]) == 1
    assert "::error title=No benchmark results for dev::" + "b" * 40 in capsys.readouterr().out


def test_pr_mode_warns_when_results_missing(tmp_path, capsys):
    write_result_file(tmp_path, "m", "a" * 40, 1, {"x.C.time_a": [[0.1, 1.0], PARAMS]})
    assert _run(tmp_path, "pr", ["--base", "a" * 40, "--head", "b" * 40]) == 0
    out = capsys.readouterr().out
    assert "::warning title=No benchmark results::" + "b" * 40 in out
    assert "::warning title=No benchmark results::" + "a" * 40 not in out


def test_find_errors_raises_missing_results(tmp_path):
    write_result_file(tmp_path, "m", "a" * 40, 1, {})
    with pytest.raises(MissingResults):
        find_errors(str(tmp_path), "m", "c" * 40, None)
