import json
import os
from pathlib import Path

import pytest

# Same environment as ci/pr.sh, set before anything imports numpy. PYTHONHASHSEED cannot be set
# from inside the process, so it is checked instead.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TQDM_DISABLE"):
    os.environ.setdefault(_name, "1")


def pytest_configure(config):
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise pytest.UsageError(
            "The benchmark harness needs a fixed hash seed (several pgmpy algorithms depend on set "
            "order). Run it as: PYTHONHASHSEED=0 pytest ..."
        )


COLUMNS = [
    "result", "params", "version", "started_at", "duration",
    "stats_ci_99_a", "stats_ci_99_b", "stats_q_25", "stats_q_75",
    "stats_number", "stats_repeat", "samples",
]  # fmt: skip


def _row(value):
    values, params, *rest = value
    row = [values, params, "1", 0, 0.0]
    if rest and isinstance(rest[0], dict):
        # One parameter combination, statistics only (what `asv run` saves for a real run):
        # `rest[0]["ci"]` is the (low, high) 99% confidence interval.
        lo, hi = rest[0]["ci"]
        row += [[lo], [hi], [lo], [hi], [1], [8]]
    elif rest:
        # One parameter combination: `rest[0]` is its list of timing samples.
        s = sorted(rest[0])
        row += [[s[0]], [s[-1]], [s[len(s) // 4]], [s[-1 - len(s) // 4]], [1], [len(s)], [rest[0]]]
    return row


def write_result_file(results_dir, machine, commit, date, results):
    """Write a version-2 asv results file (plus machine.json) like a real run does.

    `results` maps benchmark key -> [values, params], [values, params, samples] or
    [values, params, {"ci": (low, high)}] (statistics without samples).
    """
    mdir = Path(results_dir) / machine
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "machine.json").write_text(json.dumps({"machine": machine, "version": 1}))
    data = {
        "commit_hash": commit,
        "env_name": "virtualenv-py3.14",
        "date": date,
        "params": {"machine": machine},
        "python": "3.14",
        "requirements": {},
        "env_vars": {},
        "result_columns": COLUMNS,
        "results": {k: _row(v) for k, v in results.items()},
        "durations": {},
        "version": 2,
    }
    (mdir / f"{commit[:8]}-virtualenv-py3.14.json").write_text(json.dumps(data))
