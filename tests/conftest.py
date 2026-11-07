import json
import os
from pathlib import Path

import pytest

# Same environment as ci/nightly.sh (PYTHONHASHSEED is checked below).
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TQDM_DISABLE"):
    os.environ.setdefault(_name, "1")


def pytest_configure(config):
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise pytest.UsageError("Run as: PYTHONHASHSEED=0 pytest ...")


COLUMNS = [
    "result", "params", "version", "started_at", "duration",
    "stats_ci_99_a", "stats_ci_99_b", "stats_q_25", "stats_q_75",
    "stats_number", "stats_repeat", "samples",
]  # fmt: skip


def write_result_file(results_dir, machine, commit, date):
    """Write an empty version-2 asv results file (plus machine.json) for `commit`."""
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
        "results": {},
        "durations": {},
        "version": 2,
    }
    (mdir / f"{commit[:8]}-virtualenv-py3.14.json").write_text(json.dumps(data))
