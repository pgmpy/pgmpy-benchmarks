# pgmpy-benchmarks

This repository contains the performance benchmarks for [pgmpy](https://github.com/pgmpy/pgmpy). The benchmarks use
[asv](https://asv.readthedocs.io/). Each night, a GitHub workflow measures the new commits on the pgmpy `dev` branch.
The dashboard is at http://pgmpy.org/pgmpy-benchmarks/.

This repository has three branches:

- `main` has the benchmarks and the workflow.
- `results` has the measurements, one JSON file for each pgmpy commit.
- `gh-pages` has the dashboard. The workflow makes it from `results`.

## Set up a local copy

You do not need a pgmpy checkout. asv clones pgmpy from GitHub into `pgmpy-dev/`.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --group asv
asv machine --yes
```

## Run benchmarks

This command runs each benchmark one time on the newest `dev` commit. It does not keep the results. To run only some
benchmarks, add `-b <regex>`.

```bash
asv run --quick --dry-run --show-stderr dev^!
```

This command compares two pgmpy commits, for example for a pull request:

```bash
asv continuous --interleave-rounds --factor 1.5 --split <base-sha> <head-sha>
```

**NOTE:** The benchmarks work only on `dev` commits from `43e6fc0e` (pgmpy #3688) or newer.

## Add a benchmark

This example shows the structure of a benchmark module:

```python
"""Benchmarks for ``pgmpy.models``.

Skipped:
    DiscreteBayesianNetwork.states: trivial
"""

from benchmarks._bench import DISCRETE, ROWS, Bench
from benchmarks._fixtures import discrete_bn


class TimeDiscreteBayesianNetwork(Bench):  # One class for each pgmpy class.
    def setup(self):  # asv runs setup again before each measured call. Keep it fast.
        self.model = discrete_bn(DISCRETE)  # A new copy from _fixtures.py, because remove_node changes the model.
        self.node = sorted(self.model.nodes)[0]  # A fixed rule to select the variables.

    def time_get_markov_blanket(self):  # One method for each method that the class defines. Not inherited methods.
        self.model.get_markov_blanket(self.node)

    def time_remove_node(self):
        self.model.remove_node(self.node)

    def time_simulate(self):
        self.model.simulate(n_samples=ROWS, rng=0, show_progress=False)
```

- Make the inputs only with `benchmarks/_fixtures.py`. If the method does not change its inputs, use one shared copy,
  for example `discrete_bn.shared(DISCRETE)`.
- Write each method that has no benchmark in the `Skipped:` list. Give one of these reasons: `deprecated`, `plotting`,
  `trivial`, `llm`, `torch`, `abstract`, `internal`, `unsupported`, `infeasible`, `alias`.
- Do not add benchmarks for torch, LLM code, plots, or deprecated or private code. Also do not add benchmarks for
  abstract `Base*` classes, `pgmpy.parameterization`, `SEMEstimator`, `FunctionalBayesianNetwork` or `FunctionalCPD`.
- Some pgmpy classes have the same API, for example the scores and the CI tests. Measure them in one class with
  `params`. Refer to `benchmarks/structure_score.py`.
- Use only `time_` and `timeraw_` benchmarks. Make sure that one call takes less than 0.5 s.
- Run your benchmarks with `-b <module>`. Refer to [Run benchmarks](#run-benchmarks).

**NOTE:** If a pgmpy commit changes a fixture, the two commits that asv compares can get different inputs.

## Nightly workflow

GitHub disables the schedule after 60 days without activity in the repository.

```bash
gh workflow run nightly.yml --repo pgmpy/pgmpy-benchmarks     # Start a run.
gh workflow enable nightly.yml --repo pgmpy/pgmpy-benchmarks  # Enable the schedule again.
```
