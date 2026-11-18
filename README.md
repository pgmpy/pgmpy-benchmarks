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

1. For each pgmpy class, make one class `Time<PgmpyClass>`. Make it a subclass of `benchmarks._bench.Bench`.
2. For each method that the pgmpy class defines, make one method `time_<method>`. Do not measure inherited methods.
3. Use only `time_` and `timeraw_` benchmarks.
4. Some pgmpy classes have the same API, for example the scores and the CI tests. Measure them in one benchmark class
   with `params`.
5. Write each method that has no benchmark in the `Skipped:` list of the module docstring. Give one of these reasons:
   `deprecated`, `plotting`, `trivial`, `llm`, `torch`, `abstract`, `internal`, `unsupported`, `infeasible`, `alias`.
6. Do not add benchmarks for torch, LLM code, plots, or deprecated or private code. Also do not add benchmarks for
   abstract `Base*` classes, `pgmpy.parameterization`, `SEMEstimator`, `FunctionalBayesianNetwork` or `FunctionalCPD`.
7. In `setup`, make the inputs only from `benchmarks/_fixtures.py`. This file makes all inputs with pgmpy and fixed
   seeds. Its only data file is `benchmarks/data/rocksample.pomdpx`.
8. Keep `setup` fast. asv runs `setup` again before each measured call.
9. If the method changes its inputs, make new objects in `setup` each time.
10. Make the inputs the same for each run. Use `rng=0`, `show_progress=False` and a fixed rule to select the variables
    and the evidence.
11. Make sure that one call takes less than 0.5 s.
12. Run your benchmarks with `-b <module>`. Refer to [Run benchmarks](#run-benchmarks).

**NOTE:** If a pgmpy commit changes a fixture, the two commits that asv compares can get different inputs.

## Nightly workflow

GitHub disables the schedule after 60 days without activity in the repository.

```bash
gh workflow run nightly.yml --repo pgmpy/pgmpy-benchmarks     # Start a run.
gh workflow enable nightly.yml --repo pgmpy/pgmpy-benchmarks  # Enable the schedule again.
```
