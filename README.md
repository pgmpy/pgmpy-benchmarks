# pgmpy-benchmarks

This repository contains the performance benchmarks for [pgmpy](https://github.com/pgmpy/pgmpy). The benchmarks use
[asv](https://asv.readthedocs.io/) 0.6.6 and `asv_runner` 0.3.2 on Python 3.14. Each night, a GitHub workflow measures
the new commits on the pgmpy `dev` branch. The dashboard is at http://pgmpy.org/pgmpy-benchmarks/.

## Set up a local copy

You do not need a pgmpy checkout. asv clones pgmpy from GitHub into `pgmpy-dev/`. Use pip 25.1 or later.

```bash
python3.14 -m venv .venv
. .venv/bin/activate
pip install --group asv
asv machine --yes
```

**CAUTION:** Do not remove `PYTHONHASHSEED=0` from `asv.conf.json`. pgmpy uses the order of Python sets. Without this
value, the times can change by up to 200 times.

**CAUTION:** Do not change `launch_method` from `forkserver` to `spawn`. With `spawn`, the benchmarks are approximately
5 times slower.

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

**NOTE:** The benchmarks work only on `dev` commits from `43e6fc0e` (pgmpy #3688, 2026-11-06) or newer. They do not
work on pgmpy v1.1.2 or older releases.

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

The workflow starts each day at 03:00 UTC. It runs `asv run NEW`. This command measures each first-parent `dev`
commit that is newer than the newest commit with results. If `dev` has no new commits, the workflow stops.

The `results` branch keeps the measurements, one JSON file for each commit. The workflow adds the new files to
`results`. Then it makes the dashboard from `results` and replaces the `gh-pages` branch. It also does these steps
when a benchmark has an error.

A run fails only when a benchmark has an error. A slower benchmark does not cause a failure. To find slowdowns, look at
the "Regressions" page of the dashboard. Small changes can be noise, because each night uses a different GitHub
computer.

When a run fails, the workflow opens the issue "Nightly benchmarks failed" or adds a comment to it. Usually, a `dev`
commit changed an API that a benchmark uses. Update the benchmark on `main`. Then close the issue.

Each commit takes approximately 5 minutes, and the time limit of the job is 60 minutes. If there are more than 10 new
commits, increase `timeout-minutes` in `.github/workflows/nightly.yml`.

To start a run, use `gh workflow run nightly.yml --repo pgmpy/pgmpy-benchmarks`. GitHub disables the schedule after 60
days without activity in the repository. To enable it again, use
`gh workflow enable nightly.yml --repo pgmpy/pgmpy-benchmarks`.

## Set up a new repository

GitHub Pages must use the branch `gh-pages` and the folder `/ (root)`. These commands make an empty `results` branch:

```bash
tree=$(git hash-object -w -t tree /dev/null)
git push origin "$(git commit-tree "$tree" -m 'Benchmark results store')":refs/heads/results
```

With no results, the workflow measures only the newest `dev` commit.

**CAUTION:** Do not delete the `results` branch. It is the only copy of the measurements.
