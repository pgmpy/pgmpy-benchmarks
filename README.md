# pgmpy-benchmarks

This repository contains the performance benchmarks for [pgmpy](https://github.com/pgmpy/pgmpy). The benchmarks use
[asv](https://asv.readthedocs.io/) 0.6.6 and `asv_runner` 0.3.2 on Python 3.14 only. Each night, a GitHub workflow
measures the new commits on the `dev` branch of pgmpy. The dashboard is at
http://pgmpy.org/pgmpy-benchmarks/.

## Set up a local copy

You do not need a pgmpy checkout. asv clones pgmpy from GitHub into `pgmpy-dev/`.

1. Make a virtual environment and activate it:

   ```bash
   python3.14 -m venv .venv
   . .venv/bin/activate
   ```
2. Install asv and virtualenv from `pyproject.toml` with pip 25.1 or later: `pip install --group asv`.
3. Record the information about your computer: `asv machine --yes`.

**Note:** `asv.conf.json` sets `PYTHONHASHSEED=0`, one thread for the math libraries, and `TQDM_DISABLE=1` for the
benchmarks (`env_nobuild`). Do not remove `PYTHONHASHSEED=0`. pgmpy uses the order of Python sets, so without it the
times can change by up to 200 times.

**Note:** Do not change `launch_method` in `asv.conf.json` from `forkserver` to `spawn`. With `spawn`, the benchmarks
are approximately 5 times slower.

## Check benchmarks

To call each benchmark one time on the newest `dev` commit, run this command:

```bash
asv run --quick --dry-run --show-stderr dev^!
```

Add `-b <regex>` to run only the benchmarks that match. `--dry-run` tells asv not to keep the results.

To compare two pgmpy commits, for example for a pull request, run this command:

```bash
asv continuous --interleave-rounds --factor 1.5 --split <base-sha> <head-sha>
```

asv fetches the commits from GitHub.

## Add a benchmark

Obey these rules:

1. Make one class `Time<PgmpyClass>` for each pgmpy class. Make it a subclass of `benchmarks._bench.Bench`.
2. Make one method `time_<method>` for each method that the pgmpy class defines. Do not measure inherited methods.
3. Use only `time_` and `timeraw_` benchmarks.
4. Some methods do not have a benchmark. Write each of these methods in the `Skipped:` list in the module docstring.
   Give one of these reasons: `deprecated`, `plotting`, `trivial`, `llm`, `torch`, `abstract`, `internal`,
   `unsupported`, `infeasible`, `alias`.
5. Do not add benchmarks for torch, `pgmpy.parameterization`, deprecated or private code, abstract `Base*` classes,
   LLM code, plotting, `SEMEstimator`, `FunctionalBayesianNetwork` or `FunctionalCPD`.
6. Some pgmpy classes are interchangeable, for example the scores and the CI tests. These classes can share one
   benchmark class with `params`.
7. In `setup`, make the inputs from `benchmarks/_fixtures.py` only.
8. Keep `setup` fast. asv runs `setup` again before each measured call.
9. If the method changes its inputs, `setup` must make new objects each time.
10. Make the inputs the same for each run. Use `rng=0` and `show_progress=False`. Use a fixed rule to select the
    variables and the evidence.
11. Make sure that one call takes less than 0.5 s.
12. Check your benchmarks with `-b <module>`. Refer to [Check benchmarks](#check-benchmarks).

## Fixtures

`benchmarks/_fixtures.py` makes all inputs with pgmpy and fixed seeds. The only data file is
`benchmarks/data/rocksample.pomdpx`. If a pgmpy change alters a fixture, the two commits that asv compares can get
different inputs.

## Nightly workflow

The workflow starts each day at 03:00 UTC. It runs `asv run NEW`. This command measures each first-parent `dev` commit
that is newer than the newest commit with results. If `dev` has no new commits, the workflow stops. Then the workflow
saves the results on the `results` branch and updates the dashboard on `gh-pages`. It also does this when a benchmark
has an error.

A run fails when a benchmark has an error. A slower benchmark does not cause a failure. To find slowdowns, look at the
"Regressions" page of the dashboard. Each night runs on a different GitHub computer, so small changes can be noise.

When a run fails, the workflow opens one issue, "Nightly benchmarks failed". The issue has a link to the run. While the
issue is open, each new failure adds a comment. Close the issue after you repair the cause.

To start the workflow manually, run `gh workflow run nightly.yml --repo pgmpy/pgmpy-benchmarks`.

When `dev` changes an API that a benchmark uses, the benchmark has an error, and each run fails. Update the benchmark
on the `main` branch of this repository. Then close the issue.

GitHub disables a scheduled workflow after 60 days without activity in the repository. To enable it again, run
`gh workflow enable nightly.yml --repo pgmpy/pgmpy-benchmarks`.

## One-time setup for maintainers

1. Make the `results` branch. It is an empty orphan branch.

   ```bash
   git switch --orphan results
   git commit --allow-empty -m "Benchmark results store"
   git push origin results
   git switch -
   ```
2. In the repository settings, make sure that GitHub Pages uses the branch `gh-pages` and the folder `/ (root)`.

**Note:** The first nightly run replaces all of the `gh-pages` branch. The old dashboard is then deleted.
