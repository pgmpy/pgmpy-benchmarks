# pgmpy-benchmarks

This repository contains the performance benchmarks for [pgmpy](https://github.com/pgmpy/pgmpy). The benchmarks use
[asv](https://asv.readthedocs.io/) 0.6.6 and `asv_runner` 0.3.2. They operate on Python 3.14 only.

Each night, a GitHub workflow (`.github/workflows/nightly.yml`) does these steps:

1. It measures the `dev` branch of pgmpy.
2. It compares the results with an earlier `dev` commit.
3. It keeps the results on the `results` branch.
4. It updates the dashboard on the `gh-pages` branch: http://pgmpy.org/pgmpy-benchmarks/.

## Benchmark settings

Each benchmark class is a subclass of `benchmarks._bench.Bench`. This class sets how asv measures:

- `repeat = (4, 10, 0.25)`
- 2 rounds
- a warm-up before each measurement

The inputs are small: 1,000 data rows (`ROWS`) and graphs with 20 nodes.

## Set up a local copy

You do not need a pgmpy checkout to run asv. asv clones pgmpy from GitHub into `pgmpy-dev/` (git ignores this
directory). Before each run, asv fetches the newest commits.

1. Make a virtual environment and activate it:

   ```bash
   python3.14 -m venv .venv
   . .venv/bin/activate
   ```
2. Install asv and virtualenv from `pyproject.toml`:

   ```bash
   pip install --group asv
   ```

   This command must have pip 25.1 or later.
3. Before you run asv manually, set these environment variables:

   ```bash
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 TQDM_DISABLE=1
   ```
4. Record the information about your computer. asv must have it before the first run.

   ```bash
   asv machine --yes
   ```

**Note:** `PYTHONHASHSEED=0` is the most important variable. pgmpy uses the order of Python sets. Without this
variable, the times can be different by up to 200 times. `ci/nightly.sh` sets all of these variables automatically.

**Note:** asv starts the benchmark processes with the `forkserver` method (`launch_method` in `asv.conf.json`). Do not
change it to `spawn`. With `spawn`, the benchmarks are approximately 5 times slower.

## Run some of the benchmarks

To run only some benchmarks, give a regular expression to asv with `-b`:

```bash
asv run -b TimeFactor
```

## Check the benchmarks

To call each benchmark one time on the newest `dev` commit, run this command:

```bash
asv run --quick --dry-run --show-stderr dev^!
```

Add `-b <regex>` to check only some benchmarks. `--dry-run` tells asv not to keep the results.

## Check a pgmpy pull request

The nightly workflow does not check pull requests. To check a pull request manually, do these steps:

1. Set the environment variables from step 3 of the local setup.
2. Compare the head commit of the pull request with its base commit:

   ```bash
   asv continuous --interleave-rounds --factor 1.5 <base-sha> <head-sha>
   ```

asv fetches the pull request commits from GitHub automatically.

## Run the nightly script locally

The workflow runs `ci/nightly.sh`. You can also run this script on your computer. The script measures the newest
`dev` commit on GitHub.

1. Go to the root of this repository.
2. Make a clone of the `results` branch in a temporary directory.
3. Run the script with that directory:

   ```bash
   NIGHTLY_PUSH=0 ci/nightly.sh <results-clone-dir>
   ```

**CAUTION:** If `NIGHTLY_PUSH` is not `0`, the script pushes the results and the dashboard to GitHub. For a local
test, always set `NIGHTLY_PUSH=0`.

The script uses these environment variables:

| Variable | Function |
|---|---|
| `NIGHTLY_PUSH` | `1` (default): push the results and the dashboard. `0`: commit the results to the local clone only, and make the dashboard with `asv publish`. |
| `BENCH_REGEX` | Run only the benchmarks that match this regular expression. Use a small set for a fast test. |
| `SUMMARY_FILE` | The script writes the table of changed benchmarks to this file. |

## Add a benchmark

Obey these rules:

1. Make one class `Time<PgmpyClass>` for each pgmpy class. Make it a subclass of `Bench`.
2. Make one method `time_<method>` for each method that the pgmpy class defines. Do not measure inherited methods.
3. Use only `time_` and `timeraw_` benchmarks.
4. Some methods do not have a benchmark. Write each of these methods in the `Skipped:` list in the module docstring.
   Give one of these reasons: `deprecated`, `plotting`, `trivial`, `llm`, `torch`.
5. Some pgmpy classes are interchangeable, for example the scores and the CI tests. These classes can share one
   benchmark class with `params`.
6. In `setup`, make the inputs from `benchmarks/_fixtures.py` only.
7. Keep `setup` fast. asv runs `setup` again before each measured call.
8. If the method changes its inputs, `setup` must make new objects each time.
9. Make the inputs the same for each run. Use `rng=0` and `show_progress=False`. Use a fixed rule to select the
   variables and the evidence.
10. Make sure that one call takes less than 0.5 s.
11. Check your benchmarks:

    ```bash
    asv run --quick --dry-run --show-stderr -b <module> dev^!
    ```

Do not add benchmarks for these items:

- torch
- `pgmpy.parameterization`
- deprecated functions and classes
- abstract `Base*` classes
- private functions
- LLM code
- `SEMEstimator`, `FunctionalBayesianNetwork` and `FunctionalCPD`
- plotting

## Fixtures

The benchmark inputs are in the files `benchmarks/data/*.json.gz`. `benchmarks/_fixtures.py` makes the inputs from
these files. It uses only pgmpy constructors and numpy.

Do not use pgmpy to make an input during a benchmark. For example, do not use `simulate`, `forward_sample`,
`get_random`, readers, writers or conversions. The two commits that asv compares must use the same inputs.

To change a fixture, do these steps:

1. Install pgmpy `dev`. The script uses it:

   ```bash
   pip install git+https://github.com/pgmpy/pgmpy.git@dev
   ```
2. Edit `tools/build_fixtures.py`. Add a spec function, and add it to `build_specs`.
3. Make the fixture files again:

   ```bash
   python tools/build_fixtures.py --out benchmarks/data
   ```

   This command must have network access for `load_model`. It also uses a pgmpy checkout (`--pgmpy`, default
   `../pgmpy`). To make only some of the files, add `--only-dbn`, `--only-derived` or `--only-readwrite`.
4. Commit the new fixture files.

## Nightly workflow

The workflow starts each day at 03:00 UTC. You can also start it manually.

The workflow does these steps:

1. It finds the most recent `dev` commit that has results. It looks only at the first-parent history of `dev`.
2. If `dev` did not change after that commit, the workflow stops. It does not update the dashboard.
3. It compares `dev` with that commit. It uses `asv continuous --interleave-rounds --factor 1.5`.
4. It writes the results to the `results` branch.
5. It updates the dashboard on `gh-pages`.

Steps 4 and 5 also occur when the run fails. On the first run there are no earlier results. Then the workflow
measures only `dev`.

### Failures

A run fails for one of these causes:

- A benchmark is more than 1.5 times slower. asv uses a Mann-Whitney test (p < 0.002) on the samples of the two
  commits.
- A benchmark has an error on one of the two commits.

A faster benchmark never causes a failure.

When a run fails, the workflow opens an issue with the title "Nightly benchmarks failed". The issue contains a link
to the run and the table of changed benchmarks. If this issue is open, the workflow adds a comment for each new
failure. Close the issue after you repair the cause.

The workflow reports a slowdown one time only. The next night, it compares `dev` with the slower commit.

### Start the workflow manually

1. Open the Actions tab of this repository.
2. Select **Nightly benchmarks**.
3. Click **Run workflow**.

You can also use the GitHub CLI:

```bash
gh workflow run nightly.yml --repo pgmpy/pgmpy-benchmarks
```

**Note:** The manual run has an optional field, `bench_regex`. Use it only to examine a problem. The run keeps the
results for that `dev` commit, so the full nightly run does not measure that commit again.

### When pgmpy changes an API

When `dev` changes a public API that a benchmark uses, the benchmark has an error. Do these steps:

1. Update the benchmark on the `main` branch of this repository.
2. After the next nightly run, close the "Nightly benchmarks failed" issue.

**Note:** The run that compares the commits before and after the API change always fails. One of the two commits does
not have the API that the benchmark uses.

### Scheduled runs stop after 60 days

GitHub disables a scheduled workflow after 60 days without activity in the repository. To start it again, use the
Actions tab, or run this command:

```bash
gh workflow enable nightly.yml --repo pgmpy/pgmpy-benchmarks
```

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
