# pgmpy-benchmarks

Performance benchmarks for [pgmpy](https://github.com/pgmpy/pgmpy), run with
[asv](https://asv.readthedocs.io/) (pinned `asv==0.6.6`, `asv_runner==0.3.2`). Python 3.14 only.
A nightly workflow (`.github/workflows/nightly.yml`) benchmarks pgmpy's `dev`, stores the results on the
`results` branch and rebuilds the dashboard on `gh-pages`.

Each class subclasses `benchmarks._bench.Bench` (`repeat=(4, 10, 0.25)`, 2 rounds, warm-up). Inputs are small
(`ROWS` = 1,000 rows, 20-node graphs). Select benchmarks with asv's `-b` regex, e.g. `asv run -b TimeFactor`.

## Local setup

pgmpy is expected at `../pgmpy` (asv's `repo` in `asv.conf.json`), as in CI.

```bash
python3.14 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt     # asv, virtualenv, pytest
pip install -e ../pgmpy                 # pgmpy itself, for pytest and for the fixtures
```

Export `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 TQDM_DISABLE=1` before running
by hand (`ci/nightly.sh` does). `PYTHONHASHSEED=0` matters most: pgmpy depends on set order, and timings differ by up
to 200x without it.

asv uses `launch_method: forkserver`; `spawn` is about 5x slower.

### Tests

```bash
PYTHONHASHSEED=0 pytest
```

Calls every benchmark once, fails one over 0.5 s (10 s for `timeraw_`), and checks the sample count and the fixtures.

### Reproducing CI

```bash
ci/nightly.sh <results-store-dir>     # pushes unless NIGHTLY_PUSH=0
```

Run it from this repo's root, with the pgmpy commit to benchmark at `../pgmpy`. Env: `BENCH_REGEX`, `NIGHTLY_PUSH`
(`0` commits to the local store and runs `asv publish`) and `SUMMARY_FILE`. For a dry run use
`NIGHTLY_PUSH=0 BENCH_REGEX=<small regex> ci/nightly.sh <a scratch git clone of the results branch>`.

To check a pgmpy PR before it merges: `asv continuous --interleave-rounds --factor 1.5 <base-sha> <head-sha>`.

## Adding a benchmark

1. One `Time<PgmpyClass>` per class, subclassing `Bench`, with one `time_<method>` per method defined on it
   (not inherited). List the rest under `Skipped:` in the module docstring (`deprecated`, `plotting`, `trivial`,
   `llm`, `torch`). Only `time_` and `timeraw_` benchmarks. Interchangeable classes (scores, CI tests) may share a
   class with `params`.
2. `setup` builds inputs from `benchmarks/_fixtures.py` only. It re-runs before every timed call, so it must be cheap
   and return fresh objects if the method mutates.
3. Be deterministic: `rng=0`, `show_progress=False`, a fixed selection rule for variables and evidence.
4. Keep a call under 0.5 s.
5. Out of scope: torch, `pgmpy.parameterization`, deprecated shims, abstract `Base*` classes, private helpers, LLM
   code, `SEMEstimator`, `FunctionalBayesianNetwork`, `FunctionalCPD`, plotting.
6. Run `PYTHONHASHSEED=0 pytest tests/test_suite.py -k <module>`.

## Fixtures

Inputs are frozen in `benchmarks/data/*.json.gz` and rebuilt by `benchmarks/_fixtures.py` with pgmpy constructors and
numpy only. Never derive an input from pgmpy at benchmark time (`simulate`, `forward_sample`, `get_random`,
readers/writers, conversions): base and head must see identical inputs.

To change a fixture, edit `tools/build_fixtures.py` (add a spec function and register it in `build_specs`),
regenerate, add a test in `tests/test_fixtures.py`, and commit the output:

```bash
python tools/build_fixtures.py --out benchmarks/data
```

This needs network access (`load_model`) and a pgmpy checkout (`--pgmpy`, default `../pgmpy`). `--only-dbn`,
`--only-derived` and `--only-readwrite` regenerate just those files.

## CI

**Nightly** (03:00 UTC and on manual dispatch, which takes an optional benchmark regex). `ci/nightly.sh` benchmarks
pgmpy `dev` against the newest commit in `dev`'s first-parent history that has results, with
`asv continuous --interleave-rounds --factor 1.5`. The first time it runs just `dev`; it does nothing if `dev` has not
moved. It writes the results to `results` and rebuilds `gh-pages`, even when the run failed.

- The run fails when a benchmark is more than 1.5x slower (Mann-Whitney p < 0.002 on the commits' samples) or errors
  on either commit. Getting faster never fails.
- On failure the workflow opens an issue, "Nightly benchmarks failed", with the run link and the table of changed
  benchmarks; while it is open, later failures comment on it. Close it when the cause is fixed.
- A regression is reported once: the next night compares against the regressed commit.
- GitHub disables scheduled workflows after 60 days without activity. Re-enable it in the Actions tab or with
  `gh workflow enable nightly.yml`.

When `dev` changes a public API that a benchmark uses, the benchmark errors and the nightly fails. Update the benchmark
on `main`; the night after such a change fails once, so close the issue after it.

## One-time maintainer setup

1. Create the `results` branch (an empty orphan branch):

   ```bash
   git switch --orphan results
   git commit --allow-empty -m "Benchmark results store"
   git push origin results
   ```
2. The first nightly's `asv gh-pages --rewrite` replaces `gh-pages`; to keep the old dashboard, tag it first:
   `git fetch origin gh-pages && git tag old-dashboard origin/gh-pages && git push origin old-dashboard`.
3. In the repository settings, enable GitHub Pages from branch `gh-pages`, folder `/ (root)`.
