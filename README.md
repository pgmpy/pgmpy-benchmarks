# pgmpy-benchmarks

Performance benchmarks for [pgmpy](https://github.com/pgmpy/pgmpy), run with
[asv](https://asv.readthedocs.io/) (pinned `asv==0.6.6`, `asv_runner==0.3.2`). Python 3.14 only.

The benchmarks live here. The workflows that run them live in pgmpy
(`.github/workflows/benchmarks-pr.yml` and `benchmarks-nightly.yml`) and check out this repo's
`master` next to pgmpy. Results are stored on the `results` branch of this repo, and the dashboard
on `gh-pages` is rebuilt from them.

## Tiers

Every benchmark class subclasses `benchmarks._tiers.Bench` and takes `tier` as its first asv
parameter. There are two tiers (`benchmarks/_tiers.py`):

| tier   | data rows | samples per benchmark                    | used by                  |
|--------|-----------|------------------------------------------|--------------------------|
| `fast` | 1,000     | `repeat=(4, 10, 0.25)`, 2 rounds         | every PR                 |
| `slow` | 100,000   | `repeat=(4, 4, 20.0)`, 2 rounds          | nightly                  |

Both tiers cover the same methods; only the input size (model, rows) differs. Only benchmarks with
no size at all (import time, `list_models`, `list_datasets`) have just `fast`.

Select a tier with asv's `-b` regex, which matches the parametrized names, e.g.
`inference.TimeVariableElimination.time_query('fast')`:

```bash
asv run -b "\('fast'"     # fast tier only (what the PR job runs)
asv run -b "\('slow'"     # slow tier only
asv run -b TimeFactor     # one family, both tiers
```

## Local setup

pgmpy is expected at `../pgmpy` (asv's `repo` in `asv.conf.json`), the same layout CI uses.

```bash
python3.14 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt     # asv, virtualenv, pytest
pip install -e ../pgmpy                 # pgmpy itself, for pytest and for the fixtures
```

Always export these before running anything by hand:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 TQDM_DISABLE=1
```

`PYTHONHASHSEED=0` matters most: several pgmpy algorithms depend on set order, and without a fixed
hash seed the timings of identical code differ by up to about 200x between processes. `ci/pr.sh` and
`ci/nightly.sh` export all five variables; a bare `asv run` or `pytest` does not.

asv uses `launch_method: forkserver`. With `spawn` the PR job took about 1900 s instead of about 390 s.

### Tests

```bash
PYTHONHASHSEED=0 pytest              # fast tier: calls every benchmark once, and checks conventions
PYTHONHASHSEED=0 pytest -m slow      # slow tier (long; see below)
```

`tests/conftest.py` sets the thread-count variables and `TQDM_DISABLE` itself, but `PYTHONHASHSEED`
cannot be set from inside a process, so pytest refuses to start unless it is `0`.

`pytest.ini` deselects `slow` by default. The fast tier fails a benchmark that takes more than 0.5 s
per call (10 s for `timeraw_` benchmarks, which start an interpreter). The slow tier fails above 15 s
per call. The suite also checks that every benchmark yields at least 8 timing samples per commit
(`asv run` saves statistics, not raw samples, so the comparison flags a benchmark only when
head/base is at least 1.5 and the two 99% confidence intervals are disjoint; with 8 samples each
interval is the [min, max] of that commit's samples),
and the fixture tests check that the frozen inputs are deterministic and valid.

### Reproducing CI

```bash
ci/pr.sh <base-sha> <head-sha>        # fast tier on two pgmpy commits, then compare
ci/nightly.sh <results-store-dir>     # see CI below; pushes unless NIGHTLY_PUSH=0
```

Both run from this repo's root. `ci/pr.sh` takes `BENCH_REGEX` (default `\('fast'`). `ci/nightly.sh`
takes `BENCH_REGEX` (default: all benchmarks) and `NIGHTLY_PUSH` (default `1`; with `0` it commits to
the local store, runs `asv publish` and does not push). For a dry run use
`NIGHTLY_PUSH=0 BENCH_REGEX=<small regex> ci/nightly.sh <a scratch git clone of the results branch>`.

Approximate timings on a development machine: the PR job (two commits, fast tier) takes about
6 minutes; the slow tier takes about 27 minutes per commit.

## Adding a benchmark

1. List the public methods defined on the pgmpy class (not inherited ones). Time each one, or list it
   under `Skipped:` in the module docstring with a one-word reason (`deprecated`, `plotting`,
   `trivial`, `llm`, `torch`).
2. One benchmark class per pgmpy class, named `Time<PgmpyClassName>`, subclassing `Bench`, with one
   `time_<method>` per method. Only `time_` and `timeraw_` benchmarks (no `peakmem_`, `mem_`,
   `track_`). Families of interchangeable classes (scores, CI tests) may share a class with a second
   param, e.g. `params = [TIERS, ["K2", "BDeu"]]`.
3. `params[0]` is `TIERS` and `param_names[0] == "tier"` (inherit from `Bench`). Build inputs in
   `prepare(self, tier, *params)`, from `benchmarks/_fixtures.py` only, keyed by module-level dicts
   such as `MODEL = {"fast": "alarm", "slow": "munin"}`. Under asv_runner 0.3.2 `setup` (and so
   `prepare`) re-runs before every timed call, so it must be cheap (the fixture functions are
   memoized) and must hand out fresh objects if the timed method mutates its input. Give a heavy
   one-off input its own `Time<Class><Variant>` class.
4. Make calls deterministic: pass `rng=0` and `show_progress=False` wherever accepted, and pick query
   variables and evidence by a fixed rule (for example the first 4 nodes in sorted order).
5. Respect the budgets: a fast call under 0.5 s; slow calls around 1 s on average, at most 15 s. If a
   method is infeasible at the area's slow fixture, use the largest size that fits and say so in the
   docstring.
6. Out of scope: the torch backend, `pgmpy.parameterization`, deprecated shims, abstract `Base*`
   classes, private helpers, LLM code, `SEMEstimator`, `FunctionalBayesianNetwork`, `FunctionalCPD`,
   plotting. An inherited method is benchmarked once, on the class that defines it, unless a subclass
   overrides it.
7. Run `PYTHONHASHSEED=0 pytest tests/test_suite.py -k <module>`, then
   `PYTHONHASHSEED=0 pytest -m slow tests/test_suite.py -k <module>`.
   Both must pass.
8. Every benchmark module imports the pgmpy names it uses at module level. A newly imported name
   makes those benchmarks error on any PR whose merge-base predates the pgmpy commit that added it
   (asv discovery falls back to the `dev` tip, where the name exists, so only that module's
   benchmarks error, on both base and head). They are reported as warnings, not failures, until the
   PR merges `dev`.

## Fixtures

Inputs are frozen in `benchmarks/data/*.json.gz` and rebuilt by `benchmarks/_fixtures.py` using
only pgmpy constructors and numpy samplers. Never derive an input from pgmpy at benchmark time
(`simulate`, `forward_sample`, `get_random`, readers/writers, model conversions): that would make
the input depend on the commit being measured, and base and head must see identical inputs.

To add or change a fixture, edit `tools/build_fixtures.py` (add a `spec_<name>` function and register
it in `SPECS`), regenerate, add a determinism/validity test in `tests/test_fixtures.py`, and commit
the output:

```bash
python tools/build_fixtures.py --out benchmarks/data
```

This needs network access (it calls `load_model`) and a pgmpy checkout (`--pgmpy`, default
`../pgmpy`) for the PomdpX test document. The offline subsets `--only-dbn`, `--only-derived` and
`--only-readwrite` regenerate just those files.

## CI

**Pull requests** (`benchmarks-pr.yml` in pgmpy). One job, `ci/pr.sh <merge-base> <head>`: runs the
fast tier on both commits with `asv run --interleave-rounds`, then compares them.

- A benchmark that is 1.5x or more slower on head fails the job. Getting faster never fails.
- A benchmark that errors on both commits is only a warning (annotation); one that errors on head
  alone fails.
- No PR comments are posted; a failure just fails the job.

**Nightly** (`benchmarks-nightly.yml` in pgmpy, 03:00 UTC and on manual dispatch). `ci/nightly.sh`
benchmarks both tiers on pgmpy `dev`, compares with the newest commit in `dev`'s first-parent history that has results (or runs just
`dev` the first time; it does nothing if `dev` has not moved), writes results to the `results`
branch, and rebuilds the dashboard on `gh-pages`. Results are published even when it then fails.
The job fails on any regression (1.5x) or any benchmark error. It has a 300-minute limit; the target
is about two hours.

The workflows check out this repo's `master`, so a PR here only takes effect on merge.

### API changes in pgmpy

When a pgmpy PR changes a public API, the PR benchmark job fails on head (the benchmark errors there
but not on base). Update the benchmarks on `master` of this repo first, in a way that works on both
the old and the new API if the base still has to run, and then re-run the PR's benchmark check.

## One-time maintainer setup

Merge order: this branch must be on `master` of this repo before the pgmpy workflows merge, because
the workflows check out `master` and run `ci/pr.sh` and `ci/nightly.sh` from it.

1. Create a fine-grained personal access token with **Contents: read and write** on
   `pgmpy/pgmpy-benchmarks`, and store it as the secret `BENCHMARKS_PUSH_TOKEN` in `pgmpy/pgmpy`.
   Fine-grained tokens expire; when this one does, the nightly fails at checkout. Renew the token
   and update the secret.
2. Create the `results` branch of this repo (an empty orphan branch):

   ```bash
   git switch --orphan results
   git commit --allow-empty -m "Benchmark results store"
   git push origin results
   ```
3. The first nightly's `asv gh-pages --rewrite` force-replaces the existing `gh-pages` branch, which
   holds the old dashboard. If you want to keep that, tag it before the first nightly:

   ```bash
   git fetch origin gh-pages
   git tag old-dashboard origin/gh-pages
   git push origin old-dashboard
   ```
4. In the repository settings, enable GitHub Pages with source "deploy from branch", branch
   `gh-pages`, folder `/ (root)`.
