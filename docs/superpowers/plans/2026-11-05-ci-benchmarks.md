# pgmpy CI Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite pgmpy-benchmarks into a two-tier asv suite covering pgmpy's public API, run the fast tier on every pgmpy PR and the full suite nightly, failing the check on ≥1.5× slowdowns and publishing nightly results to the dashboard.

**Architecture:** Every benchmark class subclasses one `Bench` base whose first asv parameter is `tier` (`"fast"`/`"slow"`). The PR run selects the fast tier with asv's parameter-aware `-b` regex, and the nightly runs both tiers. Inputs come from frozen, pgmpy-independent fixture files plus numpy samplers, so base and head always see identical data. Thin workflow YAML files in pgmpy call `ci/pr.sh` / `ci/nightly.sh` in this repo, which wrap `asv continuous` and two small Python helpers (`ci/check_errors.py`, `ci/last_benchmarked_commit.py`).

**Tech Stack:** asv 0.6.6, asv_runner 0.3.2, Python 3.14, virtualenv, pytest, numpy/pandas (via pgmpy), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-05-ci-benchmarks-design.md` (read it before starting).

## Global Constraints

- Python **3.14** only; runner `ubuntu-latest` only.
- Pin `asv==0.6.6` (runner side) and `asv_runner==0.3.2` (inside the asv env, via `asv.conf.json` matrix).
- Regression threshold `--factor 1.5`; getting faster never fails.
- PR job: one job, **≤ 15 minutes total**, asv run ≤ ~12 minutes. Nightly: ≈ 2 hours total (hard limit 6 h).
- Every benchmark class subclasses `benchmarks._tiers.Bench`; `param_names[0] == "tier"`; `params[0]` is `TIERS` (`["fast", "slow"]`). Only benchmarks with no size at all (import time, `list_models`, `list_datasets`) use `["fast"]`.
- Both tiers cover the same methods; only the input size differs.
- PR selection regex: `\('fast'` (matches asv's parametrized names such as `inference.TimeVariableElimination.time_query('fast')`).
- Inputs must be identical for base and head. Benchmarks never call pgmpy's `simulate`, `forward_sample`, `get_random`, readers/writers or model conversions to *produce inputs*. Inputs come from `benchmarks/_fixtures.py` only.
- Time benchmarks only (`time_` / `timeraw_`); no `peakmem_`, `mem_` or `track_`.
- Every benchmark yields **≥ 8 timing samples per commit** (`rounds × min repeat ≥ 8`). asv only flags a regression when Mann-Whitney p < 0.002, and that needs at least 7 samples on each side. With fewer, its fallback CI check also says "not different", so the regression passes silently (amended 2026-11-05 after Task 4, ruling R10).
- Out of scope: torch backend; `pgmpy.parameterization`; deprecated shims (anything emitting a deprecation warning); abstract `Base*` classes; private helpers; LLM code (`LLMPairwise`, `ExpertInLoop`, `llm_pairwise_orient`); `SEMEstimator`, `FunctionalBayesianNetwork`, `FunctionalCPD`; plotting (`to_daft`, `to_graphviz`).
- Inherited methods are benchmarked once, on the defining class, unless a subclass overrides them. A public method defined on an excluded base class is benchmarked once via one concrete subclass.
- No PR comments, no issues. A failure just fails the job.
- pgmpy core dependencies only (no extras).
- asv machine name: `github-ubuntu-latest`.
- Results live on branch `results` of `pgmpy/pgmpy-benchmarks`; the dashboard on `gh-pages` is rebuilt from it. Push secret in `pgmpy/pgmpy`: `BENCHMARKS_PUSH_TOKEN`.
- Workflows live in pgmpy: `.github/workflows/benchmarks-pr.yml`, `.github/workflows/benchmarks-nightly.yml`; nightly cron `"0 3 * * *"`.
- Directory layout convention: pgmpy is checked out at `../pgmpy` relative to this repo, both locally and in CI.
- Thread env for all benchmark runs: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

## Review Focus

1. **PR whose base branch moved on, or a PR from a fork.** The comparison must use the merge-base and the PR head SHA, both passed to asv as raw SHAs. Expected: works without any branch existing for either commit. Pinned by Task 4, Step 3 (run `ci/pr.sh` on two detached SHAs).
2. **Nightly with `dev` unchanged since last night, and the very first nightly with an empty results store.** Expected: the first exits 0 without benchmarking; the second benchmarks `dev` only and publishes. Pinned by Task 4, Steps 5–6.
3. **A benchmark that errors on both base and head.** Expected: the PR passes with a visible `::warning::` annotation; the nightly fails. Pinned by Task 3 tests `test_pr_mode_warns_on_errors_in_both` / `test_nightly_mode_fails_on_any_head_error`.
4. **A PR that changes pgmpy's sampling or random generators.** Expected: benchmark inputs unchanged. Pinned by Task 2 test `test_discrete_data_does_not_use_pgmpy_sampling`.
5. **Loader benchmarks on a cold or warm Hugging Face cache.** Expected: no network access inside the timed call. Pinned by Task 15's extra step for `datasets.py` (the test asserts `huggingface_hub.constants.HF_HUB_OFFLINE` is true during the timed call).

---

## File Structure

```
asv.conf.json                      rewritten config
.gitignore                         + .asv/ results/ html/ env/ pgmpy/
requirements-ci.txt                asv==0.6.6, virtualenv
requirements-dev.txt               -r requirements-ci.txt, pytest
pytest.ini                         pythonpath=., markers: slow
README.md                          rewritten
benchmarks/__init__.py             (exists)
benchmarks/_tiers.py               TIERS, ROWS, TIMING, Bench
benchmarks/_fixtures.py            spec loading, model builders, numpy samplers
benchmarks/data/*.json.gz          frozen fixture specs (generated once)
benchmarks/data/readwrite/*        frozen model files per format
benchmarks/import_time.py          import pgmpy
benchmarks/base.py                 DAG, PDAG, ADMG, MAG, UndirectedGraph, SimpleCausalModel
benchmarks/factors.py
benchmarks/models.py
benchmarks/inference.py
benchmarks/sampling.py
benchmarks/causal_discovery.py
benchmarks/estimators.py           non-deprecated pgmpy.estimators only
benchmarks/parameter_estimator.py
benchmarks/structure_score.py
benchmarks/ci_tests.py
benchmarks/metrics.py
benchmarks/identification.py
benchmarks/prediction.py
benchmarks/readwrite.py
benchmarks/independencies.py
benchmarks/utils.py
benchmarks/datasets.py             pgmpy.datasets + pgmpy.example_models loaders
tools/build_fixtures.py            one-off generator for benchmarks/data (uses pgmpy + network)
ci/__init__.py
ci/check_errors.py
ci/last_benchmarked_commit.py
ci/pr.sh
ci/nightly.sh
tests/conftest.py                  write_result_file() helper for synthetic asv results
tests/test_suite.py                harness: conventions + runs every benchmark once
tests/test_fixtures.py
tests/test_check_errors.py
tests/test_last_benchmarked_commit.py
```

Deleted from `master`: `benchmarks/{factor,import,inference,parameter_learning,readwrite,sampling,structure_learning}.py`, `results/`, `html/`, `hashfile`.

In pgmpy (separate repo, Task 17): `.github/workflows/benchmarks-pr.yml`, `.github/workflows/benchmarks-nightly.yml`.

## Benchmark Module Recipe (used by Tasks 5–15)

Every benchmark-module task follows this recipe. Each task lists only its classes, fixtures and caveats.

1. **Enumerate methods** for each pgmpy class in the task (methods defined on the class itself, so inherited ones are excluded):
   ```bash
   python - <<'EOF'
   import importlib, inspect
   mod, names = "pgmpy.base", ["DAG", "PDAG"]          # edit per task
   m = importlib.import_module(mod)
   for n in names:
       c = getattr(m, n)
       print(n, sorted(k for k, v in vars(c).items() if not k.startswith("_")
             and (inspect.isfunction(v) or isinstance(v, (classmethod, staticmethod)))))
   EOF
   ```
2. **One benchmark class per pgmpy class**, named `Time<PgmpyClassName>`, subclassing `Bench`, with one `time_<method>` per public method. Families of interchangeable classes (scores, CI tests) may share one benchmark class with a second param naming the pgmpy class, e.g. `params = [TIERS, ["K2", "BDeu"]]`.
3. **Inputs** are built in `prepare(self, tier, *params)` from `_fixtures` only, keyed by module-level dicts such as `MODEL = {"fast": "alarm", "slow": "munin"}`. Under asv_runner 0.3.2, `setup` (and so `prepare`) re-runs before **every** timed call. So `prepare` must be cheap after its first call in a process (the `_fixtures` functions are memoized), and it must hand out fresh objects when the timed method mutates its input.
4. **Skip** anything listed under out-of-scope in Global Constraints. Trivial accessors (a `get_*` that returns a stored attribute) may be skipped. List every skipped public method in the module docstring under `Skipped:` with a one-word reason (`deprecated`, `plotting`, `trivial`, `llm`, `torch`).
5. **Calls are deterministic:** pass `seed=0` / `show_progress=False` wherever the method accepts them, and choose query variables and evidence by fixed rules (e.g. the first 4 nodes in sorted order).
6. **Budgets:** a fast-tier call takes < 0.5 s (enforced by the harness). Slow-tier calls should average ≲ 1 s, with a hard cap of 15 s per call (enforced by the harness). Every benchmark runs ≥ 8 samples on each of two commits, and the nightly must fit about 2 hours. If a method is infeasible at the area's slow fixture (exponential algorithms, kernel tests), use the largest size that fits and say so in the docstring.
7. **Verify:** `pytest tests/test_suite.py -k <module>` (fast tier), then `pytest -m slow tests/test_suite.py -k <module>` (slow tier). Both pass.
8. **Commit** the module alone.

---

### Task 1: Scaffold the new suite (config, tiers, harness, import benchmark)

**Files:**
- Modify: `asv.conf.json` (full rewrite), `.gitignore` (create)
- Create: `requirements-ci.txt`, `requirements-dev.txt`, `pytest.ini`, `benchmarks/_tiers.py`, `benchmarks/import_time.py`, `tests/test_suite.py`
- Delete: old `benchmarks/*.py` (except `__init__.py`), `results/`, `html/`, `hashfile`

**Interfaces:**
- Produces: `benchmarks._tiers.TIERS: list[str] = ["fast", "slow"]`; `ROWS: dict[str, int] = {"fast": 1_000, "slow": 100_000}`; `TIMING: dict[str, dict]`; `class Bench` with `params = [TIERS]`, `param_names = ["tier"]`, `rounds = 2`, `timeout = 600`, `setup(self, tier, *params)` (applies `TIMING[tier]` attributes to `self`, then calls `self.prepare(tier, *params)`), `prepare(self, tier, *params) -> None` (no-op default).
- Produces: `tests/test_suite.py` helpers `benchmark_classes() -> Iterator[tuple[str, type, list[str]]]` and `run_once(cls, method, args, max_seconds)`.

- [ ] **Step 1: Set up a dev env.** `python3.14 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt && pip install -e ../pgmpy`. `requirements-ci.txt` holds `asv==0.6.6` and `virtualenv`; `requirements-dev.txt` holds `-r requirements-ci.txt` and `pytest>=8`. `.gitignore` covers `.venv/ .asv/ results/ html/ env/ pgmpy/ __pycache__/`.

- [ ] **Step 2: Write the harness test** `tests/test_suite.py`:

```python
# discovery: every non-underscore module in benchmarks/, every class defined in it
# (obj.__module__ == module) having attributes starting with "time_" or "timeraw_".

def test_all_classes_follow_conventions():
    classes = list(benchmark_classes())
    assert classes, "no benchmarks discovered"
    for _, cls, _ in classes:
        assert issubclass(cls, Bench), cls
        assert cls.param_names[0] == "tier", cls
        assert cls.params[0] in (TIERS, ["fast"]), cls
        assert not [m for m in dir(cls) if m.startswith(("peakmem_", "mem_", "track_"))], cls

@pytest.mark.parametrize("cls,method,args", cases("fast"))
def test_fast(cls, method, args):
    run_once(cls, method, args, max_seconds=0.5)

@pytest.mark.slow
@pytest.mark.parametrize("cls,method,args", cases("slow"))
def test_slow(cls, method, args):
    run_once(cls, method, args, max_seconds=15)   # amended by R10 (was 60)
```

`cases(tier)` yields `pytest.param(cls, method, (tier, *combo), id="<module>.<Class>.<method>(<tier>, ...)")` for every combo of `cls.params[1:]` when `tier in cls.params[0]`. `run_once` instantiates the class, calls `setup(*args)`, then times one call of the method. For `timeraw_` methods, the timed part is running the returned code string with `subprocess.run([sys.executable, "-c", code], check=True)`. It calls `teardown(*args)` if defined, and asserts:
  - no recorded warning of category `DeprecationWarning`/`FutureWarning` whose message contains `"deprecat"` (case-insensitive), using `warnings.catch_warnings(record=True)` + `simplefilter("always")` around setup and call;
  - `elapsed < max_seconds`.

`pytest.ini`: `pythonpath = .`, `markers = slow: slow-tier benchmarks`, `addopts = -m "not slow"`. With this, `-m slow` overrides the default selection.

- [ ] **Step 3: Run it to see it fail.** `pytest tests/test_suite.py -q`. Expected: FAIL in `test_all_classes_follow_conventions` ("no benchmarks discovered") or an ImportError for `benchmarks._tiers`.

- [ ] **Step 4: Implement `benchmarks/_tiers.py` and `benchmarks/import_time.py`.** Initial `TIMING = {"fast": {"repeat": (3, 10, 0.25), "warmup_time": 0.05}, "slow": {"repeat": (1, 3, 30.0), "warmup_time": 0.0}}` (tuned in Task 16). R10 amends this to `{"fast": {"repeat": (4, 10, 0.25), "warmup_time": 0.05}, "slow": {"repeat": (4, 4, 20.0), "warmup_time": 0.0}}` with `rounds = 2`, giving ≥ 8 samples per commit. `import_time.TimeImport(Bench)` has `params = [["fast"]]` and `timeraw_import_pgmpy(self, tier) -> str` returning `"import pgmpy"`. Delete the old benchmark modules, `results/`, `html/`, `hashfile` with `git rm -r`.

- [ ] **Step 5: Rewrite `asv.conf.json`** with exactly these keys: `"version": 1`, `"project": "pgmpy"`, `"project_url": "https://pgmpy.org/"`, `"repo": "../pgmpy"`, `"dvcs": "git"`, `"branches": ["dev"]`, `"environment_type": "virtualenv"`, `"pythons": ["3.14"]`, `"matrix": {"req": {"pip+asv_runner": ["0.3.2"]}}`, `"benchmark_dir": "benchmarks"`, `"env_dir": ".asv/env"`, `"results_dir": "results"`, `"html_dir": "html"`. No `build_command` (asv's default `python -m build --wheel` handles pgmpy's setuptools backend).

- [ ] **Step 6: Run tests and asv.**
  - `pytest tests/test_suite.py -q`: expected PASS (2 tests: conventions + `import_time.TimeImport.timeraw_import_pgmpy('fast')`).
  - `asv machine --yes --machine local-dev && asv run --quick --show-stderr "$(git -C ../pgmpy rev-parse dev)^!"`: expected one benchmark, no failure.
  - `ls .asv/env/*/lib/python3.14/site-packages | grep asv_runner`: expected `asv_runner-0.3.2.dist-info`. If the matrix pin doesn't take, try `"req": {"asv_runner": ["0.3.2"]}` and record which form worked in a comment in the config.

- [ ] **Step 7: Commit.** `git add -A && git commit -m "Scaffold two-tier asv suite and harness"`

---

### Task 2: Frozen fixtures and numpy samplers

**Files:**
- Create: `tools/build_fixtures.py`, `benchmarks/_fixtures.py`, `benchmarks/data/*.json.gz`, `tests/test_fixtures.py`

**Interfaces:**
- Produces, in `benchmarks/_fixtures.py`. All are memoized where noted; builders return fresh objects on every call.
  - `load_spec(name: str) -> dict` (memoized; reads `benchmarks/data/<name>.json.gz`)
  - `discrete_bn(name: str) -> DiscreteBayesianNetwork`
  - `gaussian_bn(name: str) -> LinearGaussianBayesianNetwork`
  - `graph(name: str) -> DAG | PDAG | ADMG | MAG | UndirectedGraph` (type from spec `"kind"`)
  - `markov_network(name: str) -> DiscreteMarkovNetwork` (one factor per CPD family; edges = moral graph computed in Python from spec parents)
  - `factor_graph(name: str) -> FactorGraph`, `junction_tree(name: str) -> JunctionTree`
  - `random_factor(n_vars: int, card: int, seed: int = 0) -> DiscreteFactor` (numpy values)
  - `discrete_data(name: str, n_rows: int, seed: int = 0) -> pd.DataFrame`: returns `.copy()` of a frame built from the memoized `_discrete_array(name, n_rows, seed) -> np.ndarray` (an `lru_cache`; the tests call its `cache_clear()`)
  - `gaussian_data(name: str, n_rows: int, seed: int = 0) -> pd.DataFrame` (same)
  - `mixed_data(name: str, n_rows: int, seed: int = 0) -> pd.DataFrame`: `gaussian_data` with every odd-indexed column binned by tertiles into `"low"/"mid"/"high"`.
- Spec file format. **Nodes are stored in topological order.**
  - discrete: `{"kind": "discrete", "nodes": [{"name", "states": [...], "parents": [...], "values": [[...]]}]}`. `values` = `cpd.get_values().tolist()`, shape `(card, prod(parent cards))`, parents in CPD evidence order, last parent varying fastest.
  - gaussian: `{"kind": "gaussian", "nodes": [{"name", "parents": [...], "intercept", "coefficients": [...], "std"}]}`
  - graphs: `{"kind": "dag"|"pdag"|"admg"|"mag"|"ug", "nodes": [...], "directed": [[u, v]], "undirected": [[u, v]], "bidirected": [[u, v]], "latents": [...]}` (unused lists empty)
  - junction tree: `{"kind": "junction_tree", "cliques": [[...]], "edges": [[i, j]], "factors": [{"scope", "cardinality", "values", "state_names"}]}`
- Fixture names, generated by `tools/build_fixtures.py` with fixed seeds:
  - discrete: `asia`, `alarm`, `andes`, `munin`, `pathfinder` (from `load_model("bnlearn/<name>")`); `random_discrete_200` (`DiscreteBayesianNetwork.get_random(n_nodes=200, edge_prob=0.02, n_states=3, seed=0)`)
  - gaussian: `random_gaussian_8` (`LinearGaussianBayesianNetwork.get_random(n_nodes=8, edge_prob=0.3, seed=0)`), `ecoli70`, `arth150` (from `bnlearn/`)
  - graphs: `dag_20` (`DAG.get_random(n_nodes=20, edge_prob=0.15, seed=0)`), `dag_500` (`n_nodes=500, edge_prob=0.005, seed=0`), and `pdag_`/`admg_`/`mag_`/`ug_` at 20 and 500, derived from the DAGs at build time with pgmpy's own conversions, then frozen
  - junction trees: `jt_alarm`, `jt_andes`

  Later tasks may add specs (DBN, Markov chain, readwrite files, independencies) by extending `tools/build_fixtures.py`. They must follow the same rule: generated once, committed, loaded via `_fixtures`.

- [ ] **Step 1: Write failing tests** in `tests/test_fixtures.py`:

```python
def test_discrete_data_is_deterministic():
    pd.testing.assert_frame_equal(discrete_data("asia", 500), discrete_data("asia", 500))

def test_discrete_data_does_not_use_pgmpy_sampling(monkeypatch):
    def boom(*a, **k): raise AssertionError("pgmpy sampling used")
    monkeypatch.setattr(DiscreteBayesianNetwork, "simulate", boom)
    monkeypatch.setattr(BayesianModelSampling, "forward_sample", boom)
    _fixtures._discrete_array.cache_clear()          # name of the memoized inner fn
    assert discrete_data("asia", 100, seed=1).shape == (100, 8)

def test_discrete_sampler_matches_exact_marginals():
    data = discrete_data("asia", 100_000)
    infer = VariableElimination(discrete_bn("asia"))
    for var in data.columns:
        q = infer.query([var], show_progress=False)
        for state in q.state_names[var]:
            assert abs((data[var] == state).mean() - q.get_value(**{var: state})) < 0.01

def test_gaussian_sampler_recovers_parameters():
    spec = load_spec("random_gaussian_8"); data = gaussian_data("random_gaussian_8", 100_000)
    for node in spec["nodes"]:
        X = np.column_stack([np.ones(len(data))] + [data[p] for p in node["parents"]])
        beta, *_ = np.linalg.lstsq(X, data[node["name"]], rcond=None)
        np.testing.assert_allclose(beta, [node["intercept"], *node["coefficients"]], atol=0.05)

def test_models_are_valid_and_sized():
    assert discrete_bn("alarm").check_model()
    assert (len(discrete_bn("alarm").nodes()), len(discrete_bn("alarm").edges())) == (37, 46)
    assert len(discrete_bn("munin").nodes()) == 1041
    assert gaussian_bn("arth150").check_model()
    assert len(graph("dag_500").nodes()) == 500

def test_builders_return_fresh_objects():
    assert discrete_bn("asia") is not discrete_bn("asia")
    df = discrete_data("asia", 10); df.iloc[0, 0] = "MUTATED"
    assert discrete_data("asia", 10).iloc[0, 0] != "MUTATED"
```

- [ ] **Step 2: Run to see them fail.** `pytest tests/test_fixtures.py -q`. Expected: ImportError for `benchmarks._fixtures`.

- [ ] **Step 3: Implement `tools/build_fixtures.py`.** CLI: `python tools/build_fixtures.py --out benchmarks/data`. It writes every fixture listed above as gzip-compressed JSON (`gzip.open(..., "wt")`, `json.dump(..., separators=(",", ":"))`). Run it once (needs network for `load_model`).

- [ ] **Step 4: Implement `benchmarks/_fixtures.py`** with the interfaces above, using pgmpy constructors only: `TabularCPD(variable, variable_card, values, evidence, evidence_card, state_names)` and `LinearGaussianCPD(variable, beta=[intercept, *coefficients], std, evidence)`.
  - **Discrete sampler:** for each node in spec order, compute column indices with `np.ravel_multi_index(parent_state_indices, parent_cards)`, take `probs = values[:, idx].T`, then `state_idx = (np.cumsum(probs, axis=1) < rng.random(n)[:, None]).sum(axis=1)`, and map to `states` via an object array.
  - **Gaussian sampler:** `x = intercept + parents @ coefficients + rng.normal(0, std, n)`.
  - **RNG:** `np.random.default_rng(seed)`.

- [ ] **Step 5: Run tests.** `pytest tests/test_fixtures.py -q`. Expected: PASS. Then check `du -sh benchmarks/data`. Expected: under ~10 MB; if larger, report it before committing.

- [ ] **Step 6: Commit.** `git add tools benchmarks/_fixtures.py benchmarks/data tests/test_fixtures.py && git commit -m "Add frozen fixtures and numpy samplers"`

---

### Task 3: CI result helpers

**Files:**
- Create: `ci/__init__.py`, `ci/check_errors.py`, `ci/last_benchmarked_commit.py`, `tests/conftest.py`, `tests/test_check_errors.py`, `tests/test_last_benchmarked_commit.py`

**Interfaces:**
- `ci.check_errors.find_errors(results_dir: str, machine: str, commit: str, bench_regex: str | None) -> set[str]`: display names `"<key>(<param reprs joined by ', '>)"` (or bare `<key>` when unparametrized) whose value is `None` (failed). NaN means skipped, not an error. The regex is matched with `re.search` against the display name, exactly as asv's `-b` does. Uses `asv.results.iter_results_for_machine_and_hash`, `Results.get_all_result_keys`, `get_result_params`, `get_result_value`.
- `ci.check_errors.main(argv: list[str] | None = None) -> int`. CLI: `python -m ci.check_errors --mode {pr,nightly} --results-dir DIR --machine NAME --head SHA [--base SHA] [--bench-regex RE]`.
  - **pr mode** (`--base` required): prints `::warning title=Benchmark errors on base and head::<name>` for each name in `find_errors(base) & find_errors(head)` and returns 0. New errors are already failed by `asv continuous`.
  - **nightly mode:** prints `::error title=Benchmark errors on dev::<name>` for each `find_errors(head)` name and returns 1 if any, else 0.
- `ci.last_benchmarked_commit.latest_commit(results_dir: str, machine: str) -> str | None`: the `commit_hash` of the result file with the greatest `date`, or `None` when there are none or the machine dir is missing. CLI `python -m ci.last_benchmarked_commit --results-dir DIR --machine NAME` prints the hash, or an empty line.
- `tests/conftest.py`: `write_result_file(results_dir, machine, commit, date, results: dict[str, list])` writes a version-2 asv results file `<machine>/<commit[:8]>-virtualenv-py3.14.json` plus `<machine>/machine.json`. In the `results` dict, each value is `[values, params]`, e.g. `{"m.C.time_a": [[0.1, None], [["'fast'", "'slow'"]]]}`. If unsure of the column layout, copy it from `git show 3eff2100:results/7950x/fffba3b2-conda-py3.11.json` (a real asv 0.6 results file from before Task 1 deleted them).

- [ ] **Step 1: Write failing tests.**

```python
# tests/test_check_errors.py
FAST = r"\('fast'"
def test_pr_mode_warns_on_errors_in_both(tmp_path, capsys):
    for sha, date in (("a" * 40, 1), ("b" * 40, 2)):
        write_result_file(tmp_path, "m", sha, date, {"x.C.time_a": [[None, 1.0], [["'fast'", "'slow'"]]]})
    assert main(["--mode", "pr", "--results-dir", str(tmp_path), "--machine", "m",
                 "--base", "a" * 40, "--head", "b" * 40, "--bench-regex", FAST]) == 0
    assert "::warning title=Benchmark errors on base and head::x.C.time_a('fast')" in capsys.readouterr().out

def test_pr_mode_silent_on_new_failure(tmp_path, capsys): ...   # base 0.1, head None -> 0, no "::warning"
def test_nightly_mode_fails_on_any_head_error(tmp_path, capsys): ...  # head None -> 1, "::error" line
def test_skipped_params_are_not_errors(tmp_path): ...           # head [0.1, nan] -> nightly returns 0
def test_regex_limits_scope(tmp_path, capsys): ...              # slow-only failure + FAST regex -> pr: no warning
def test_unparametrized_names_have_no_parens(tmp_path, capsys): ...  # "x.f.time_b" -> "::error ...::x.f.time_b"

# tests/test_last_benchmarked_commit.py
def test_none_when_empty(tmp_path): assert latest_commit(str(tmp_path), "m") is None
def test_latest_date_wins(tmp_path):
    write_result_file(tmp_path, "m", "a" * 40, 2000, {}); write_result_file(tmp_path, "m", "b" * 40, 1000, {})
    assert latest_commit(str(tmp_path), "m") == "a" * 40
def test_other_machines_ignored(tmp_path): ...
```

- [ ] **Step 2: Run to see them fail.** `pytest tests/test_check_errors.py tests/test_last_benchmarked_commit.py -q`. Expected: ImportError.
- [ ] **Step 3: Implement both modules** per the interfaces.
- [ ] **Step 4: Run tests.** Same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git add ci tests && git commit -m "Add CI result helpers"`

---

### Task 4: CI driver scripts

**Files:**
- Create: `ci/pr.sh`, `ci/nightly.sh` (both `chmod +x`, `set -euo pipefail`, run from the repo root)

**Interfaces:**
- `ci/pr.sh <base-sha> <head-sha>`, with env `BENCH_REGEX` (default `\('fast'`). Steps:
  1. Export the thread env vars.
  2. `asv machine --yes --machine github-ubuntu-latest`.
  3. `asv continuous --factor 1.5 --interleave-rounds --show-stderr --machine github-ubuntu-latest -b "$BENCH_REGEX" <base> <head>`, capturing its exit status without aborting.
  4. `python -m ci.check_errors --mode pr ... --base <base> --head <head> --bench-regex "$BENCH_REGEX"`.
  5. Exit with asv's status.

  Raw SHAs are passed straight to asv (asv uses a local `repo` path directly, so any commit object in `../pgmpy` resolves).
- `ci/nightly.sh <results-store-dir>`, with env `NIGHTLY_PUSH` (default `1`) and `BENCH_REGEX` (default empty, meaning all). Steps:
  1. `rsync -a --delete --exclude .git <store>/ results/`.
  2. `HEAD=$(git -C ../pgmpy rev-parse HEAD)`, `PREV=$(python -m ci.last_benchmarked_commit ...)`.
  3. If `PREV == HEAD`: echo `dev unchanged since <sha>; nothing to benchmark` and exit 0.
  4. `asv machine --yes --machine github-ubuntu-latest`.
  5. If `PREV` is empty: `asv run --interleave-rounds --show-stderr --machine github-ubuntu-latest [-b ...] "$HEAD^!"`. Otherwise: `asv continuous --factor 1.5 --interleave-rounds --show-stderr --machine github-ubuntu-latest [-b ...] "$PREV" "$HEAD"`. Capture `bench_status`.
  6. `python -m ci.check_errors --mode nightly --head "$HEAD" ...` and capture `error_status`.
  7. **Publish regardless of the statuses:** `rsync -a --delete --exclude .git results/ <store>/`, then `git -C <store> add -A && git -C <store> commit -m "Nightly results for ${HEAD:0:8}"`, skipping the commit when nothing is staged. When `NIGHTLY_PUSH=1`: `git -C <store> push` and `asv gh-pages --rewrite`. Otherwise run `asv publish` only.
  8. Exit 1 if either status is non-zero.

- [ ] **Step 1: Write both scripts** per the interfaces.
- [ ] **Step 2: ShellCheck.** `shellcheck ci/*.sh` (install with `apt`/`pip install shellcheck-py` if missing). Expected: no errors.
- [ ] **Step 3: PR script on detached SHAs (Review Focus 1).** `ci/pr.sh "$(git -C ../pgmpy rev-parse origin/dev~3)" "$(git -C ../pgmpy rev-parse origin/dev)"`. Expected: asv builds both commits, runs `import_time...('fast')`, and exits 0. The log shows the comparison table.
- [ ] **Step 4: PR script fails on a regression.**
  1. Create a scratch worktree: `git -C ../pgmpy worktree add <tmpdir>/pgmpy-slow -b bench-scratch origin/dev`. Never touch the user's current pgmpy branch.
  2. In that worktree, commit one change: `import time; time.sleep(0.5)` at the top of `pgmpy/__init__.py`.
  3. Run `ci/pr.sh "$(git -C ../pgmpy rev-parse origin/dev)" "$(git -C ../pgmpy rev-parse bench-scratch)"`. Worktrees share the object store, so asv resolves the SHA. Expected: exit status 1 and "PERFORMANCE DECREASED" in the log.
  4. Remove the worktree and delete the `bench-scratch` branch.
- [ ] **Step 5: Nightly first run (Review Focus 2).** `STORE=$(mktemp -d) && git -C "$STORE" init && NIGHTLY_PUSH=0 ci/nightly.sh "$STORE"`. Expected: runs `asv run` on HEAD only, commits results into the store, `asv publish` writes `html/index.html`, exit 0.
- [ ] **Step 6: Nightly unchanged dev.** Run `NIGHTLY_PUSH=0 ci/nightly.sh "$STORE"` again. Expected: prints `dev unchanged since ...; nothing to benchmark`, exit 0.
- [ ] **Step 7: Commit.** `git add ci/*.sh && git commit -m "Add PR and nightly CI driver scripts"`

---

### Task 5: `benchmarks/base.py` (graphs)

Recipe applies. Classes: `DAG`, `PDAG`, `ADMG`, `MAG`, `UndirectedGraph`, `SimpleCausalModel` from `pgmpy.base`.
Fixtures: `graph("<kind>_20")` (fast) and `graph("<kind>_500")` (slow) per class. For `SimpleCausalModel`, read its constructor and add a frozen spec in `tools/build_fixtures.py` if it needs more than a graph. Exclude `to_daft` and `to_graphviz`. Methods that take node arguments (d-separation, Markov blanket, adjustment sets) use the first and last nodes in sorted order.
Commit message: `"Add graph benchmarks"`.

### Task 6: `benchmarks/factors.py`

Recipe applies. Classes: `DiscreteFactor`, `TabularCPD`, `NoisyORCPD`, `JointProbabilityDistribution`, `LinearGaussianCPD`, `FactorSet`, `FactorDict`, and functions `factor_product`, `factor_divide`, `factor_sum_product`, `factorset_product`, `factorset_divide` (one `TimeFactorFunctions` class).
Fixtures:
- `DiscreteFactor`: `random_factor(4, 3)` (fast) / `random_factor(12, 3)` (slow).
- `TabularCPD`: the CPD with the most parameters in `discrete_bn("alarm")` / `discrete_bn("pathfinder")`.
- `JointProbabilityDistribution`: 4 / 16 binary vars from numpy.
- `NoisyORCPD`: 4 / 16 parents.
- `LinearGaussianCPD`: 3 / 200 evidence.
- `FactorSet`/`FactorDict`: the CPD factors of `alarm` / `munin`.

`get_random` methods are benchmarked as methods (they are public API). They just aren't used to produce inputs.
Commit message: `"Add factor benchmarks"`.

### Task 7: `benchmarks/models.py`, part A (discrete and undirected models)

Recipe applies. Classes: `DiscreteBayesianNetwork`, `DiscreteMarkovNetwork`, `FactorGraph`, `JunctionTree`, `ClusterGraph`.
Fixtures:
- `DiscreteBayesianNetwork`: `alarm` (fast) / `munin`, `andes` or `random_discrete_200` (slow; pick per method within budget). For `fit`/`predict`, use `discrete_data(<model>, ROWS[tier])`.
- Markov network and factor graph: `markov_network`/`factor_graph` of `alarm` / `andes`.
- `JunctionTree`/`ClusterGraph`: `jt_alarm` / `jt_andes`.

`DiscreteBayesianNetwork` methods that are inherited from `DAG` are not repeated (Task 5 owns them).
Commit message: `"Add discrete model benchmarks"`.

### Task 8: `benchmarks/models.py`, part B (other models)

Recipe applies; append to `benchmarks/models.py`. Classes: `LinearGaussianBayesianNetwork`, `DynamicBayesianNetwork`, `MarkovChain`, `NaiveBayes`, `SEMGraph`, `SEMAlg`, `SEM`.
Fixtures:
- `LinearGaussianBayesianNetwork`: `random_gaussian_8` (fast) / `arth150` (slow), with `gaussian_data`.
- `DynamicBayesianNetwork`: add frozen specs `dbn_5` / `dbn_50` (variables per slice, intra- and inter-slice edges, numpy CPDs) to `tools/build_fixtures.py`, a `dbn(name)` builder, and a numpy unrolled sampler `dbn_data(name, n_rows, seed=0)` to `_fixtures`, with a determinism test in `tests/test_fixtures.py`.
- `MarkovChain`: 3 vars × 3 states / 20 vars × 5 states, with numpy transition matrices.
- `NaiveBayes`: `discrete_data("asia")` / `discrete_data("alarm")`, using the first column as the class.
- `SEM*`: graphs from `random_gaussian_8` / `arth150` edges, with `gaussian_data`.

Commit message: `"Add remaining model benchmarks"`.

### Task 9: `benchmarks/inference.py`

Recipe applies. Classes: `VariableElimination`, `BeliefPropagation`, `BeliefPropagationWithMessagePassing`, `ApproxInference`, `CausalInference`, `DBNInference`, `Mplp`.
Fixtures:
- `VariableElimination`, `BeliefPropagation*`, `ApproxInference`: `alarm` (fast) / `munin` or `pathfinder` (slow). `munin` calibration may be infeasible, so fall back to `pathfinder` or `andes` and note it.
- `CausalInference`: `alarm` / `andes`.
- `DBNInference`: `dbn_5` / `dbn_50`.
- `Mplp`: `markov_network("alarm")` / `markov_network("andes")`.

Queries: the 4 lexicographically first non-root nodes; evidence is the 2 lexicographically last nodes at their first state. Skip methods that emit deprecation warnings (the harness enforces this).
Commit message: `"Add inference benchmarks"`.

### Task 10: `benchmarks/sampling.py`

Recipe applies. Classes: `BayesianModelSampling`, `GibbsSampling`.
Fixtures: `alarm` with `ROWS[tier]` samples. In the slow tier, use `andes` if `alarm` at 100k is under 1 s. Always pass `seed=0`. Rejection and likelihood-weighted sampling use the inference evidence rule from Task 9.
Commit message: `"Add sampling benchmarks"`.

### Task 11: `benchmarks/causal_discovery.py`

Recipe applies. Classes: `PC`, `GES`, `HillClimbSearch`, `ChowLiu`, `TAN`, `SP`, `ANM`, `IGCI`, `R2Sort`, `VarSort`, `TOPIC`, `BootstrapEstimator`, `ExpertKnowledge` from `pgmpy.causal_discovery`.
Fixtures:
- Discrete algorithms: `discrete_data("asia", 1_000)` (fast) / `discrete_data("alarm", 100_000)` (slow).
- Continuous algorithms (`ANM`, `IGCI`, `R2Sort`, `VarSort`, `TOPIC`, and `PC`/`GES` with Gaussian scores if the API offers it): `gaussian_data("random_gaussian_8", 1_000)` / `gaussian_data("ecoli70", 100_000)`.
- `TAN`'s class variable: the first column.

Pairwise kernel methods (`ANM`) may cap the slow rows; record it in the docstring.
Commit message: `"Add causal discovery benchmarks"`.

### Task 12: `benchmarks/estimators.py` (non-deprecated `pgmpy.estimators`)

Recipe applies. Classes: `MmhcEstimator`, `ExhaustiveSearch`, `MirrorDescentEstimator`, `IVEstimator`. First confirm each emits no deprecation warning on construction. If one does, drop it and list it as `deprecated`.
Fixtures:
- `MmhcEstimator`: as discrete causal discovery in Task 11.
- `ExhaustiveSearch`: `discrete_data` over the first 4 (fast) / 5 (slow) columns of `asia`; it is exponential.
- `MirrorDescentEstimator`: `asia` / `alarm` model with data.
- `IVEstimator`: an `SEM` from the `random_gaussian_8` / `arth150` structure with `gaussian_data`.

Commit message: `"Add estimator benchmarks"`.

### Task 13: `benchmarks/parameter_estimator.py`

Recipe applies. Classes: `DiscreteMLE`, `DiscreteBayesianEstimator`, `DiscreteEM`, `LinearGaussianMLE`.
Fixtures:
- Discrete estimators: model `asia` with `discrete_data("asia", 1_000)` (fast) / `alarm` with 100k rows (slow).
- `DiscreteEM`: the same model with one fixed variable (lexicographically first non-root node) treated as latent and dropped from the data.
- `LinearGaussianMLE`: `random_gaussian_8` (1k rows) / `arth150` (100k rows).

Commit message: `"Add parameter estimation benchmarks"`.

### Task 14: `benchmarks/structure_score.py` and `benchmarks/ci_tests.py`

Recipe applies. Use one shared class per family, with a second param naming the pgmpy class.

`structure_score.py`:
- `TimeDiscreteScore`: params `[TIERS, ["K2", "BDeu", "BDs", "LogLikelihood", "AIC", "BIC"]]`, on `discrete_data`.
- `TimeGaussianScore`: params `[TIERS, ["LogLikelihoodGauss", "AICGauss", "BICGauss"]]`, on `gaussian_data`.
- `TimeCondGaussScore`: params `[TIERS, ["LogLikelihoodCondGauss", "AICCondGauss", "BICCondGauss"]]`, on `mixed_data`.
- `get_scoring_method` goes in a `TimeScoreFunctions` class.
- Scored structure: the generating model's DAG.

`ci_tests.py`, grouped by data type:
- discrete: `ChiSquare`, `GSq`, `LogLikelihood`, `ModifiedLogLikelihood`, `PowerDivergence`;
- continuous: `FisherZ`, `Pearsonr`, `PearsonrEquivalence`, `GCM`, `GeneralizedCov`, `HSIC`, `KCI`;
- mixed/multivariate: `HotellingLawley`, `PillaiTrace`, `RoysLargestRoot`, `WilksLambda`;
- `IndependenceMatch` (ground-truth independencies; no data);
- `get_ci_test`.

Test triple: X = first node, Y = last node, Z = first parent of Y. Kernel tests (`HSIC`, `KCI`, `GCM` if kernel-based) cap slow rows at the largest size within the per-call cap, documented.
Commit message: `"Add structure score and CI test benchmarks"`.

### Task 15: `benchmarks/metrics.py`, `identification.py`, `prediction.py`, `readwrite.py`, `independencies.py`, `utils.py`, `datasets.py`

Recipe applies. Each module is a separate commit inside this task.

- **metrics:** `AdjacencyConfusionMatrix`, `OrientationConfusionMatrix`, `SHD`, `CorrelationScore`, `ImpliedCIs`, `FisherC`, `StructureScore`, `get_metric`. True graph is `graph("dag_20"/"dag_500")`. The estimated graph is a frozen perturbed copy: add `dag_20_perturbed`/`dag_500_perturbed` to `build_fixtures` (10% of edges reversed, 10% removed, seed 0). Data-based metrics use `gaussian_data` of `random_gaussian_8`/`arth150` and their DAGs.
- **identification:** `ID`, `Adjustment`, `Frontdoor`, `ProbabilityExpressionTree` on `admg_20`/`dag_20` (fast) and the largest feasible ADMG/DAG (slow, ≤ `_500`). Treatment = first node, outcome = last descendant of it in sorted order.
- **prediction:** `NaiveAdjustmentRegressor`, `NaiveIVRegressor`, `DoubleMLRegressor` on `gaussian_data("random_gaussian_8", 1_000)` / `("arth150", 100_000)`.
- **readwrite:** reader/writer pairs for BIF, XMLBIF, XBN, XDSL, UAI, NET (`asia` fast / `andes` slow), LGBNJSON (`random_gaussian_8` / `arth150`), and PomdpX (pgmpy's test-suite example document fast / a synthetic 50-state-variable document slow). All files are generated once by `build_fixtures` into `benchmarks/data/readwrite/` and committed. Readers are benchmarked by construction from `string=` plus `get_model()`, and any `get_*` that parses. Writers are benchmarked by construction plus serialization (`__str__` or `write_*` into a `tempfile.TemporaryDirectory` made in `prepare`).
- **independencies:** `Independencies`, `IndependenceAssertion`, using frozen assertion lists (`indep_small` ≈ 5 assertions, `indep_large` ≈ 12) added to `build_fixtures`. `closure` is exponential, so size `indep_large` to the per-call cap.
- **utils:** the public functions exported by `pgmpy.utils` except `ExperimentalWarning`, `StateNameMixin`, `get_example_model` (deprecated) and private `_check_*`, on `discrete_data`/`gaussian_data` at `ROWS[tier]`.
- **datasets:** class `TimeLoaders` with `time_load_model` (`"bnlearn/asia"` fast / `"bnlearn/munin"` slow) and `time_load_dataset` (smallest / largest entry in `list_datasets()` that loads in < 60 s). Class `TimeListings` has `params = [["fast"]]` and covers `list_models`/`list_datasets`. Offline handling: `TimeLoaders.prepare` calls the loader once (warming the HF cache), then sets `huggingface_hub.constants.HF_HUB_OFFLINE = True`. `teardown` restores the previous value.

- [ ] **Extra step for `datasets.py` (Review Focus 5): pin offline timing.** Add a test `test_loader_timed_call_is_offline` to `tests/test_suite.py`. It wraps `huggingface_hub.file_download.hf_hub_download` with a spy that records `constants.HF_HUB_OFFLINE` at call time, runs `TimeLoaders.time_load_model` via `run_once` for `"fast"`, and asserts that every call made after `prepare` saw `True`.

---

### Task 16: Fit the budgets

**Files:**
- Modify: `benchmarks/_tiers.py` (`TIMING`), individual benchmark modules (sizes only)

- [ ] **Step 1: Measure the fast tier.** Run `time ci/pr.sh "$(git -C ../pgmpy rev-parse origin/dev~1)" "$(git -C ../pgmpy rev-parse origin/dev)"`. Expected wall time on the dev machine: ≤ 7 minutes, so that ≤ 12 minutes is left as headroom for a 4-vCPU GitHub runner. If over, reduce `TIMING["fast"]["repeat"]` max_time first, then shrink the biggest fast fixtures. Do not group methods (spec forbids).
- [ ] **Step 2: Check noise.** Run `ci/pr.sh <sha> <sha>` (same commit twice) 3 times. Expected: exit 0 every time. Any benchmark flagged must get more repeats or a larger fast input until the 3 runs are clean.
- [ ] **Step 3: Measure the slow tier.** `asv run --show-stderr -b "\('slow'" "$(git -C ../pgmpy rev-parse origin/dev)^!"` (one commit), timed. Expected: ≤ 25 minutes on the dev machine, since the nightly does two commits on a slower runner, about 2 hours in all. If over, shrink the slowest slow-tier fixtures. `asv show` or the run log lists per-benchmark durations.
- [ ] **Step 4: Full harness.** `pytest -q && pytest -m slow -q`. Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -am "Tune tier timing to CI budgets"`

---

### Task 17: Workflows in pgmpy, results branch, README

**Files:**
- Create (in pgmpy, via a worktree on a new branch `benchmarks-ci` from `origin/dev`; never touch the user's current pgmpy checkout): `.github/workflows/benchmarks-pr.yml`, `.github/workflows/benchmarks-nightly.yml`
- Modify (this repo): `README.md`

- [ ] **Step 1: Write `benchmarks-pr.yml`.** Key content:

```yaml
name: Benchmarks
concurrency:
  group: ${{ github.workflow }}-${{ github.event_name }}-${{ github.head_ref || github.ref }}
  cancel-in-progress: ${{ github.event_name != 'merge_group' }}
on:
  pull_request:
  merge_group:
jobs:
  benchmarks:
    name: Benchmarks (fast tier)
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v6
        with: { fetch-depth: 0, path: pgmpy }
      - uses: actions/checkout@v6
        with: { repository: pgmpy/pgmpy-benchmarks, ref: master, path: pgmpy-benchmarks }
      - uses: actions/setup-python@v6
        with:
          python-version: "3.14"
          cache: pip
          cache-dependency-path: pgmpy-benchmarks/requirements-ci.txt
      - uses: actions/cache@v4
        with: { path: ~/.cache/huggingface, key: hf-pgmpy-benchmarks-v1 }
      - run: pip install -r pgmpy-benchmarks/requirements-ci.txt
      - id: base
        run: echo "sha=$(git -C pgmpy merge-base ${{ github.event.pull_request.base.sha }} ${{ github.event.pull_request.head.sha }})" >> "$GITHUB_OUTPUT"
      - working-directory: pgmpy-benchmarks
        run: ci/pr.sh ${{ steps.base.outputs.sha }} ${{ github.event.pull_request.head.sha }}
```

- [ ] **Step 2: Write `benchmarks-nightly.yml`.**
  - Triggers and job settings: `schedule: cron "0 3 * * *"` plus `workflow_dispatch`; `concurrency: { group: benchmarks-nightly, cancel-in-progress: false }`; job `if: github.repository == 'pgmpy/pgmpy'`; `timeout-minutes: 300`.
  - Checkouts: pgmpy at `ref: dev` with `fetch-depth: 0`, `path: pgmpy`. `pgmpy/pgmpy-benchmarks` `master` at `path: pgmpy-benchmarks` with `fetch-depth: 0` and `token: ${{ secrets.BENCHMARKS_PUSH_TOKEN }}`. `pgmpy/pgmpy-benchmarks` `results` at `path: results-store` with the same token.
  - Same setup-python and HF cache steps as the PR workflow.
  - `git config --global user.name "pgmpy-benchmarks-bot"` and `user.email "pgmpy-benchmarks-bot@users.noreply.github.com"`.
  - Final step: `working-directory: pgmpy-benchmarks`, `run: ci/nightly.sh ../results-store`.
  - No `notify` job.
- [ ] **Step 3: Lint.** Run `actionlint .github/workflows/benchmarks-*.yml` in the pgmpy worktree (install from GitHub releases if missing). Expected: no errors.
- [ ] **Step 4: Rewrite `README.md`.** Cover:
  - what the tiers are and how to select them;
  - local setup (`requirements-dev.txt`, `pip install -e ../pgmpy`, `pytest`, `pytest -m slow`, `ci/pr.sh <base> <head>`);
  - how to add a benchmark (the recipe, condensed);
  - fixtures (regenerate with `tools/build_fixtures.py`; never derive inputs from pgmpy at benchmark time);
  - CI behaviour (PR fast tier ≥1.5× fails; errors on both = warning; nightly compares with last benchmarked commit, publishes, fails on regressions or any error);
  - the API-change process (update benchmarks on `master`, then re-run the PR check);
  - one-time maintainer setup: create a fine-grained token with Contents read/write on `pgmpy/pgmpy-benchmarks` and store it as `BENCHMARKS_PUSH_TOKEN` in `pgmpy/pgmpy`; create the `results` branch.
- [ ] **Step 5: Commit.** In pgmpy, commit the two workflows on `benchmarks-ci`. Here, commit the README. **Do not push either repo, create the orphan `results` branch remotely, or open PRs.** Report to the user the exact commands for: pushing this branch, creating the remote `results` branch (`git switch --orphan results && git commit --allow-empty -m "Benchmark results store" && git push origin results`), and opening the pgmpy PR. They are outward-facing and need the user's go-ahead.
