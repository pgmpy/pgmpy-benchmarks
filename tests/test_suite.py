"""Runs every benchmark once, so CI catches broken benchmarks before asv does."""

import importlib
import inspect
import pkgutil
import re
import subprocess
import sys
import time
import warnings

import pytest
from huggingface_hub import constants, file_download

import benchmarks
from benchmarks._tiers import MIN_SAMPLES, TIERS, TIMING, Bench


def benchmark_classes():
    """Yield (module name, class, [benchmark method names]) for every benchmark class."""
    for info in pkgutil.iter_modules(benchmarks.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"benchmarks.{info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            methods = [m for m in dir(obj) if m.startswith(("time_", "timeraw_"))]
            if methods:
                yield info.name, obj, methods


def cases(tier):
    return list(_cases(tier))


def _cases(tier):
    for mod, cls, methods in benchmark_classes():
        if tier not in cls.params[0]:
            continue
        combos = [()]
        for values in cls.params[1:]:
            combos = [c + (v,) for c in combos for v in values]
        for method in methods:
            for combo in combos:
                label = ", ".join(repr(a) for a in (tier, *combo))
                yield pytest.param(
                    cls, method, (tier, *combo), id=f"{mod}.{cls.__name__}.{method}({label})"
                )


def is_pgmpy_deprecation(w):
    # pgmpy backticks the deprecated name in its messages; this ignores warnings from third-party libraries.
    return issubclass(w.category, (DeprecationWarning, FutureWarning)) and bool(
        re.search(r"`[^`]+` is deprecated", str(w.message))
    )


def run_once(cls, method, args, max_seconds):
    obj = cls()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj.setup(*args)
        start = time.perf_counter()
        if method.startswith("timeraw_"):
            code = getattr(obj, method)(*args)
            subprocess.run([sys.executable, "-c", code], check=True)
        else:
            getattr(obj, method)(*args)
        elapsed = time.perf_counter() - start
    if hasattr(obj, "teardown"):
        obj.teardown(*args)
    bad = [w for w in caught if is_pgmpy_deprecation(w)]
    assert not bad, [str(w.message) for w in bad]
    assert elapsed < max_seconds, f"{elapsed:.2f}s >= {max_seconds}s"


def test_all_classes_follow_conventions():
    classes = list(benchmark_classes())
    assert classes, "no benchmarks discovered"
    for _, cls, _ in classes:
        assert issubclass(cls, Bench), cls
        assert cls.param_names[0] == "tier", cls
        assert cls.params[0] in (TIERS, ["fast"]), cls
        assert not [m for m in dir(cls) if m.startswith(("peakmem_", "mem_", "track_"))], cls


def _cap(method, fast_cap):
    # timeraw_ benchmarks spawn an interpreter, so they get a larger fast-tier cap.
    return 10 if method.startswith("timeraw_") else fast_cap


@pytest.mark.parametrize("cls,method,args", cases("fast"))
def test_fast(cls, method, args):
    run_once(cls, method, args, max_seconds=_cap(method, 0.5))


@pytest.mark.slow
@pytest.mark.parametrize("cls,method,args", cases("slow"))
def test_slow(cls, method, args):
    run_once(cls, method, args, max_seconds=15)


def test_every_benchmark_gets_enough_samples():
    for _, cls, methods in benchmark_classes():
        repeat = getattr(cls, "repeat", None)
        for tier in cls.params[0]:
            for method in methods:
                if method.startswith("timeraw_"):
                    assert repeat is not None, (cls, method, "timeraw needs explicit repeat")
                if repeat is not None:
                    min_repeat = repeat if isinstance(repeat, int) else repeat[0]
                else:
                    min_repeat = TIMING[tier]["repeat"][0]
                assert cls.rounds * min_repeat >= MIN_SAMPLES, (cls, method, tier)


def test_every_tier_warms_up():
    # Without a warm-up the first sample of each round is a cold call, which widens the CI the
    # comparison relies on and hides regressions in short benchmarks.
    for tier in TIERS:
        assert TIMING[tier]["warmup_time"] > 0, tier


def test_methods_carry_no_timing_attributes():
    # asv reads these from the method first, which would bypass the class-level sample check.
    for _, cls, methods in benchmark_classes():
        for method in methods:
            func = getattr(cls, method)
            bad = [a for a in ("repeat", "rounds", "number", "warmup_time") if hasattr(func, a)]
            assert not bad, (cls, method, bad)


def test_deprecation_check_ignores_third_party_warnings():
    def warning(message):
        return warnings.WarningMessage(DeprecationWarning(message), DeprecationWarning, "f.py", 1)

    assert is_pgmpy_deprecation(warning("`pgmpy.estimators.PC` is deprecated since v1.1.0 and will go."))
    assert not is_pgmpy_deprecation(warning("'parseString' deprecated - use 'parse_string'"))


def test_loader_timed_call_is_offline(monkeypatch):
    # pgmpy binds hf_hub_download at import (pgmpy.utils.hf_hub), so patch that name as well as the original.
    from benchmarks.datasets import TimeLoaders
    from pgmpy.utils import hf_hub

    log = []
    real_download = file_download.hf_hub_download

    def spy(*args, **kwargs):
        log.append(constants.HF_HUB_OFFLINE)
        return real_download(*args, **kwargs)

    real_prepare = TimeLoaders.prepare

    def prepare(self, *args):
        real_prepare(self, *args)
        log.clear()  # only calls made after prepare are timed

    monkeypatch.setattr(file_download, "hf_hub_download", spy)
    monkeypatch.setattr(hf_hub, "hf_hub_download", spy)
    monkeypatch.setattr(TimeLoaders, "prepare", prepare)
    before = constants.HF_HUB_OFFLINE
    run_once(TimeLoaders, "time_load_model", ("fast",), max_seconds=0.5)
    assert log and all(log), log
    assert constants.HF_HUB_OFFLINE == before  # teardown restored it
