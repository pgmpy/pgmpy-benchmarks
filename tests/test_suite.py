"""Runs every benchmark once, so CI catches broken benchmarks before asv does."""

import importlib
import inspect
import itertools
import pkgutil
import re
import subprocess
import sys
import time
import warnings

import pytest
from huggingface_hub import constants

import benchmarks
from benchmarks._bench import Bench

# asv's Mann-Whitney test can only reach p < 0.002 with enough samples per commit
MIN_SAMPLES = 8


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


def cases():
    for mod, cls, methods in benchmark_classes():
        combos = list(itertools.product(*getattr(cls, "params", [])))
        for method in methods:
            for combo in combos:
                label = ", ".join(repr(a) for a in combo)
                yield pytest.param(cls, method, combo, id=f"{mod}.{cls.__name__}.{method}({label})")


def is_pgmpy_deprecation(w):
    # pgmpy backticks the deprecated name in its messages; this ignores warnings from third-party libraries.
    return issubclass(w.category, (DeprecationWarning, FutureWarning)) and bool(
        re.search(r"`[^`]+` is deprecated", str(w.message))
    )


def run_once(cls, method, args, max_seconds):
    obj = cls()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if hasattr(obj, "setup"):
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
        assert not [m for m in dir(cls) if m.startswith(("peakmem_", "mem_", "track_"))], cls


def _cap(method):
    # timeraw_ benchmarks spawn an interpreter, so they get a larger cap.
    return 10 if method.startswith("timeraw_") else 0.5


@pytest.mark.parametrize("cls,method,args", list(cases()))
def test_benchmark(cls, method, args):
    run_once(cls, method, args, max_seconds=_cap(method))


def test_every_benchmark_gets_enough_samples():
    for _, cls, methods in benchmark_classes():
        for method in methods:
            assert cls.rounds * cls.repeat[0] >= MIN_SAMPLES, (cls, method)


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


def test_loader_timed_call_is_offline():
    from benchmarks.datasets import TimeLoaders

    before = constants.HF_HUB_OFFLINE
    bench = TimeLoaders()
    bench.setup()
    assert constants.HF_HUB_OFFLINE is True
    bench.teardown()
    assert constants.HF_HUB_OFFLINE == before
