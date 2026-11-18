"""Base class for every benchmark, and the ``fresh`` cache for inputs."""

import copy
from functools import cache, wraps


class Bench:
    rounds = 2
    repeat = (4, 10, 0.25)
    warmup_time = 0.05  # > 0 forces an untimed warm-up call per round


def fresh(build):
    """Cache ``build``. Each call returns a deep copy (some benchmarks mutate their input); ``.shared`` caches one copy."""
    cached = cache(build)
    copied = wraps(build)(lambda *args, **kwargs: copy.deepcopy(cached(*args, **kwargs)))
    copied.shared = cache(copied)
    return copied
