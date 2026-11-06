"""Tier definitions shared by every benchmark."""

TIERS = ["fast", "slow"]

ROWS = {"fast": 1_000, "slow": 100_000}

# Samples per benchmark and commit. `asv run` saves statistics but not the raw samples, so the
# comparison (ci/compare.py) flags a benchmark only when the head/base ratio is >= 1.5 AND the two
# 99% confidence intervals are disjoint. With 8 samples each interval is exactly [min, max] of that
# commit's samples, so a regression is caught only if it exceeds the benchmark's own max/min spread.
MIN_SAMPLES = 8

TIMING = {
    "fast": {"repeat": (4, 10, 0.25), "warmup_time": 0.05},
    # A warm-up above zero forces one untimed call per round. Without it the first sample of each
    # round is the first call in a fresh process (about 10x slower for ms-scale benchmarks), which
    # widens [min, max].
    "slow": {"repeat": (4, 4, 20.0), "warmup_time": 0.05},
}


class Bench:
    params = [TIERS]
    param_names = ["tier"]
    rounds = 2
    # 3x the longest slow benchmark, rounded up: 98.1 s for all its parameters and both rounds
    # (ci_tests.TimeContinuousCITest.time_is_independent), so 3 * 98.1 = 294.3 -> 300.
    timeout = 300

    def setup(self, tier, *params):
        # asv re-reads these attributes after setup, so this is how timing varies per tier.
        for name, value in TIMING[tier].items():
            setattr(self, name, value)
        self.prepare(tier, *params)

    def prepare(self, tier, *params):
        pass
