"""Base class and constants shared by every benchmark."""

ROWS = 1_000
DISCRETE = "asia"
GAUSSIAN = 8
MODEL = "alarm"


class Bench:
    rounds = 2
    repeat = (4, 10, 0.25)
    warmup_time = 0.05  # > 0 forces an untimed warm-up call per round
