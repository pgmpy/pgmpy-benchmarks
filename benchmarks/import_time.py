from benchmarks._tiers import Bench


class TimeImport(Bench):
    params = [["fast"]]
    # rounds * min repeat = 2 * 4 = 8 samples (see MIN_SAMPLES in _tiers.py).
    rounds = 2
    repeat = (4, 10, 20.0)

    def timeraw_import_pgmpy(self, tier):
        return "import pgmpy"
