from benchmarks._bench import Bench


class TimeImport(Bench):
    repeat = (4, 10, 20.0)

    def timeraw_import_pgmpy(self):
        return "import pgmpy"
