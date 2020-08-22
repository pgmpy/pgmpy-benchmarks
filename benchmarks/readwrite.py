# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


class TimeSuite:
    def setup(self):
        import pgmpy

class TimeRead:
    def setup(self):
        from pgmpy.utils import get_example_model
        self.get_example_model = get_example_model

    def time_asia_read(self):
        self.get_example_model('asia')

    def time_alarm_read(self):
        self.get_example_model('alarm')

    def time_munin_read(self):
        self.get_example_model('munin')

    def time_pathfinder_read(self):
        self.get_example_model('pathfinder')

class MemSuite:
    def mem_list(self):
        return [0] * 256
