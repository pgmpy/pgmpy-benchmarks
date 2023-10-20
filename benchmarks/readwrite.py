from pgmpy.readwrite import BIFReader, BIFWriter
from pgmpy.utils import get_example_model


class TimeReadAlarm:
    def setup(self):
        self.alarm = get_example_model('alarm')

    def time_read(self):
        get_example_model('alarm')

    def time_write(self):
        BIFWriter(self.alarm).write_bif('\tmp')


class TimeReadMunin:
    def setup(self):
        self.munin = get_example_model('munin')

    def time_read(self):
        get_example_model('munin')

    def time_write(self):
        BIFWriter(self.munin).write_bif('\tmp')

