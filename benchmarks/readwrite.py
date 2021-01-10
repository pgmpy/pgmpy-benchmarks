from pgmpy.readwrite import BIFReader, BIFWriter
from pgmpy.utils import get_example_model


class TimeRead:
    def setup(self):
        self.asia = get_example_model('asia')
        self.munin = get_example_model('munin')
        self.pathfinder = get_example_model('pathfinder')

    def time_asia_read(self):
        get_example_model('asia')

    def time_munin_read(self):
        get_example_model('munin')

    def time_pathfinder_read(self):
        get_example_model('pathfinder')

    def time_asia_write(self):
        BIFWriter(self.asia).write_bif('\tmp')

    def time_munin_write(self):
        BIFWriter(self.munin).write_bif('\tmp')

    def time_pathfinder_write(self):
        BIFWriter(self.pathfinder).write_bif('\tmp')
