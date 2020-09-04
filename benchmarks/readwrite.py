from pgmpy.readwrite import BIFReader, BIFWriter


class TimeRead:
    def setup(self):
        self.asia = BIFReader('asia.bif').get_model()
        self.munin = BIFReader('munin.bif').get_model()
        self.pathfinder = BIFReader('pathfinder.bif').get_model()

    def time_asia_read(self):
        BIFReader('asia.bif').get_model()

    def time_munin_read(self):
        BIFReader('munin.bif').get_model()

    def time_pathfinder_read(self):
        BIFReader('pathfinder.bif').get_model()

    def time_asia_write(self):
        BIFWriter(self.asia).write_bif('\tmp')

    def time_munin_write(self):
        BIFWriter(self.munin).write_bif('\tmp')

    def time_pathfinder_write(self):
        BIFWriter(self.pathfinder).write_bif('\tmp')
