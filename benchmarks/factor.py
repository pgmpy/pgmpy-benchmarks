import numpy as np

from pgmpy.factors.discrete import DiscreteFactor


class TimeDiscreteFactor:
    def setup(self):
        self.phi_large = DiscreteFactor(range(10), [2] * 10, [1] * (2**10))

    def time_factor_reduce(self):
        self.phi_large.reduce([(3, 0), (6, 1)], inplace=False)

    def time_factor_marginalize(self):
        self.phi_large.marginalize([4, 5, 8], inplace=False)

    def time_factor_multiply_large(self):
        phi = self.phi_large * self.phi_large

    def time_factor_compare(self):
        self.phi_large == self.phi_large
