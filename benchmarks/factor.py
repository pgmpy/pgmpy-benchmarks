import numpy as np

from pgmpy.factors.discrete import DiscreteFactor


class TimeDiscreteFactor:
    def setup(self):
        self.phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))
        self.phi_large = DiscreteFactor(range(10), [2] * 10, [1] * (2**10))

    def time_factor_define(self):
        phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))

    def time_factor_reduce(self):
        self.phi.reduce([('x1', 0), ('x2', 1)], inplace=False)

    def time_factor_reduce_large(self):
        self.phi_large.reduce([(3, 0), (6, 1)], inplace=False)

    def time_factor_marginalize(self):
        self.phi.marginalize(['x1'], inplace=False)

    def time_factor_marginalize_large(self):
        self.phi_large.marginalize([4, 5, 8], inplace=False)

    def time_factor_multiply(self):
        phi = self.phi * self.phi

    def time_factor_multiply_large(self):
        phi = self.phi_large * self.phi_large
