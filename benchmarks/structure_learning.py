from pgmpy.estimators import PC
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model


class TimePC:
    timeout = 600.0

    def setup(self):
        model = get_example_model('alarm')
        self.s = BayesianModelSampling(model)
        samples = self.s.forward_sample(size=int(1e4))
        self.est = PC(samples)

    def time_pc_stable(self):
        self.est.estimate(variant='stable')

    def time_pc_orig(self):
        self.est.estimate(variant='orig')

    def time_samples(self):
        self.s.forward_sample(size=int(1e5))

