from pgmpy.estimators import PC, HillClimbSearch, K2Score
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


class TimeHillClimb:
    timeout = 600.0

    def setup(self):
        model = get_example_model('alarm')
        self.samples = BayesianModelSampling(model).forward_sample(size=int(1e4))
        scoring_method = K2Score(self.samples)
        est = HillClimbSearch(data=self.samples, scoring_method=scoring_method)

    def time_hillclimb(self):
        est.estimate(max_indegree=4, max_iter=int(1e4))

