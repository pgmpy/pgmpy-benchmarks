from pgmpy.estimators import PC, HillClimbSearch, K2Score
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model


class TimePCAlarmModel:
    timeout = 600.0

    def setup(self):
        model = get_example_model('alarm')
        samples = model.simulate(n_samples=int(1e4), seed=42)
        self.est = PC(samples)

    def time_pc_stable(self):
        self.est.estimate(variant='stable')

    def time_pc_orig(self):
        self.est.estimate(variant='orig')


class TimeHillClimbAlarmModel:
    timeout = 600.0

    def setup(self):
        model = get_example_model('alarm')
        samples = model.simulate(n_samples=int(1e4), seed=42)
        self.scoring_method = K2Score(samples)
        est = HillClimbSearch(data=samples)

    def time_hillclimb(self):
        est.estimate(max_indegree=4, scoring_method=self.scoring_method, max_iter=int(1e4))


class TimeTreeSearchAlarmModel:
    timeout = 600.0

    def setup(self):
        model = get_example_model('alarm')
        samples = model.simulate(n_samples=int(1e4))
        self.est = TreeSearch(samples)

    def time_tan(self):
        self.est.estimate(estimator_type="tan", class_node="HISTORY", show_progress=False)


class TimeMmhcAlarmModel:
    timeout = 600.0

    def setup(self):
        model = get_example_model('alarm')
        samples = model.simulate(n_samples=int(1e4))
        self.est = MmhcEstimator(samples)

    def time_mmhc(self):
        self.est.estimate()
