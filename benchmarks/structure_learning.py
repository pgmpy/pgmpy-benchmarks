import gc
from joblib.externals.loky import get_reusable_executor

from pgmpy.estimators import PC, HillClimbSearch, K2Score, MmhcEstimator, TreeSearch
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model


class TimePCAlarm:
    timeout = 1200.0
    rounds = 1
    repeat = 1
    number = 1

    def setup_cache(self):
        model = get_example_model('alarm')
        samples = model.simulate(n_samples=int(1e3), seed=42, show_progress=False)
        self.est = PC(samples)
        return self.est

    def time_pc_stable(self, est):
        est.estimate(variant='stable', max_cond_vars=3)

    def time_pc_orig(self, est):
        est.estimate(variant='orig', max_cond_vars=3)

    def time_pc_parallel(self, est):
        est.estimate(variant='parallel', max_cond_vars=3, n_jobs=-1)

    def teardown(self, est):
        get_reusable_executor().shutdown(wait=True)


class TimePCMunin:
    timeout = 600.0
    rounds = 1
    repeat = 1
    number = 1

    def setup_cache(self):
        model = get_example_model('munin1')
        samples = model.simulate(n_samples=int(1e3), seed=42, show_progress=False)
        self.est = PC(samples)
        return self.est

    def time_pc_orig(self, est):
        est.estimate(variant='orig', max_cond_vars=3)

    def time_pc_parallel(self, est):
        est.estimate(variant='parallel', max_cond_vars=3, n_jobs=-1)

    def teardown(self, est):
        get_reusable_executor().shutdown(wait=True)


class TimeHillClimbAlarm:
    timeout = 1200.0
    rounds = 1
    repeat = 1
    number = 1

    def setup(self):
        model = get_example_model('alarm')
        samples = model.simulate(n_samples=int(1e3), seed=42, show_progress=False)
        self.scoring_method = K2Score(samples)
        self.est = HillClimbSearch(data=samples)

    def time_hillclimb(self):
        self.est.estimate(max_indegree=4, scoring_method=self.scoring_method, max_iter=int(1e3))

    def teardown(self):
        get_reusable_executor().shutdown(wait=True)


# class TimeHillClimbMunin:
#     timeout = 2400.0
#     rounds = 1
#     repeat = 1
#     number = 1
# 
#     def setup(self):
#         model = get_example_model('munin1')
#         samples = model.simulate(n_samples=int(1e3), seed=42, show_progress=False)
#         self.scoring_method = K2Score(samples)
#         self.est = HillClimbSearch(data=samples)
# 
#     def time_hillclimb(self):
#         self.est.estimate(max_indegree=4, scoring_method=self.scoring_method, max_iter=int(1e2))
# 
#     def teardown(self):
#         get_reusable_executor().shutdown(wait=True)
# 

class TimeTreeSearchAlarm:
    timeout = 1200.0
    rounds = 1
    repeat = 1
    number = 1

    def setup(self):
        model = get_example_model('alarm')
        samples = model.simulate(n_samples=int(1e3), show_progress=False)
        self.est = TreeSearch(samples, n_jobs=-1)

    def time_tan(self):
        self.est.estimate(estimator_type="tan", class_node="HISTORY", show_progress=False)

    def teardown(self):
        get_reusable_executor().shutdown(wait=True)


class TimeTreeSearchMunin:
    timeout = 1200.0
    rounds = 1
    repeat = 1
    number = 1

    def setup(self):
        model = get_example_model('munin1')
        samples = model.simulate(n_samples=int(1e3), show_progress=False)
        self.est = TreeSearch(samples, n_jobs=-1)

    def time_tan(self):
        self.est.estimate(estimator_type="tan", class_node="R_LNLW_APB_MALOSS", show_progress=False)

    def teardown(self):
        get_reusable_executor().shutdown(wait=True)
