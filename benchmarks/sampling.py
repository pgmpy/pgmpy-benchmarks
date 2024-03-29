import gc

from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling, GibbsSampling


class TimeSamplingAlarm:
    timeout = 600.0

    def setup(self):
        self.model = get_example_model('alarm')

    def time_forward_sample(self):
        self.model.simulate(n_samples=int(1e3), show_progress=False)

    def teardown(self):
        del self.model
        gc.collect()

class TimeSamplingMunin:
    timeout = 600.0

    def setup(self):
        self.model = get_example_model('munin')

    def time_forward_sample(self):
        self.model.simulate(n_samples=int(1e3), show_progress=False)

    def teardown(self):
        del self.model
        gc.collect()

#     def time_rejection_sample(self):
#         self.model.simulate(n_samples=int(1e4), evidence={"HISTORY": "TRUE", "HR": "NORMAL"}, show_progress=False)
# 
#     def time_likelihood_sample(self):
#         self.s.likelihood_weighted_sample(evidence=[("HISTORY", "TRUE"), ("HR", "NORMAL")], size=int(1e4))
# 
#     def time_gibbs_sampling(self):
#         gibbs_samples = GibbsSampling(model=self.model)
#         gibbs_sampling.sample(size=int(1e4))
