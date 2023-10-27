import gc
from joblib.externals.loky import get_reusable_executor

from pgmpy.utils import get_example_model
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, ExpectationMaximization


class TimeMLEAlarm:
    timeout = 600
    repeat = 1
    number = 1

    def setup_cache(self):
        alarm = get_example_model('alarm')
        alarm_model = BayesianNetwork(alarm.edges())
        return MaximumLikelihoodEstimator(alarm_model, alarm.simulate(int(1e3)))

    def time_mle(self, alarm_est):
        alarm_est.get_parameters(n_jobs=1)

    def peakmem_mle(self, alarm_est):
        alarm_est.get_parameters(n_jobs=1)

    def teardown(self, alarm_est):
        get_reusable_executor().shutdown(wait=True)


class TimeMLEMunin:
    timeout = 600
    repeat = 1
    number = 1

    def setup_cache(self):
        munin = get_example_model('munin')
        munin_model = BayesianNetwork(munin.edges())
        return MaximumLikelihoodEstimator(munin_model, munin.simulate(int(1e3)))

    def time_mle(self, munin_est):
        munin_est.get_parameters(n_jobs=-1)

    def peakmem_mle(self, munin_est):
        munin_est.get_parameters(n_jobs=-1)

    def teardown(self, munin_est):
        get_reusable_executor().shutdown(wait=True)


class TimeBayesianEstimatorAlarm:
    timeout = 1200
    repeat = 1
    number = 1

    def setup_cache(self):
        alarm = get_example_model('alarm')
        alarm_model = BayesianNetwork(alarm.edges())
        self.alarm_est = BayesianEstimator(alarm_model, alarm.simulate(int(1e3)))
        return self.alarm_est

    def time_bayesian_estimator(self, est):
        est.get_parameters(n_jobs=1)

    def peakmem_bayesian_estimator(self, est):
        est.get_parameters(n_jobs=1)

    def teardown(self, est):
        get_reusable_executor().shutdown(wait=True)


class TimeBayesianEstimatorMunin:
    timeout = 1200
    repeat = 1
    number = 1

    def setup_cache(self):
        munin = get_example_model('munin')
        munin_model = BayesianNetwork(munin.edges())
        self.munin_est = BayesianEstimator(munin_model, munin.simulate(int(1e3)))
        return self.munin_est

    def time_bayesian_estimator(self, est):
        est.get_parameters(n_jobs=-1)

    def peakmem_bayesian_estimator(self, est):
        est.get_parameters(n_jobs=-1)

    def teardown(self, est):
        get_reusable_executor().shutdown(wait=True)


class TimeExpectationMaximizationAlarm:
    timeout = 600
    repeat = 1
    number = 1

    def setup_cache(self):
        alarm = get_example_model('alarm')
        alarm_model = BayesianNetwork(alarm.edges())
        self.alarm_est = ExpectationMaximization(alarm_model, alarm.simulate(int(1e3)))
        return self.alarm_est

    def time_em(self, est):
        est.get_parameters(n_jobs=1)

    def peakmem_em(self, est):
        est.get_parameters(n_jobs=1)

    def teardown(self, est):
        get_reusable_executor().shutdown(wait=True)


class TimeExpectationMaximizationMunin:
    timeout = 600
    repeat = 1
    number = 1

    def setup_cache(self):
        munin = get_example_model('munin')
        munin_model = BayesianNetwork(munin.edges())
        self.munin_est = ExpectationMaximization(munin_model, munin.simulate(int(1e3)))
        return self.munin_est

    def time_em(self, est):
        est.get_parameters(n_jobs=-1)

    def peakmem_em(self, est):
        est.get_parameters(n_jobs=-1)

    def teardown(self, est):
        get_reusable_executor().shutdown(wait=True)
