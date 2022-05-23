from pgmpy.utils import get_example_model
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, ExpectationMaximization


class TimeMLE:
    timeout = 600

    def setup(self):
        self.alarm = get_example_model('alarm')
        self.alarm_model = BayesianNetwork(self.alarm.edges())
        self.alarm_est = MaximumLikelihoodEstimator(self.alarm_model, self.alarm.simulate(int(1e4), show_progress=False))

        self.munin = get_example_model('munin1')
        self.munin_model = BayesianNetwork(self.munin.edges())
        self.munin_est = MaximumLikelihoodEstimator(self.munin_model, self.munin.simulate(int(1e4), show_progress=False))

    def time_alarm_mle(self):
        self.alarm_est.get_parameters()

    def time_munin_mle(self):
        self.munin_est.get_parameters()


class TimeBayesianEstimator:
    timeout = 1200

    def setup(self):
        self.alarm = get_example_model('alarm')
        self.alarm_model = BayesianNetwork(self.alarm.edges())
        self.alarm_est = BayesianEstimator(self.alarm_model, self.alarm.simulate(int(1e4), show_progress=False))

        self.munin = get_example_model('munin1')
        self.munin_model = BayesianNetwork(self.munin.edges())
        self.munin_est = BayesianEstimator(self.munin_model, self.munin.simulate(int(1e4), show_progress=False))

    def time_alarm_bayesian_estimator(self):
        self.alarm_est.get_parameters()

    def time_munin_bayesian_estimator(self):
        self.munin_est.get_parameters()


class TimeExpectationMaximization:
    timeout = 600

    def setup(self):
        self.alarm = get_example_model('alarm')
        self.alarm_model = BayesianNetwork(self.alarm.edges())
        self.alarm_est = ExpectationMaximization(self.alarm_model, self.alarm.simulate(int(1e4), show_progress=False))

        self.munin = get_example_model('munin1')
        self.munin_model = BayesianNetwork(self.munin.edges())
        self.munin_est = ExpectationMaximization(self.munin_model, self.munin.simulate(int(1e4), show_progress=False))

    def time_alarm_em(self):
        self.alarm_est.get_parameters()

    def time_munin_em(self):
        self.munin_est.get_parameters()

