from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination, BeliefPropagation


class TimeVariableElimination:
    def setup(self):
        self.alarm = get_example_model('alarm')
        self.munin = get_example_model('munin')

    def time_query_alarm(self):
        infer = VariableElimination(self.alarm)
        infer.query(variables=['VENTLUNG', 'VENTALV', 'ARTCO2', 'CATECHOL'])

    def time_query_munin(self):
        infer = VariableElimination(self.munin)
        infer.query(variables=['L_LNLC8_ADM_MALOSS', 'L_LNLLP_ADM_MALOSS', 'L_LNLC8_LP_ADM_MALOSS', 'L_LNLE_ADM_MALOSS'])


class TimeBeliefPropagation:
    timeout = 600

    def setup(self):
        self.alarm = get_example_model('alarm')

    def time_query_alarm(self):
        infer = BeliefPropagation(self.alarm)
        infer.query(variables=['VENTLUNG'])

