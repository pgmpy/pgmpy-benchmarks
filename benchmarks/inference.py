from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination, BeliefPropagation


class TimeVariableEliminationAlarm:
    def setup(self):
        self.alarm = get_example_model('alarm')

    def time_query(self):
        infer = VariableElimination(self.alarm)
        infer.query(variables=['VENTLUNG', 'VENTALV', 'ARTCO2', 'CATECHOL'])

    def peakmem_query(self):
        infer = VariableElimination(self.alarm)
        infer.query(variables=['VENTLUNG', 'VENTALV', 'ARTCO2', 'CATECHOL'])

    def teardown(self):
        del self.alarm


class TimeVariableEliminationMunin:
    def setup(self):
        self.munin = get_example_model('munin')

    def time_query(self):
        infer = VariableElimination(self.munin)
        infer.query(variables=['L_LNLC8_ADM_MALOSS', 'L_LNLLP_ADM_MALOSS', 'L_LNLC8_LP_ADM_MALOSS', 'L_LNLE_ADM_MALOSS'])

    def peakmem_query(self):
        infer = VariableElimination(self.munin)
        infer.query(variables=['L_LNLC8_ADM_MALOSS', 'L_LNLLP_ADM_MALOSS', 'L_LNLC8_LP_ADM_MALOSS', 'L_LNLE_ADM_MALOSS'])

    def teardown(self):
        del self.munin


class TimeBeliefPropagationAlarm:
    timeout = 600

    def setup(self):
        self.alarm = get_example_model('alarm')

    def time_query(self):
        infer = BeliefPropagation(self.alarm)
        infer.query(variables=['VENTLUNG'])

    def peakmem_query(self):
        infer = BeliefPropagation(self.alarm)
        infer.query(variables=['VENTLUNG'])

    def teardown(self):
        del self.alarm
