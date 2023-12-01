from problems.regression.logistic.lr import LR
from problems.quadratic.qp import QP


class Problem_Factory:
    @staticmethod
    def get_problem(problem_type, **kwargs):

        if problem_type == 'LR':

            return LR(kwargs['dat_prob_path'],
                      s=kwargs['s'],
                      L=kwargs['L'],
                      L_inc_factor=kwargs['L_inc_factor'])

        elif problem_type == 'QP':

            return QP(kwargs['dat_prob_path'],
                      s=kwargs['s'],
                      L=kwargs['L'],
                      L_inc_factor=kwargs['L_inc_factor'])

        else:
            raise NotImplementedError
