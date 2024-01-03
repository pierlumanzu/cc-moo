from problems.quadratic.qp import QP


class Problem_Factory:
    @staticmethod
    def get_problem(problem_type, **kwargs):

        if problem_type == 'QP':

            return QP(kwargs['prob_path'],
                      kwargs['s'],
                      kwargs['sparsity_tol'])

        else:
            raise NotImplementedError
