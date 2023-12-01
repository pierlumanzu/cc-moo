from problems.logistic_regression.lr import LR


class Problem_Factory:
    @staticmethod
    def get_problem(problem_type, **kwargs):

        if problem_type == 'LR':

            return LR(kwargs['dat_prob_path'],
                      kwargs['s'],
                      kwargs['sparsity_tol'])

        else:
            raise NotImplementedError
