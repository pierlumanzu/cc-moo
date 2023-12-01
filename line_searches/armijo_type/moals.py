import numpy as np

from line_searches.armijo_type.als import ALS


class MOALS(ALS):

    def __init__(self, alpha_0, delta, beta, min_alpha):
        ALS.__init__(self, alpha_0, delta, beta, min_alpha)

    def search(self, problem, x, d, f=None, J=None):
        assert f is not None and J is not None

        alpha = self._alpha_0
        new_x = x + alpha * d
        new_f = problem.evaluateFunctions(new_x)
        f_eval = 1

        while (not problem.checkPointFeasibility(new_x) or np.isnan(new_f).any() or np.isinf(new_f).any() or np.any(new_f >= f + self._beta * alpha * np.dot(J, d))) and alpha > self._min_alpha:
            alpha *= self._delta
            new_x = x + alpha * d
            new_f = problem.evaluateFunctions(new_x)
            f_eval += 1

        if alpha <= self._min_alpha:
            alpha = 0
            return None, None, alpha, f_eval

        return new_x, new_f, alpha, f_eval
