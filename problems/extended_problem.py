from abc import ABC
import numpy as np

from nsma.problems.problem import Problem


class ExtendedProblem(Problem, ABC):

    def __init__(self, n, s, sparsity_tol):

        Problem.__init__(self, n)

        self.__lb_for_ini = np.array([-np.inf] * self.n, dtype=float)
        self.__filtered_lb_for_ini = np.array([-2.0e19] * self.n, dtype=float)
        self.__ub_for_ini = np.array([np.inf] * self.n, dtype=float)
        self.__filtered_ub_for_ini = np.array([2.0e19] * self.n, dtype=float)

        self.__s = s
        self.__sparsity_tol = sparsity_tol

        self._L = None

    def generateFeasiblePoints(self, mod, size, seed=None):
        assert mod.lower() == 'rand_sparse'
        assert seed is not None

        rng = np.random.default_rng(seed)
        p_list = np.zeros((size, self.n), dtype=float)
        for i in range(size):
            p_list[i, :] = np.random.uniform(self.__filtered_lb_for_ini, self.__filtered_ub_for_ini)
            p_list[i, rng.choice(self.n, size=self.n - self.__s, replace=False)] = 0.

        return p_list

    @property
    def lb_for_ini(self):
        return self.__lb_for_ini

    @lb_for_ini.setter
    def lb_for_ini(self, lb_for_ini):
        assert len(lb_for_ini) == self.n
        assert not np.isnan(np.sum(lb_for_ini))
        assert (lb_for_ini != np.inf).all()

        self.__lb_for_ini = lb_for_ini

        self.__filtered_lb_for_ini = self.__lb_for_ini
        self.__filtered_lb_for_ini[~np.isfinite(self.__filtered_lb_for_ini)] = -2.0e19

    @property
    def ub_for_ini(self):
        return self.__ub_for_ini

    @ub_for_ini.setter
    def ub_for_ini(self, ub_for_ini):
        assert len(ub_for_ini) == self.n
        assert not np.isnan(np.sum(ub_for_ini))
        assert (ub_for_ini != -np.inf).all()

        self.__ub_for_ini = ub_for_ini

        self.__filtered_ub_for_ini = self.__ub_for_ini
        self.__filtered_ub_for_ini[~np.isfinite(self.__filtered_ub_for_ini)] = 2.0e19

    @property
    def s(self):
        return self.__s

    @property
    def sparsity_tol(self):
        return self.__sparsity_tol

    @property
    def L(self):
        return self._L
