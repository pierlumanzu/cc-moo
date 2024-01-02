import numpy as np
import time
from abc import ABC
from itertools import chain, combinations
from nsma.algorithms.gradient_based.gradient_based_algorithm import GradientBasedAlgorithm

from direction_solvers.direction_solver_factory import Direction_Descent_Factory
from line_searches.line_search_factory import Line_Search_Factory


class ExtendedGradientBasedAlgorithm(GradientBasedAlgorithm, ABC):

    def __init__(self,
                 max_time, max_f_evals,
                 verbose, verbose_interspace,
                 plot_pareto_front, plot_pareto_solutions, plot_dpi,
                 theta_tol,
                 gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float,
                 name_DDS=None, name_ALS=None, refiner_instance: GradientBasedAlgorithm = None):

        GradientBasedAlgorithm.__init__(self,
                                        np.inf, max_time, max_f_evals,
                                        verbose, verbose_interspace,
                                        plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                        theta_tol,
                                        True, gurobi_method, gurobi_verbose,
                                        0., 0., 0., 0.)

        self._max_time = max_time

        self._direction_solver = Direction_Descent_Factory.getDirectionCalculator(name_DDS, gurobi_method, gurobi_verbose, gurobi_feasibility_tol) if name_DDS is not None else None
        self._line_search = Line_Search_Factory.getLineSearch(name_ALS, ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha) if name_ALS is not None else None
        self._refiner_instance = refiner_instance if refiner_instance is not None else None

    @staticmethod
    def objectivesPowerset(m):
        s = list(range(m))
        return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))

    @staticmethod
    def existsDominatingPoint(f, f_list):

        if np.isnan(f).any():
            return True

        n_obj = len(f)

        f = np.reshape(f, (1, n_obj))
        dominance_matrix = f_list - f

        return (np.logical_and(np.sum(dominance_matrix <= 0, axis=1) == n_obj, np.sum(dominance_matrix < 0, axis=1) > 0)).any()

    def callRefiner(self, p_list, f_list, problem):
        assert self._refiner_instance is not None

        self._refiner_instance.update_stopping_condition_reference_value('max_time', self.__max_time - time.time() + self.get_stopping_condition_current_value('max_time'))
        return self._refiner_instance.search(p_list, f_list, problem)
