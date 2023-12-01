import numpy as np
from abc import ABC
from itertools import chain, combinations

from algorithms.Algorithm import Algorithm

from direction_solvers.Direction_Solver_Factory import Direction_Descent_Factory
from line_searches.Line_Search_Factory import Line_Search_Factory


class Gradient_Based_Algorithm(Algorithm, ABC):

    def __init__(self, max_iter, max_time, max_f_evals, verbose, verbose_interspace, plot_pareto_front, plot_pareto_solutions, plot_dpi, theta_for_stationarity, Gurobi_verbose, Gurobi_feas_tol, name_DDS=None, Gurobi_method=None, name_ALS=None, ALS_settings=None):
        Algorithm.__init__(self, max_iter, max_time, max_f_evals, verbose, verbose_interspace, plot_pareto_front, plot_pareto_solutions, plot_dpi)

        self._theta_for_stationarity = theta_for_stationarity

        self._direction_solver = Direction_Descent_Factory.getDirectionCalculator(name_DDS, Gurobi_verbose, Gurobi_feas_tol, gurobi_method=Gurobi_method) if name_DDS is not None else None
        self._line_search = Line_Search_Factory.getLineSearch(name_ALS, ALS_settings['alpha_0'], ALS_settings['delta'], ALS_settings['beta'], ALS_settings['min_alpha']) if name_ALS is not None and ALS_settings is not None else None

    @staticmethod
    def objectivesPowerset(m):
        s = list(range(m))
        return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))

    @staticmethod
    def existsDominatingPoint(new_f, f_list, equal_allowed=False):

        if np.isnan(new_f).any():
            return True

        n_obj = len(new_f)

        new_f = np.reshape(new_f, (1, n_obj))
        dominance_matrix = f_list - new_f

        if equal_allowed:
            is_dominated = (np.logical_and(np.sum(dominance_matrix <= 0, axis=1) == n_obj, np.sum(dominance_matrix < 0, axis=1) > 0)).any()
        else:
            is_dominated = (np.sum(dominance_matrix <= 0, axis=1) == n_obj).any()

        return is_dominated

    @staticmethod
    def fastNonDominatedFilter(curr_f_list, new_f_list):

        n_new_points, m = new_f_list.shape
        efficient = np.array([True] * curr_f_list.shape[0])

        for i in range(n_new_points):
            dominance_matrix = curr_f_list - np.reshape(new_f_list[i, :], newshape=(1, m))
            dominated_idx = np.logical_and(np.sum(dominance_matrix >= 0, axis=1) == m, np.sum(dominance_matrix > 0, axis=1) > 0)

            assert len(dominated_idx.shape) == 1
            dom_indices = np.where(dominated_idx)[0]

            if len(dom_indices) > 0:
                efficient[dom_indices] = False

        return efficient
