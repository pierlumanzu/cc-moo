import time
import scipy
import random
import numpy as np

from Algorithms.Algorithm_Utils.Selection_Strategy import Spread_Selection_Strategy
from Algorithms.Gradient_Based.Gradient_Based_Algorithm import Gradient_Based_Algorithm

from Direction_Solvers.Direction_Solver_Factory import Direction_Descent_Factory
from Line_Searches.Line_Search_Factory import Line_Search_Factory

from General_Utils.Pareto_Utils import paretoEfficient


class Refiner_FPGA(Gradient_Based_Algorithm):

    def __init__(self, max_iter, max_time, max_f_evals, verbose, verbose_interspace, plot_pareto_front, plot_pareto_solutions, plot_dpi, sparse_tol, theta_for_stationarity, Gurobi_verbose, Gurobi_feas_tol, Gurobi_method, ALS_settings):
        Gradient_Based_Algorithm.__init__(self,
                                          max_iter, max_time, max_f_evals, verbose, verbose_interspace, plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                          theta_for_stationarity,
                                          Gurobi_verbose, Gurobi_feas_tol,
                                          name_DDS='MOPGD',
                                          Gurobi_method=Gurobi_method,
                                          name_ALS='F-MOALS',
                                          ALS_settings=ALS_settings)

        self.__sparse_tol = sparse_tol

        self.__additional_direction_solver = Direction_Descent_Factory.getDirectionCalculator('BF', Gurobi_verbose, Gurobi_feas_tol)
        self.__additional_line_search = Line_Search_Factory.getLineSearch('MOALS', ALS_settings['alpha_0'], ALS_settings['delta'], ALS_settings['beta'], ALS_settings['min_alpha'])

    def search(self, p_list, f_list, problem):
        self.updateStoppingConditionCurrentValue('max_time', time.time())

        random.seed(16007)

        efficient_point_idx = paretoEfficient(f_list)
        p_list = p_list[efficient_point_idx, :]
        f_list = f_list[efficient_point_idx, :]

        encountered_supports = []
        point_support_matching = None

        for index_p in range(len(p_list)):
            if len(np.where(np.abs(p_list[index_p, :]) > self.__sparse_tol)[0]) < problem.s:
                J_k = problem.evaluateFunctionsJacobian(p_list[index_p, :])
                self.addToStoppingConditionCurrentValue('max_f_evals', problem.n)

                z_k, theta_k = self.__additional_direction_solver.computeDirection(problem, J_k, p_list[index_p, :], self.getStoppingConditionReferenceValue('max_time') - self.getStoppingConditionCurrentValue('max_time'))

                try:
                    assert np.linalg.norm(z_k, 0) <= problem.s

                    if not self.evaluateStoppingConditions() and theta_k < self._theta_for_stationarity:
                        new_p, new_f, _, f_eval_ls = self.__additional_line_search.search(problem, p_list[index_p, :], z_k - p_list[index_p, :], f_list[index_p, :], J_k)
                        self.addToStoppingConditionCurrentValue('max_f_evals', f_eval_ls)

                        if not self.evaluateStoppingConditions() and new_p is not None:
                            p_list[index_p, :] = new_p
                            f_list[index_p, :] = new_f

                except AssertionError:
                    print(str(index_p), np.linalg.norm(z_k, 0), problem.s, np.sum(np.abs(z_k) < self.__sparse_tol))
                    if np.sum(np.abs(z_k) < self.__sparse_tol) >= problem.n - problem.s:
                        z_k[np.abs(z_k) < self.__sparse_tol] = 0

                        if not self.evaluateStoppingConditions() and theta_k < self._theta_for_stationarity:
                            new_p, new_f, _, f_eval_ls = self.__additional_line_search.search(problem, p_list[index_p, :], z_k - p_list[index_p, :], f_list[index_p, :], J_k)
                            self.addToStoppingConditionCurrentValue('max_f_evals', f_eval_ls)

                            if not self.evaluateStoppingConditions() and new_p is not None:
                                p_list[index_p, :] = new_p
                                f_list[index_p, :] = new_f
                    else:
                        print('Warning!')
                        print(z_k)

            support = list(np.where(np.abs(p_list[index_p, :]) > self.__sparse_tol)[0])
            if support not in encountered_supports:
                encountered_supports.append(support)
                point_support_matching = np.concatenate((point_support_matching, np.array([len(encountered_supports) - 1]))) if point_support_matching is not None else np.array([len(encountered_supports) - 1])
            else:
                point_support_matching = np.concatenate((point_support_matching, np.array([encountered_supports.index(support)]))) if point_support_matching is not None else np.array([encountered_supports.index(support)])

        for index_support_1 in range(len(encountered_supports) - 1, -1, -1):
            if len(encountered_supports[index_support_1]) < problem.s:
                for index_support_2 in range(len(encountered_supports)):
                    if index_support_2 != index_support_1:
                        if set(encountered_supports[index_support_1]) <= set(encountered_supports[index_support_2]) and len(encountered_supports[index_support_2]) == problem.s:
                            del encountered_supports[index_support_1]

                            matching_to_change = np.where(point_support_matching == index_support_1)[0]

                            for index_matching in range(len(point_support_matching)):
                                if point_support_matching[index_matching] > index_support_1:
                                    point_support_matching[index_matching] -= 1

                            point_support_matching[matching_to_change] = index_support_2 - (1 if index_support_2 > index_support_1 else 0)

                            break

                    if index_support_2 == len(encountered_supports) - 1:
                        while len(encountered_supports[index_support_1]) < problem.s:
                            random_number = random.randint(0, problem.n - 1)
                            while random_number in encountered_supports[index_support_1]:
                                random_number = random.randint(0, problem.n - 1)
                            encountered_supports[index_support_1].append(random_number)

        n_points, n = p_list.shape
        m = f_list.shape[1]

        self.showFigure(p_list, f_list)

        update = True

        while not self.evaluateStoppingConditions() and update:
            self.outputData(f_list)

            self.addToStoppingConditionCurrentValue('max_iter', 1)

            update = False

            p_list_prev = np.copy(p_list)
            f_list_prev = np.copy(f_list)
            visited_prev = np.zeros(len(p_list_prev))
            point_support_matching_prev = np.copy(point_support_matching)

            while not self.evaluateStoppingConditions() and (visited_prev != 1).any():

                p = Spread_Selection_Strategy.select(f_list_prev, visited_prev)

                visited_prev[p] = 1

                x_k = p_list_prev[p, :]
                f_k = f_list_prev[p, :]

                p_list_by_support = p_list[np.where(point_support_matching == point_support_matching_prev[p])[0], :]
                f_list_by_support = f_list[np.where(point_support_matching == point_support_matching_prev[p])[0], :]
                point_support_matching_by_support = point_support_matching[np.where(point_support_matching == point_support_matching_prev[p])[0]]

                if self.existsDominatingPoint(f_k, f_list_by_support, equal_allowed=True):
                    continue

                p_in_support = np.where((f_list_by_support == f_k).all(axis=1))[0][0]

                crowding = self.calcCrowdingDistance(f_list_by_support, filter_out_duplicates=True)

                isfinite_index = np.isfinite(crowding)
                if len(crowding[isfinite_index]) > 0:
                    crowding_quantile = np.quantile(crowding[isfinite_index], 0.95)
                else:
                    crowding_quantile = np.inf

                J_k = problem.evaluateFunctionsJacobian(x_k)
                self.addToStoppingConditionCurrentValue('max_f_evals', n)

                power_set = self.objectivesPowerset(m)

                d_k, theta_k = self._direction_solver.computeDirection(problem, J_k, x_k, self.getStoppingConditionReferenceValue('max_time') - self.getStoppingConditionCurrentValue('max_time'), consider_support=encountered_supports[point_support_matching_prev[p]])
                try:
                    assert np.linalg.norm(d_k, 0) <= problem.s
                except AssertionError:
                    print(str(p), np.linalg.norm(d_k, 0), problem.s, np.sum(np.abs(d_k) < self.__sparse_tol))
                    if np.sum(np.abs(d_k) < self.__sparse_tol) >= problem.n - problem.s:
                        d_k[np.abs(d_k) < self.__sparse_tol] = 0
                    else:
                        print('Warning!')
                        print(d_k)
                        continue

                if not self.evaluateStoppingConditions() and theta_k < self._theta_for_stationarity:

                    new_p, new_f, _, f_eval_ls = self.__additional_line_search.search(problem, x_k, d_k, f_k, J_k)
                    self.addToStoppingConditionCurrentValue('max_f_evals', f_eval_ls)

                    if not self.evaluateStoppingConditions() and new_p is not None:

                        efficient_point_idx = self.fastNonDominatedFilter(f_list_by_support, new_f.reshape((1, m)))

                        p_list_by_support = np.concatenate((p_list_by_support[efficient_point_idx, :], new_p.reshape((1, n))), axis=0)
                        f_list_by_support = np.concatenate((f_list_by_support[efficient_point_idx, :], new_f.reshape((1, m))), axis=0)
                        point_support_matching_by_support = np.concatenate((point_support_matching_by_support[efficient_point_idx], np.array([point_support_matching_prev[p]])))

                        x_k = np.copy(new_p)
                        f_k = np.copy(new_f)

                        J_k = problem.evaluateFunctionsJacobian(x_k)
                        self.addToStoppingConditionCurrentValue('max_f_evals', n)
                else:
                    power_set.remove(tuple(range(m)))

                for I_k in power_set:

                    if self.evaluateStoppingConditions():
                        break

                    if self.existsDominatingPoint(f_k, f_list_by_support, equal_allowed=True) or crowding[p_in_support] < crowding_quantile:
                        break

                    d_k, theta_k = self._direction_solver.computeDirection(problem, J_k[I_k, ], x_k, self.getStoppingConditionReferenceValue('max_time') - self.getStoppingConditionCurrentValue('max_time'), consider_support=encountered_supports[point_support_matching_prev[p]])
                    try:
                        assert np.linalg.norm(d_k, 0) <= problem.s
                    except AssertionError:
                        print(str(p), np.linalg.norm(d_k, 0), problem.s, np.sum(np.abs(d_k) < self.__sparse_tol))
                        if np.sum(np.abs(d_k) < self.__sparse_tol) >= problem.n - problem.s:
                            d_k[np.abs(d_k) < self.__sparse_tol] = 0
                        else:
                            print('Warning!')
                            print(d_k)
                            continue

                    if not self.evaluateStoppingConditions() and theta_k < self._theta_for_stationarity:

                        new_p, new_f, _, f_eval_ls = self._line_search.search(problem, x_k, d_k, f_list_by_support, 0, np.arange(m))
                        self.addToStoppingConditionCurrentValue('max_f_evals', f_eval_ls)

                        if not self.evaluateStoppingConditions() and new_p is not None:

                            efficient_point_idx = self.fastNonDominatedFilter(f_list_by_support, new_f.reshape((1, m)))

                            update = True

                            p_list_by_support = np.concatenate((p_list_by_support[efficient_point_idx, :], new_p.reshape((1, n))), axis=0)
                            f_list_by_support = np.concatenate((f_list_by_support[efficient_point_idx, :], new_f.reshape((1, m))), axis=0)
                            point_support_matching_by_support = np.concatenate((point_support_matching_by_support[efficient_point_idx], np.array([point_support_matching_prev[p]])))

                            self.showFigure(p_list, f_list)

                p_list = np.concatenate((p_list[np.setdiff1d(np.arange(len(p_list)), np.where(point_support_matching == point_support_matching_prev[p])[0]), :],
                                         p_list_by_support))
                f_list = np.concatenate((f_list[np.setdiff1d(np.arange(len(f_list)), np.where(point_support_matching == point_support_matching_prev[p])[0]), :],
                                         f_list_by_support))
                point_support_matching = np.concatenate((point_support_matching[np.setdiff1d(np.arange(len(point_support_matching)), np.where(point_support_matching == point_support_matching_prev[p])[0])],
                                                         point_support_matching_by_support))

            self.showFigure(p_list, f_list)

        self.closeFigure()
        self.outputData(f_list)

        return p_list, f_list, self.getStoppingConditionCurrentValue('max_time')

    def findDuplicates(self, p_list, epsilon=1e-24):

        D = scipy.spatial.distance.cdist(p_list, p_list)
        D[np.triu_indices(len(p_list))] = np.inf
        is_duplicate = np.any(D < epsilon, axis=1)

        return is_duplicate

    def calcCrowdingDistance(self, f_list, filter_out_duplicates=True):

        n_points, n_obj = f_list.shape

        if n_points <= 2:
            return np.full(n_points, np.inf)

        else:
            if filter_out_duplicates:
                is_unique = np.where(np.logical_not(self.findDuplicates(f_list)))[0]
            else:
                is_unique = np.arange(n_points)

            _F = f_list[is_unique]

            I = np.argsort(_F, axis=0, kind='mergesort')

            _F = _F[I, np.arange(n_obj)]

            dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])

            norm = np.max(_F, axis=0) - np.min(_F, axis=0)
            norm[norm == 0] = np.nan

            dist_to_last, dist_to_next = dist, np.copy(dist)
            dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

            dist_to_last[np.isnan(dist_to_last)] = 0.0
            dist_to_next[np.isnan(dist_to_next)] = 0.0

            J = np.argsort(I, axis=0)
            _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

            crowding = np.zeros(n_points)
            crowding[is_unique] = _cd

        return crowding