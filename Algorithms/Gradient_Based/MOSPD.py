import time
import numpy as np

from Algorithms.Gradient_Based.Gradient_Based_Algorithm import Gradient_Based_Algorithm
from Algorithms.Gradient_Based.Refiner.Refiner_MOPG import Refiner_MOPG
from Problems.PenaltyFunction.MOPF import MOPF


class MOSPD(Gradient_Based_Algorithm):

    def __init__(self, max_iter, max_time, max_f_evals, verbose, verbose_interspace, plot_pareto_front, plot_pareto_solutions, plot_dpi, sparse_tol, xy_diff, max_inner_iter_count, max_mopgd_iters, tau_0, tau_0_inc_factor, tau_inc_factor, epsilon_0, epsilon_0_dec_factor, epsilon_dec_factor, Gurobi_verbose, Gurobi_method, Gurobi_feas_tol, ALS_settings, Ref_settings):

        Gradient_Based_Algorithm.__init__(self,
                                          max_iter, max_time, max_f_evals, verbose, verbose_interspace, plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                          None,
                                          Gurobi_verbose,
                                          Gurobi_feas_tol,
                                          Gurobi_method=Gurobi_method,
                                          name_DDS='MOPGD',
                                          name_ALS='MOALS',
                                          ALS_settings=ALS_settings)

        self.__xy_diff = xy_diff

        self.__max_inner_iter_count = max_inner_iter_count
        self.__max_mopgd_iters = max_mopgd_iters

        self.__tau_0 = tau_0
        self.__tau_0_inc_factor = tau_0_inc_factor
        self.__tau_inc_factor = tau_inc_factor

        self.__epsilon_0 = epsilon_0
        self.__epsilon_0_dec_factor = epsilon_0_dec_factor
        self.__epsilon_dec_factor = epsilon_dec_factor

        self.__epsilon = epsilon_0

        self._refiner = Refiner_MOPG(Ref_settings['MOPG']['theta_for_stationarity'],
                                     max_time,
                                     sparse_tol,
                                     Gurobi_verbose,
                                     Gurobi_feas_tol,
                                     Gurobi_method,
                                     ALS_settings)

        self._front_mode = False

    @staticmethod
    def project(x, problem):
        n_zeros = problem.n - problem.s

        new_y = np.zeros(problem.n)
        indices = np.argpartition(np.abs(x), n_zeros)
        new_y[indices[n_zeros:]] = x[indices[n_zeros:]]

        return new_y

    def search(self, p_list, f_list, problem):
        self.updateStoppingConditionCurrentValue('max_time', time.time())

        p_list_y = np.copy(p_list)
        f_list_y = np.copy(f_list)

        self.showFigure(p_list_y, f_list_y)

        index_point = 0

        while not self.evaluateStoppingConditions() and index_point < len(p_list):
            self.addToStoppingConditionCurrentValue('max_iter', 1)

            self.__epsilon = self.__epsilon_0

            penalty_function = MOPF(problem, np.copy(p_list[index_point, :]), self.__tau_0)

            J = penalty_function.evaluateFunctionsJacobian(p_list[index_point, :])
            self.addToStoppingConditionCurrentValue('max_f_evals', problem.n)

            first_time = True

            while (not self.evaluateStoppingConditions() and np.linalg.norm(p_list[index_point, :] - penalty_function.y) > self.__xy_diff) or first_time:

                inner_iter_count = 0
                d, theta = self._direction_solver.computeDirection(penalty_function, J, p_list[index_point, :], self.getStoppingConditionReferenceValue('max_time') - self.getStoppingConditionCurrentValue('max_time'))

                while not self.evaluateStoppingConditions() and theta < self.__epsilon and inner_iter_count < self.__max_inner_iter_count:

                    first_time = False

                    mopgd_iters = 0
                    alpha = 1

                    update = False

                    f_penalty_function = penalty_function.evaluateFunctions(p_list[index_point, :])
                    self.addToStoppingConditionCurrentValue('max_f_evals', 1)

                    while not self.evaluateStoppingConditions() and theta < self.__epsilon and mopgd_iters < self.__max_mopgd_iters and alpha != 0:

                        new_p, new_f, alpha, f_eval = self._line_search.search(penalty_function, p_list[index_point, :], d, f_penalty_function, J)
                        self.addToStoppingConditionCurrentValue('max_f_evals', f_eval)

                        if self.evaluateStoppingConditions():
                            break

                        if alpha != 0:
                            update = True

                            p_list[index_point, :] = new_p
                            f_list[index_point, :] = problem.evaluateFunctions(p_list[index_point, :])
                            self.addToStoppingConditionCurrentValue('max_f_evals', 1)

                            f_penalty_function = new_f

                            J = penalty_function.evaluateFunctionsJacobian(p_list[index_point, :])
                            self.addToStoppingConditionCurrentValue('max_f_evals', problem.n)

                            d, theta = self._direction_solver.computeDirection(penalty_function, J, p_list[index_point, :], self.getStoppingConditionReferenceValue('max_time') - self.getStoppingConditionCurrentValue('max_time'))

                        mopgd_iters += 1

                    if update:

                        penalty_function.y = self.project(p_list[index_point, :], problem)

                        J = penalty_function.evaluateFunctionsJacobian(p_list[index_point, :])
                        self.addToStoppingConditionCurrentValue('max_f_evals', problem.n)

                        d, theta = self._direction_solver.computeDirection(penalty_function, J, p_list[index_point, :], self.getStoppingConditionReferenceValue('max_time') - self.getStoppingConditionCurrentValue('max_time'))

                        inner_iter_count += 1

                    else:
                        break

                penalty_function.tau = min(penalty_function.tau * self.__tau_inc_factor, self.__tau_0 * self.__tau_0_inc_factor)

                J = penalty_function.evaluateFunctionsJacobian(p_list[index_point, :])
                self.addToStoppingConditionCurrentValue('max_f_evals', problem.n)

                self.__epsilon = min(self.__epsilon * self.__epsilon_dec_factor, self.__epsilon_0 * self.__epsilon_0_dec_factor)

                if self.__epsilon >= self.__epsilon_0 * self.__epsilon_0_dec_factor:
                    first_time = False

                p_list_y[index_point, :] = penalty_function.y
                f_list_y[index_point, :] = problem.evaluateFunctions(p_list_y[index_point, :])
                self.addToStoppingConditionCurrentValue('max_f_evals', 1)

                self.showFigure(p_list_y, f_list_y)

            self.showFigure(p_list_y, f_list_y)

            index_point += 1

        if not self._front_mode:
            time_elapsed = self.getStoppingConditionCurrentValue('max_time')
            self.updateStoppingConditionCurrentValue('max_time', time.time())
            for i in range(index_point):
                p_list_y[i, :], f_list_y[i, :] = self._refiner.refine(p_list_y[i, :], f_list_y[i, :], i, problem, self.getStoppingConditionCurrentValue('max_time'))
            p_list_y = p_list_y[:index_point, :]
            f_list_y = f_list_y[:index_point, :]
            self.closeFigure()
            time_elapsed += self.getStoppingConditionCurrentValue('max_time')
        else:
            self.closeFigure()
            self._refiner.updateStoppingConditionReferenceValue('max_time', self.getStoppingConditionReferenceValue('max_time'))
            p_list_y, f_list_y, _ = self._refiner.search(p_list_y[:index_point, :], f_list_y[:index_point, :], problem)
            time_elapsed = self.getStoppingConditionCurrentValue('max_time')

        return p_list_y, f_list_y, time_elapsed
