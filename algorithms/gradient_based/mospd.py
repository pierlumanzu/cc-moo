import time
import numpy as np

from algorithms.gradient_based.cc_adaptations.ifsd_adaptation import IFSDAdaptation
from algorithms.gradient_based.cc_adaptations.mosd_adaptation import MOSDAdaptation
from algorithms.gradient_based.extended_gradient_based_algorithm import ExtendedGradientBasedAlgorithm
from problems.penalty_problem import PenaltyProblem
from problems.extended_problem import ExtendedProblem

'''
For more details about the MOSPD algorithm, the user is referred to 

Lapucci, M. (2022). 
A penalty decomposition approach for multi-objective cardinality-constrained optimization problems. 
Optimization Methods and Software, 37(6), 2157–2189. 
https://doi.org/10.1080/10556788.2022.2060972

'''


class MOSPD(ExtendedGradientBasedAlgorithm):

    def __init__(self,
                 max_iter, max_time, max_f_evals,
                 verbose, verbose_interspace,
                 plot_pareto_front, plot_pareto_solutions, plot_dpi,
                 xy_diff, max_inner_iter_count, max_MOSD_iters, tau_0, max_tau_0_inc_factor, tau_inc_factor, epsilon_0, min_epsilon_0_dec_factor, epsilon_dec_factor,
                 gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                 approach, MOSD_IFSD_settings,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):
        
        if approach == 'Multi-Start':
            refiner_instance = MOSDAdaptation(max_time, max_f_evals,
                                              verbose, verbose_interspace,
                                              plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                              MOSD_IFSD_settings["theta_tol"],
                                              gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                              ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)
        elif approach == 'SFSD':
            refiner_instance = IFSDAdaptation(max_time, max_f_evals,
                                              verbose, verbose_interspace,
                                              plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                              MOSD_IFSD_settings["theta_tol"], MOSD_IFSD_settings["qth_quantile"],
                                              gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                              ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)
        else:
            refiner_instance = None

        ExtendedGradientBasedAlgorithm.__init__(self,
                                                max_time, max_f_evals,
                                                verbose, verbose_interspace,
                                                plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                                0.,
                                                gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                                0., 0., 0., 0.,
                                                name_DDS='Subspace_Steepest_Descent_DS', name_ALS='MOALS', refiner_instance=refiner_instance)

        self.__xy_diff = xy_diff

        self.__max_inner_iter_count = max_inner_iter_count
        self.__max_MOSD_iters = max_MOSD_iters

        self.__tau_0 = tau_0
        self.__max_tau_0_inc_factor = max_tau_0_inc_factor
        self.__tau_inc_factor = tau_inc_factor

        self.__epsilon_0 = epsilon_0
        self.__min_epsilon_0_dec_factor = min_epsilon_0_dec_factor
        self.__epsilon_dec_factor = epsilon_dec_factor

    def search(self, p_list, f_list, problem: ExtendedProblem):
        self.update_stopping_condition_current_value('max_time', time.time())

        p_list_y = np.copy(p_list)
        f_list_y = np.copy(f_list)

        self.show_figure(p_list_y, f_list_y)

        for index_p in range(len(p_list)):

            self.output_data(f_list_y)

            if self.evaluate_stopping_conditions():
                break

            x_p_tmp = np.copy(p_list[index_p, :])
            f_p_tmp = np.copy(f_list[index_p, :])
            y_p_tmp = np.copy(p_list_y[index_p, :])

            epsilon = self.__epsilon_0

            penalty_problem = PenaltyProblem(problem, np.copy(y_p_tmp), self.__tau_0)

            while not self.evaluate_stopping_conditions():

                J_p = penalty_problem.evaluate_functions_jacobian(x_p_tmp)
                self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

                for inner_iter_count in range(self.__max_inner_iter_count):

                    if self.evaluate_stopping_conditions():
                        break

                    d_p, theta_p = self._direction_solver.compute_direction(penalty_problem, J_p, x_p=x_p_tmp, time_limit=self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))

                    if self.evaluate_stopping_conditions() or theta_p >= epsilon:
                        break

                    new_point_found = False

                    penalty_f_p_tmp = penalty_problem.evaluate_functions(x_p_tmp)
                    self.add_to_stopping_condition_current_value('max_f_evals', 1)

                    for MOSD_iters in range(self.__max_MOSD_iters):

                        if self.evaluate_stopping_conditions() or theta_p >= epsilon:
                            break

                        new_x_p_tmp, new_penalty_f_p_tmp, _, f_eval = self._line_search.search(penalty_problem, x_p_tmp, penalty_f_p_tmp, d_p, theta_p)
                        self.add_to_stopping_condition_current_value('max_f_evals', f_eval)

                        if not self.evaluate_stopping_conditions() and new_x_p_tmp is not None:
                            new_point_found = True

                            x_p_tmp = new_x_p_tmp
                            new_penalty_f_p_tmp = penalty_f_p_tmp

                            # TODO: Restart from here
                        else:
                            break

                while not self.evaluateStoppingConditions() and theta < self.__epsilon and inner_iter_count < self.__max_inner_iter_count:

                    first_time = False

                    mopgd_iters = 0
                    alpha = 1

                    update = False

                    f_penalty_function = penalty_problem.evaluateFunctions(p_list[index_point, :])
                    self.addToStoppingConditionCurrentValue('max_f_evals', 1)

                    while not self.evaluateStoppingConditions() and theta < self.__epsilon and mopgd_iters < self.__max_mopgd_iters and alpha != 0:

                        new_p, new_f, alpha, f_eval = self._line_search.search(penalty_problem, p_list[index_point, :], d, f_penalty_function, J)
                        self.addToStoppingConditionCurrentValue('max_f_evals', f_eval)

                        if self.evaluateStoppingConditions():
                            break

                        if alpha != 0:
                            update = True

                            p_list[index_point, :] = new_p
                            f_list[index_point, :] = problem.evaluateFunctions(p_list[index_point, :])
                            self.addToStoppingConditionCurrentValue('max_f_evals', 1)

                            f_penalty_function = new_f

                            J = penalty_problem.evaluateFunctionsJacobian(p_list[index_point, :])
                            self.addToStoppingConditionCurrentValue('max_f_evals', problem.n)

                            d, theta = self._direction_solver.computeDirection(penalty_problem, J, p_list[index_point, :], self.getStoppingConditionReferenceValue('max_time') - self.getStoppingConditionCurrentValue('max_time'))

                        mopgd_iters += 1

                    if update:

                        penalty_problem.y = self.project(p_list[index_point, :], problem)

                        J = penalty_problem.evaluateFunctionsJacobian(p_list[index_point, :])
                        self.addToStoppingConditionCurrentValue('max_f_evals', problem.n)

                        d, theta = self._direction_solver.computeDirection(penalty_problem, J, p_list[index_point, :], self.getStoppingConditionReferenceValue('max_time') - self.getStoppingConditionCurrentValue('max_time'))

                        inner_iter_count += 1

                    else:
                        break

                penalty_problem.tau = min(penalty_problem.tau * self.__tau_inc_factor, self.__tau_0 * self.__tau_0_inc_factor)

                J = penalty_problem.evaluateFunctionsJacobian(p_list[index_point, :])
                self.addToStoppingConditionCurrentValue('max_f_evals', problem.n)

                self.__epsilon = min(self.__epsilon * self.__epsilon_dec_factor, self.__epsilon_0 * self.__epsilon_0_dec_factor)

                if self.__epsilon >= self.__epsilon_0 * self.__epsilon_0_dec_factor:
                    first_time = False

                p_list_y[index_point, :] = penalty_problem.y
                f_list_y[index_point, :] = problem.evaluateFunctions(p_list_y[index_point, :])
                self.addToStoppingConditionCurrentValue('max_f_evals', 1)

                if xy_diff

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

    @staticmethod
    def project(x, problem: ExtendedProblem):
        indices = np.argpartition(np.abs(x), problem.n - problem.s)

        x_proj = np.zeros(problem.n)
        x_proj[indices[problem.n - problem.s:]] = x[indices[problem.n - problem.s:]]

        return x_proj
