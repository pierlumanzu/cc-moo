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
Optimization Methods and Software, 37(6), 2157-2189. 
https://doi.org/10.1080/10556788.2022.2060972

'''


class MOSPD(ExtendedGradientBasedAlgorithm):

    def __init__(self,
                 max_time, max_f_evals,
                 verbose, verbose_interspace,
                 plot_pareto_front, plot_pareto_solutions, plot_dpi,
                 xy_diff, max_inner_iter_count, max_MOSD_iters, tau_0, max_tau_0_inc_factor, tau_inc_factor, epsilon_0, min_epsilon_0_dec_factor, epsilon_dec_factor,
                 gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                 refiner, MOSD_IFSD_settings,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):
        
        if refiner == 'Multi-Start':
            refiner_instance = MOSDAdaptation(max_time, max_f_evals,
                                              verbose, verbose_interspace,
                                              plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                              MOSD_IFSD_settings["theta_tol"],
                                              gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                              ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)
        elif refiner == 'SFSD':
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
                                                ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha,
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

        self.show_figure(p_list, f_list)

        for index_p in range(len(p_list)):

            self.output_data(f_list)

            if self.evaluate_stopping_conditions():
                break

            x_p_tmp = np.copy(p_list[index_p, :])

            epsilon = self.__epsilon_0

            penalty_problem = PenaltyProblem(problem, np.copy(x_p_tmp), self.__tau_0)

            while not self.evaluate_stopping_conditions():

                J_p = penalty_problem.evaluate_functions_jacobian(x_p_tmp)
                self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

                for _ in range(self.__max_inner_iter_count):

                    if self.evaluate_stopping_conditions():
                        break

                    d_p, theta_p = self._direction_solver.compute_direction(penalty_problem, J_p, x_p=x_p_tmp, time_limit=self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))

                    if self.evaluate_stopping_conditions() or theta_p >= epsilon:
                        break

                    new_point_found = False

                    penalty_f_p_tmp = penalty_problem.evaluate_functions(x_p_tmp)
                    self.add_to_stopping_condition_current_value('max_f_evals', 1)

                    for _ in range(self.__max_MOSD_iters):

                        if self.evaluate_stopping_conditions() or theta_p >= epsilon:
                            break

                        new_x_p_tmp, new_penalty_f_p_tmp, _, f_eval = self._line_search.search(penalty_problem, x_p_tmp, penalty_f_p_tmp, d_p, theta_p)
                        self.add_to_stopping_condition_current_value('max_f_evals', f_eval)

                        if not self.evaluate_stopping_conditions() and new_x_p_tmp is not None:
                            new_point_found = True

                            x_p_tmp = new_x_p_tmp
                            penalty_f_p_tmp = new_penalty_f_p_tmp

                            J_p = penalty_problem.evaluate_functions_jacobian(x_p_tmp)
                            self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

                            d_p, theta_p = self._direction_solver.compute_direction(penalty_problem, J_p, x_p=x_p_tmp, time_limit=self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))

                        else:
                            break

                    if new_point_found:

                        penalty_problem.y = self.project(x_p_tmp, problem)

                        J_p = penalty_problem.evaluate_functions_jacobian(x_p_tmp)
                        self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

                    else:
                        break

                penalty_problem.tau = min(penalty_problem.tau * self.__tau_inc_factor, self.__tau_0 * self.__max_tau_0_inc_factor)

                epsilon = max(epsilon * self.__epsilon_dec_factor, self.__epsilon_0 * self.__min_epsilon_0_dec_factor)

                if np.linalg.norm(x_p_tmp - penalty_problem.y) <= self.__xy_diff:
                    break

            p_list[index_p, :] = penalty_problem.y

            f_list[index_p, :] = problem.evaluate_functions(penalty_problem.y)
            self.add_to_stopping_condition_current_value('max_f_evals', 1)

            self.show_figure(p_list, f_list)

        self.output_data(f_list)
        self.close_figure()

        p_list, f_list, _ = self.callRefiner(p_list[:index_p+(1 if index_p == len(p_list)-1 else 0)], f_list[:index_p+(1 if index_p == len(p_list)-1 else 0)], problem)

        return p_list, f_list, time.time() - self.get_stopping_condition_current_value('max_time')

    @staticmethod
    def project(x, problem: ExtendedProblem):
        indices = np.argpartition(np.abs(x), problem.n - problem.s)

        x_proj = np.zeros(problem.n)
        x_proj[indices[problem.n - problem.s:]] = x[indices[problem.n - problem.s:]]

        return x_proj
