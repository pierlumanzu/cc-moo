import time
import numpy as np

from algorithms.gradient_based.cc_adaptations.ifsd_adaptation import IFSDAdaptation
from algorithms.gradient_based.cc_adaptations.mosd_adaptation import MOSDAdaptation
from algorithms.gradient_based.extended_gradient_based_algorithm import ExtendedGradientBasedAlgorithm
from problems.extended_problem import ExtendedProblem


class MOIHT(ExtendedGradientBasedAlgorithm):

    def __init__(self,
                 max_time, max_f_evals,
                 verbose, verbose_interspace,
                 plot_pareto_front, plot_pareto_solutions, plot_dpi,
                 L, L_inc_factor, theta_tol,
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
                                                theta_tol,
                                                gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                                0., 0., 0., 0.,
                                                name_DDS='MOIHT_DS', refiner_instance=refiner_instance)

        self.__L = L
        self.__L_inc_factor = L_inc_factor

    def search(self, p_list, f_list, problem: ExtendedProblem):
        self.update_stopping_condition_current_value('max_time', time.time())

        self.show_figure(p_list, f_list)

        for index_p in range(len(p_list)):

            self.output_data(f_list)

            if self.evaluate_stopping_conditions():
                break

            x_p_tmp = np.copy(p_list[index_p, :])
            f_p_tmp = np.copy(f_list[index_p, :])

            theta_p = -np.inf

            while not self.evaluate_stopping_conditions() and theta_p < self._theta_tol:

                J_p = problem.evaluate_functions_jacobian(x_p_tmp)
                self.add_to_stopping_condition_current_value('max_f_evals', problem.n)

                new_x_p_tmp, theta_p = self._direction_solver.compute_direction(problem, J_p, x_p=x_p_tmp, L=self.__L * self.__L_inc_factor, time_limit=self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))

                if not self.evaluate_stopping_conditions() and theta_p < self._theta_tol:

                    new_f_p_tmp = problem.evaluate_functions(new_x_p_tmp)
                    self.add_to_stopping_condition_current_value('max_f_evals', 1)
                    
                    if np.sum(np.abs(new_x_p_tmp) >= problem.sparsity_tol) > problem.s:
                        print('Warning! Not found a feasible point! Optimization over!')
                        print(new_x_p_tmp)
                        break
                    else:
                        new_x_p_tmp[np.abs(new_x_p_tmp) < problem.sparsity_tol] = 0.

                    x_p_tmp = new_x_p_tmp
                    f_p_tmp = new_f_p_tmp

            p_list[index_p, :] = x_p_tmp
            f_list[index_p, :] = f_p_tmp

            self.show_figure(p_list, f_list)

        self.output_data(f_list)
        self.close_figure()

        return p_list, f_list, time.time() - self.get_stopping_condition_current_value('max_time')
