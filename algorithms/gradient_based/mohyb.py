import time
import numpy as np

from algorithms.gradient_based.cc_adaptations.ifsd_adaptation import IFSDAdaptation
from algorithms.gradient_based.cc_adaptations.mosd_adaptation import MOSDAdaptation
from algorithms.gradient_based.moiht import MOIHT
from algorithms.gradient_based.mospd import MOSPD
from algorithms.gradient_based.extended_gradient_based_algorithm import ExtendedGradientBasedAlgorithm
from problems.extended_problem import ExtendedProblem


class MOHyb(ExtendedGradientBasedAlgorithm):

    def __init__(self,
                 max_time, max_f_evals,
                 verbose, verbose_interspace,
                 plot_pareto_front, plot_pareto_solutions, plot_dpi,
                 xy_diff, max_inner_iter_count, max_MOSD_iters, tau_0, max_tau_0_inc_factor, tau_inc_factor, epsilon_0, min_epsilon_0_dec_factor, epsilon_dec_factor,
                 L, L_inc_factor, theta_tol,
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
                                                0., 0., 0., 0.,
                                                refiner_instance=refiner_instance)

        self.__mospd_instance = MOSPD(np.inf, np.inf,
                                      False, verbose_interspace,
                                      False, False, plot_dpi,
                                      xy_diff, max_inner_iter_count, max_MOSD_iters, tau_0, max_tau_0_inc_factor, tau_inc_factor, epsilon_0, min_epsilon_0_dec_factor, epsilon_dec_factor,
                                      gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                      None, MOSD_IFSD_settings,
                                      ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)
        
        self.__moiht_instance = MOIHT(np.inf, np.inf,
                                      False, verbose_interspace,
                                      False, False, plot_dpi,
                                      L, L_inc_factor, theta_tol,
                                      gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                                      None, MOSD_IFSD_settings,
                                      ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)
        
        self.__max_f_evals = max_f_evals if max_f_evals is not None else np.inf

    def search(self, p_list, f_list, problem: ExtendedProblem):
        self.update_stopping_condition_current_value('max_time', time.time())

        self.show_figure(p_list, f_list)

        for index_p in range(len(p_list)):

            self.output_data(f_list)

            if self.evaluate_stopping_conditions():
                break

            self.__mospd_instance.update_stopping_condition_current_value('max_f_evals', 0)
            self.__mospd_instance.update_stopping_condition_reference_value('max_f_evals', self.__max_f_evals - self.get_stopping_condition_current_value('max_f_evals'))

            self.__mospd_instance.update_stopping_condition_reference_value('max_time', self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))
            
            p_list[index_p, :], f_list[index_p, :], _ = self.__mospd_instance.search(p_list[index_p, :].reshape(1, problem.n), f_list[index_p, :].reshape(1, problem.m), problem)

            self.update_stopping_condition_current_value('max_f_evals', self.get_stopping_condition_current_value('max_f_evals') + self.__mospd_instance.get_stopping_condition_current_value('max_f_evals'))


            self.__moiht_instance.update_stopping_condition_current_value('max_f_evals', 0)
            self.__moiht_instance.update_stopping_condition_reference_value('max_f_evals', self.__max_f_evals - self.get_stopping_condition_current_value('max_f_evals'))

            self.__moiht_instance.update_stopping_condition_reference_value('max_time', self._max_time - time.time() + self.get_stopping_condition_current_value('max_time'))
            
            p_list[index_p, :], f_list[index_p, :], _ = self.__moiht_instance.search(p_list[index_p, :].reshape(1, problem.n), f_list[index_p, :].reshape(1, problem.m), problem)

            self.update_stopping_condition_current_value('max_f_evals', self.get_stopping_condition_current_value('max_f_evals') + self.__moiht_instance.get_stopping_condition_current_value('max_f_evals'))


            self.show_figure(p_list, f_list)

        self.output_data(f_list)
        self.close_figure()

        p_list, f_list, _ = self.callRefiner(p_list[:index_p+(1 if index_p == len(p_list)-1 else 0)], f_list[:index_p+(1 if index_p == len(p_list)-1 else 0)], problem)

        return p_list, f_list, time.time() - self.get_stopping_condition_current_value('max_time')
