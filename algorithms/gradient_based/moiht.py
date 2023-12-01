import time
import numpy as np

from algorithms.gradient_based.extended_gradient_based_algorithm import ExtendedGradientBasedAlgorithm


class MOIHT(ExtendedGradientBasedAlgorithm):

    def __init__(self,
                 max_iter, max_time, max_f_evals,
                 verbose, verbose_interspace,
                 plot_pareto_front, plot_pareto_solutions, plot_dpi,
                 theta_tol,
                 gurobi_method, gurobi_verbose, gurobi_feasibility_tol,
                 Gurobi_verbose, Gurobi_method, Gurobi_feas_tol, ALS_settings, Ref_settings):

        Gradient_Based_Algorithm.__init__(self,
                                          max_iter, max_time, max_f_evals, verbose, verbose_interspace, plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                          theta_tol,
                                          Gurobi_verbose,
                                          Gurobi_feas_tol,
                                          name_DDS='L_DS')

        self.__theta = -np.inf

        self.__sparse_tol = sparse_tol

        self._refiner = Refiner_MOPG(Ref_settings['MOPG']['theta_tol'],
                                     max_time,
                                     sparse_tol,
                                     Gurobi_verbose,
                                     Gurobi_feas_tol,
                                     Gurobi_method,
                                     ALS_settings)
        
        self._front_mode = False

    def search(self, p_list, f_list, problem):
        self.updateStoppingConditionCurrentValue('max_time', time.time())

        self.showFigure(p_list, f_list)

        index_point = 0

        while not self.evaluateStoppingConditions() and index_point < len(p_list):
            self.addToStoppingConditionCurrentValue('max_iter', 1)

            self.__theta = -np.inf

            J = problem.evaluateFunctionsJacobian(p_list[index_point, :])
            self.addToStoppingConditionCurrentValue('max_f_evals', problem.n)

            while not self.evaluateStoppingConditions() and self.__theta < self._theta_tol:
                new_p, self.__theta = self._direction_solver.computeDirection(problem, J, p_list[index_point, :], self.getStoppingConditionReferenceValue('max_time') - self.getStoppingConditionCurrentValue('max_time'))
                new_f = problem.evaluateFunctions(new_p)
                self.addToStoppingConditionCurrentValue('max_f_evals', 1)

                try:
                    assert np.linalg.norm(new_p, 0) <= problem.s
                except AssertionError:
                    print(str(index_point), np.linalg.norm(new_p, 0), problem.s, np.sum(np.abs(new_p) < self.__sparse_tol))
                    if np.sum(np.abs(new_p) < self.__sparse_tol) >= problem.n - problem.s:
                        new_p[np.abs(new_p) < self.__sparse_tol] = 0
                    else:
                        print('Warning!')
                        print(new_p)
                        break

                if not self.evaluateStoppingConditions() and self.__theta < self._theta_tol:
                    p_list[index_point, :] = new_p
                    f_list[index_point, :] = new_f

                    J = problem.evaluateFunctionsJacobian(p_list[index_point, :])
                    self.addToStoppingConditionCurrentValue('max_f_evals', problem.n)

                self.showFigure(p_list, f_list)

            self.showFigure(p_list, f_list)

            index_point += 1

        if not self._front_mode:
            time_elapsed = self.getStoppingConditionCurrentValue('max_time')
            self.updateStoppingConditionCurrentValue('max_time', time.time())
            for i in range(index_point):
                p_list[i, :], f_list[i, :] = self._refiner.refine(p_list[i, :], f_list[i, :], i, problem, self.getStoppingConditionCurrentValue('max_time'))
            p_list = p_list[:index_point, :]
            f_list = f_list[:index_point, :]
            self.closeFigure()
            time_elapsed += self.getStoppingConditionCurrentValue('max_time')
        else:
            self.closeFigure()
            self._refiner.updateStoppingConditionReferenceValue('max_time', self.getStoppingConditionReferenceValue('max_time'))
            p_list, f_list, _ = self._refiner.search(p_list[:index_point, :], f_list[:index_point, :], problem)
            time_elapsed = self.getStoppingConditionCurrentValue('max_time')

        return p_list, f_list, time_elapsed
