import time
import numpy as np

from direction_solvers.Direction_Solver_Factory import Direction_Descent_Factory
from line_searches.Line_Search_Factory import Line_Search_Factory


class Refiner_MOPG:

    def __init__(self, theta_for_stationarity, max_time, sparse_tol, Gurobi_verbose, Gurobi_feas_tol, Gurobi_method, ALS_settings):
        self.__direction_solver = Direction_Descent_Factory.getDirectionCalculator('MOPGD', Gurobi_verbose, Gurobi_feas_tol, gurobi_method=Gurobi_method)
        self.__line_search = Line_Search_Factory.getLineSearch('MOALS', ALS_settings['alpha_0'], ALS_settings['delta'], ALS_settings['beta'], ALS_settings['min_alpha'])

        self.__theta_for_stationarity = theta_for_stationarity
        self.__max_time = max_time * 60 if max_time is not None else np.inf
        self.__sparse_tol = sparse_tol

    def refine(self, x_p, f_p, index_point, problem, time_elapsed):
        start_time = time.time()

        x_p_tmp = np.copy(x_p)
        f_p_tmp = np.copy(f_p)
        theta = -np.inf

        num_iter = 0
        num_f_evals = 0

        while theta < self.__theta_for_stationarity and time.time() - start_time + time_elapsed < self.__max_time:
            num_iter += 1

            J = problem.evaluateFunctionsJacobian(x_p_tmp)
            num_f_evals += problem.n

            d, theta = self.__direction_solver.computeDirection(problem, J, x_p_tmp, self.__max_time - time.time() + start_time - time_elapsed, consider_support=True)

            if theta < self.__theta_for_stationarity and time.time() - start_time + time_elapsed < self.__max_time:

                new_p, new_f, alpha, f_eval = self.__line_search.search(problem, x_p_tmp, d, f_p_tmp, J)
                num_f_evals += f_eval

                if alpha != 0 and time.time() - start_time + time_elapsed < self.__max_time:

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

                    x_p_tmp = new_p
                    f_p_tmp = new_f

                else:

                    break

        # J = problem.evaluateFunctionsJacobian(x_p_tmp)
        # num_f_evals += problem.n

        # _, theta = self.__direction_solver.computeDirection(problem, J, x_p_tmp, np.inf, consider_support=True)

        return x_p_tmp, f_p_tmp