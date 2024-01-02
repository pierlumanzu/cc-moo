import numpy as np

from nsma.problems.problem import Problem


class PenaltyProblem(Problem):

    def __init__(self, problem: Problem, y_0, tau_0):
        Problem.__init__(self, problem.n)

        self.__problem = problem

        self.__y = y_0
        self.__tau = tau_0

    def evaluate_functions(self, x):
        return self.__problem.evaluate_functions(x) + self.__tau/2 * np.dot(x - self.__y, x - self.__y)

    def evaluate_functions_jacobian(self, x):
        functions_jacobian = self.__problem.evaluate_functions_jacobian(x)
        penalty_gradient = self.__tau * (x - self.__y)

        jacobian = np.zeros((self.__problem.m, self.__problem.n))
        for i in range(self.__problem.m):
            jacobian[i, :] = functions_jacobian[i, :] + penalty_gradient

        return jacobian

    def evaluate_constraints(self, x: np.array):
        return np.empty(0)

    def evaluate_constraints_jacobian(self, x: np.array):
        return np.empty(0)

    def check_point_feasibility(self, x: np.array):
        return True

    @Problem.objectives.setter
    def objectives(self, objectives: list):
        raise RuntimeError

    @Problem.general_constraints.setter
    def general_constraints(self, general_constraints: list):
        raise RuntimeError

    @Problem.lb.setter
    def lb(self, lb: np.array):
        raise RuntimeError

    @Problem.ub.setter
    def ub(self, ub: np.array):
        raise RuntimeError

    @property
    def n(self):
        return self.__problem.n

    @property
    def m(self):
        return self.__problem.m

    @staticmethod
    def name():
        return "Penalty Problem"

    @staticmethod
    def family_name():
        return "Penalty Problem"

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y):
        self.__y = y

    @property
    def tau(self):
        return self.__tau

    @tau.setter
    def tau(self, tau):
        self.__tau = tau
