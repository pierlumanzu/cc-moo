import numpy as np
import tensorflow as tf
import pickle

from problems.extended_problem import ExtendedProblem


class QP(ExtendedProblem):

    def __init__(self, problem_path, s, sparsity_tol):

        self.__problem_path = problem_path

        (self.__Q_a, self.__c_a) = self.load_QP_problem(problem_path[:-4] + '_a' + problem_path[-4:])
        (self.__Q_b, self.__c_b) = self.load_QP_problem(problem_path[:-4] + '_b' + problem_path[-4:])

        ExtendedProblem.__init__(self, self.__Q_a.shape[1], s, sparsity_tol)

        self.objectives = [
            1 / 2 * tf.tensordot(self._z, tf.tensordot(self.__Q_a, self._z, axes=1), axes=1) - tf.tensordot(self.__c_a, self._z, axes=1),
            1 / 2 * tf.tensordot(self._z, tf.tensordot(self.__Q_b, self._z, axes=1), axes=1) - tf.tensordot(self.__c_b, self._z, axes=1)
        ]

        self.lb_for_ini = -2 * np.ones(self.n)
        self.ub_for_ini = 2 * np.ones(self.n)

        self._L = max(np.linalg.norm(self.__Q_a, ord=2), np.linalg.norm(self.__Q_b, ord=2))

    @staticmethod
    def load_QP_problem(problem_path):
        loaded_problem = pickle.load(open(problem_path, "rb"))
        return loaded_problem['Q'], loaded_problem['c']

    def name(self):
        return self.__problem_path

    @staticmethod
    def family_name():
        return "Quadratic"

