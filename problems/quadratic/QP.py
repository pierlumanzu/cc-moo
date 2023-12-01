import numpy as np
import tensorflow as tf
import pickle

from problems.Problem import Problem


class QP(Problem):

    def __init__(self, problem_path, s, L=None, L_inc_factor=None):

        (self.__Q_a, self.__c_a) = self.load_QP_problem(problem_path[:-4] + '_a' + problem_path[-4:])
        (self.__Q_b, self.__c_b) = self.load_QP_problem(problem_path[:-4] + '_b' + problem_path[-4:])

        Problem.__init__(self, self.__Q_a.shape[1], s=s, L=L, L_inc_factor=L_inc_factor)

        self.objectives = [
            1 / 2 * tf.tensordot(self._z, tf.tensordot(self.__Q_a, self._z, axes=1), axes=1) - tf.tensordot(self.__c_a, self._z, axes=1),
            1 / 2 * tf.tensordot(self._z, tf.tensordot(self.__Q_b, self._z, axes=1), axes=1) - tf.tensordot(self.__c_b, self._z, axes=1)
        ]

        self.lb_for_ini = -2 * np.ones(self.n)
        self.ub_for_ini = 2 * np.ones(self.n)

        if self._L is None:
            assert self._L_inc_factor is not None
            self._L = self._L_inc_factor * max(np.linalg.norm(self.__Q_a, ord=2), np.linalg.norm(self.__Q_b, ord=2))

        self._problem_name = problem_path.split('/')[-1][:-4]
        self._family_name = 'quadratic'

    @staticmethod
    def load_QP_problem(problem_path):
        loaded_problem = pickle.load(open(problem_path, "rb"))
        return loaded_problem['Q'], loaded_problem['c']

    @property
    def Q_a(self):
        return self.__Q_a

    @property
    def c_a(self):
        return self.__c_a

    @property
    def Q_b(self):
        return self.__Q_b

    @property
    def c_b(self):
        return self.__c_b

