import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from problems.extended_problem import ExtendedProblem


class LR(ExtendedProblem):

    def __init__(self, dataset_path, s, sparsity_tol):

        self.__dataset_path = dataset_path

        (self.__X, self.__y) = load_svmlight_file(f=dataset_path, dtype=float)

        self.__X = np.asarray(self.__X.todense())
        scaler = StandardScaler()
        self.__X = scaler.fit_transform(self.__X)

        self.__y = self.__y.astype(float)
        if np.min(self.__y) == 0:
            self.__y = 2 * self.__y - 1

        ExtendedProblem.__init__(self, self.__X.shape[1], s, sparsity_tol)

        self.objectives = [
            1/self.__X.shape[0] * tf.reduce_sum([tf.math.log(1 + tf.exp(-self.__y[i] * tf.tensordot(self._z, self.__X[i, :], axes=1))) for i in range(len(self.__X))]),
            0.5 * tf.norm(self._z)**2
        ]

        self.lb_for_ini = 0 * np.ones(self.n)
        self.ub_for_ini = 1 * np.ones(self.n)

        self._L = max(np.linalg.norm(1/self.__X.shape[0] * np.dot(self.__X.T, self.__X), ord=2), 1)

    def name(self):
        return self.__dataset_path.split('/')[-1]

    @staticmethod
    def family_name():
        return "Logistic Regression"
