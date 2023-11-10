import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from Problems.Problem import Problem


class LR(Problem):

    def __init__(self, dataset_path, s, L=None, L_inc_factor=None):

        (self.__X, self.__y) = load_svmlight_file(f=dataset_path, dtype=float)

        self.__X = np.asarray(self.__X.todense())
        scaler = StandardScaler()
        self.__X = scaler.fit_transform(self.__X)

        self.__y = self.__y.astype(float)
        if np.min(self.__y) == 0:
            self.__y = 2 * self.__y - 1

        Problem.__init__(self, self.__X.shape[1], s=s, L=L, L_inc_factor=L_inc_factor)

        self.objectives = [
            1/self.__X.shape[0] * tf.reduce_sum([tf.math.log(1 + tf.exp(-self.__y[i] * tf.tensordot(self._z, self.__X[i, :], axes=1))) for i in range(len(self.__X))]),
            0.5 * tf.norm(self._z)**2
        ]

        self.lb_for_ini = 0 * np.ones(self.n)
        self.ub_for_ini = 1 * np.ones(self.n)

        if self._L is None:
            assert self._L_inc_factor is not None
            self._L = self._L_inc_factor * max(np.linalg.norm(1/self.__X.shape[0] * np.dot(self.__X.T, self.__X), ord=2), 1)

        self._problem_name = dataset_path.split('/')[-1]
        self._family_name = 'Logistic Regression'

