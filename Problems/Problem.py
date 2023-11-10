import numpy as np
import tensorflow as tf


class Problem:

    def __init__(self, n, s=None, L=None, L_inc_factor=None):
        self.__n = n
        self._z = tf.compat.v1.placeholder(dtype=tf.double, shape=[n, ])

        self.__objectives = np.empty(0)
        self.__objectives_gradients = np.empty(0)

        self.__constrained = False

        self.__lb = np.array([-np.inf] * self.__n, dtype=float)
        self.__lb_for_ini = np.array([-np.inf] * self.__n, dtype=float)

        self.__ub = np.array([np.inf] * self.__n, dtype=float)
        self.__ub_for_ini = np.array([np.inf] * self.__n, dtype=float)

        self._L = L
        self.__s = s
        self._L_inc_factor = L_inc_factor

        self._problem_name = None
        self._family_name = None

    def evaluateFunctions(self, x):
        return np.array([obj.eval({self._z: x}) for obj in self.__objectives])

    def evaluateFunctionsJacobian(self, x):

        jacobian = np.zeros((self.m, self.__n))
        for i in range(self.m):
            if self.__objectives_gradients[i] is not None:
                jacobian[i, :] = self.__objectives_gradients[i].eval({self._z: x})

        return jacobian

    def evaluateConstraints(self, x):

        if self.__constrained:
            constraints_evaluated = None

            constraints_evaluated = np.concatenate((constraints_evaluated, self.__lb[np.isfinite(self.__lb)] - x[np.isfinite(self.__lb)])) if constraints_evaluated is not None else self.__lb[np.isfinite(self.__lb)] - x[np.isfinite(self.__lb)]
            constraints_evaluated = np.concatenate((constraints_evaluated, x[np.isfinite(self.__ub)] - self.__ub[np.isfinite(self.__ub)])) if constraints_evaluated is not None else x[np.isfinite(self.__ub)] - self.__ub[np.isfinite(self.__ub)]

            return constraints_evaluated

        return np.empty(0)

    def checkPointFeasibility(self, x):

        if self.__constrained:

            if len(self.__lb) != 0:
                if (self.__lb > x).any():
                    return False

            if len(self.__ub) != 0:
                if (self.__ub < x).any():
                    return False

        return True

    def generateFeasiblePoints(self, mod, size, seed=None):
        assert mod.lower() in ['rand', 'hyper', 'rand_sparse']

        filtered_lb_for_ini = self.__lb_for_ini
        filtered_lb_for_ini[~np.isfinite(filtered_lb_for_ini)] = -2.0e19

        filtered_ub_for_ini = self.__ub_for_ini
        filtered_ub_for_ini[~np.isfinite(filtered_ub_for_ini)] = 2.0e19

        if mod.lower() == 'rand':
            p_list = np.zeros((size, self.__n), dtype=float)
            for i in range(size):
                p_list[i, :] = np.random.uniform(filtered_lb_for_ini, filtered_ub_for_ini)

        elif mod.lower() == 'rand_sparse':
            assert self.__s is not None
            assert seed is not None

            rng = np.random.default_rng(seed)
            p_list = np.zeros((size, self.__n), dtype=float)
            for i in range(size):
                p_list[i, :] = np.random.uniform(filtered_lb_for_ini, filtered_ub_for_ini)
                p_list[i, rng.choice(self.__n, size=self.__n - self.__s, replace=False)] = 0.

        elif mod.lower() == 'hyper':
            assert size > 1

            scale = filtered_ub_for_ini - filtered_lb_for_ini - 2e-3
            shift = filtered_lb_for_ini + 1e-3

            p_list = np.zeros((size, self.__n), dtype=float)
            for i in range(size):
                p_list[i, :] = shift + (i / (size - 1)) * scale

        else:
            raise NotImplementedError

        return p_list

    @property
    def objectives(self):
        raise RuntimeError

    @objectives.setter
    def objectives(self, objectives):
        for obj in objectives:
            assert obj is not np.nan and obj is not np.inf and obj is not -np.inf
        self.__objectives = objectives
        self.__objectives_gradients = [tf.gradients(obj, self._z)[0] for obj in self.__objectives]

    @property
    def constrained(self):
        return self.__constrained

    @property
    def lb(self):
        return self.__lb

    @lb.setter
    def lb(self, lb):
        assert len(lb) == self.__n
        assert not np.isnan(np.sum(lb))
        assert (lb != np.inf).all()

        self.__lb = lb
        self.__lb_for_ini = lb
        self.__constrained = True

    @property
    def lb_for_ini(self):
        raise RuntimeError

    @lb_for_ini.setter
    def lb_for_ini(self, lb_for_ini):
        assert len(lb_for_ini) == self.__n
        assert not np.isnan(np.sum(lb_for_ini))
        assert (lb_for_ini != np.inf).all()

        self.__lb_for_ini = lb_for_ini

    @property
    def ub(self):
        return self.__ub

    @ub.setter
    def ub(self, ub):
        assert len(ub) == self.__n
        assert not np.isnan(np.sum(ub))
        assert (ub != -np.inf).all()

        self.__ub = ub
        self.__ub_for_ini = ub
        self.__constrained = True

    @property
    def ub_for_ini(self):
        raise RuntimeError

    @ub_for_ini.setter
    def ub_for_ini(self, ub_for_ini):
        assert len(ub_for_ini) == self.__n
        assert not np.isnan(np.sum(ub_for_ini))
        assert (ub_for_ini != -np.inf).all()

        self.__ub_for_ini = ub_for_ini

    @property
    def n(self):
        return self.__n

    @property
    def m(self):
        return len(self.__objectives)

    @property
    def L(self):
        if self._L is None:
            raise RuntimeError
        return self._L

    @property
    def s(self):
        if self.__s is None:
            raise RuntimeError
        return self.__s

    @property
    def L_inc_factor(self):
        if self._L_inc_factor is None:
            raise RuntimeError
        return self._L_inc_factor

    def name(self):
        return self._problem_name

    def familyName(self):
        return self._family_name
