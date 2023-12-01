from abc import abstractmethod


class ALS:

    def __init__(self, alpha_0, delta, beta, min_alpha):

        self._alpha_0 = alpha_0
        self._delta = delta
        self._beta = beta

        self._min_alpha = min_alpha

    @abstractmethod
    def search(self, problem, x, d):
        return
