import numpy as np


class Spread_Selection_Strategy:

    @staticmethod
    def select(f_list, alpha_list):
        n_points, m = f_list.shape
        distances = [0] * n_points

        for i in range(m):
            current_obj = np.array(f_list[:, i])
            points = np.argsort(current_obj)
            points = np.array(np.reshape(points, newshape=(n_points,)))
            current_obj.sort()

            for p, j in enumerate(points):
                try:
                    distances[j] = min(-current_obj[p + 1] + current_obj[p], distances[j])
                except IndexError:
                    pass

        for el in np.argsort(distances):
            if not alpha_list[el]:
                return el
