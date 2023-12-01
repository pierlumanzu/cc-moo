import numpy as np
import scipy


def pointsInitialization(problem, algorithm_name, seed=None):
    if algorithm_name not in ['GSS', 'MIP', 'F-MIP', 'F-GSS']:
        assert seed is not None
        p_list = problem.generateFeasiblePoints('rand_sparse', 2 * problem.n, seed=seed)
    else:
        p_list = np.zeros((2 * problem.n, problem.n), dtype=float)

    n_initial_points = len(p_list)
    f_list = np.zeros((n_initial_points, problem.m), dtype=float)
    for p in range(n_initial_points):
        f_list[p, :] = problem.evaluateFunctions(p_list[p, :])

    return p_list, f_list, n_initial_points


def pointsPostprocessing(p_list, f_list, problem):
    assert len(p_list) == len(f_list)
    old_n_points, _ = p_list.shape

    for p in range(old_n_points):
        f_list[p, :] = problem.evaluateFunctions(p_list[p, :])

    p_list, f_list = removeDuplicatesPoint(p_list, f_list)

    n_points, n = p_list.shape

    if old_n_points - n_points > 0:
        print('Warning: found {} duplicate points'.format(old_n_points - n_points))

    feasible = [True] * n_points
    infeasible_points = 0
    for p in range(n_points):
        constraints = problem.evaluateConstraints(p_list[p, :])
        if (constraints > 0).any():
            feasible[p] = False
            infeasible_points += 1
    if infeasible_points > 0:
        print('Warning: found {} infeasible points'.format(infeasible_points))

    p_list = p_list[feasible, :]
    f_list = f_list[feasible, :]

    efficient_point_idx = paretoEfficient(f_list)
    p_list = p_list[efficient_point_idx, :]
    f_list = f_list[efficient_point_idx, :]

    print('Results: found {} points'.format(len(p_list)))
    print()

    return p_list, f_list


def paretoEfficient(f_list):
    n_points, m = f_list.shape
    efficient = np.array([False] * n_points, dtype=bool)

    _, index = np.unique(f_list, return_index=True, axis=0)
    index = sorted(index)
    for el in np.arange(n_points):
        if el not in index:
            efficient[el] = False
    duplicates = [el for el in np.arange(n_points) if el not in index]
    indices = np.arange(n_points)

    for i in range(n_points):
        partial_ix = duplicates + [i]
        partial_matrix = f_list[np.delete(indices, partial_ix), :]
        dominance_matrix = partial_matrix - np.reshape(f_list[i, :], newshape=(1, m))
        is_dominated = (np.logical_and(np.sum(dominance_matrix <= 0, axis=1) == m, np.sum(dominance_matrix < 0, axis=1) > 0)).any()
        if not is_dominated:
            efficient[i] = True

    return efficient


def removeDuplicatesPoint(p_list, f_list, epsilon: float = 1e-16):

    is_duplicate = np.array([False] * p_list.shape[0])

    D = scipy.spatial.distance.cdist(p_list, p_list)
    D[np.triu_indices(len(p_list))] = np.inf

    D[np.isnan(D)] = np.inf

    is_duplicate[np.any(D < epsilon, axis=1)] = True

    p_list = p_list[~is_duplicate]
    f_list = f_list[~is_duplicate]

    return p_list, f_list
