import numpy as np
import scipy

from nsma.general_utils.pareto_utils import pareto_efficient

from problems.extended_problem import ExtendedProblem


def pointsInitialization(problem: ExtendedProblem, seed):
    p_list = problem.generateFeasiblePoints('rand_sparse', 2 * problem.n, seed=seed)

    n_initial_points = len(p_list)
    f_list = np.zeros((n_initial_points, problem.m), dtype=float)
    for p in range(n_initial_points):
        f_list[p, :] = problem.evaluate_functions(p_list[p, :])

    return p_list, f_list, n_initial_points


def pointsPostprocessing(p_list, f_list, problem):
    assert len(p_list) == len(f_list)
    old_n_points, _ = p_list.shape

    for p in range(old_n_points):
        f_list[p, :] = problem.evaluate_functions(p_list[p, :])

    p_list, f_list = removeDuplicatesPoint(p_list, f_list)

    n_points, n = p_list.shape

    if old_n_points - n_points > 0:
        print('Warning: found {} duplicate points'.format(old_n_points - n_points))

    feasible = [True] * n_points
    infeasible_points = 0
    for p in range(n_points):
        if np.sum(np.abs(p_list[p, :]) >= problem.sparsity_tol) > problem.s:
            feasible[p] = False
            infeasible_points += 1
    if infeasible_points > 0:
        print('Warning: found {} infeasible points'.format(infeasible_points))

    p_list = p_list[feasible, :]
    f_list = f_list[feasible, :]

    efficient_point_idx = pareto_efficient(f_list)
    p_list = p_list[efficient_point_idx, :]
    f_list = f_list[efficient_point_idx, :]

    print('Results: found {} feasible efficient points'.format(len(p_list)))
    print()

    return p_list, f_list


def removeDuplicatesPoint(p_list, f_list):

    is_duplicate = np.array([False] * p_list.shape[0])

    D = scipy.spatial.distance.cdist(p_list, p_list)
    D[np.triu_indices(len(p_list))] = np.inf

    D[np.isnan(D)] = np.inf

    is_duplicate[np.any(D < 1e-16, axis=1)] = True

    p_list = p_list[~is_duplicate]
    f_list = f_list[~is_duplicate]

    return p_list, f_list
