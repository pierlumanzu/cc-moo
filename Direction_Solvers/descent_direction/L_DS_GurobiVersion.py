import numpy as np
from gurobipy import Model, GRB, GurobiError

from Direction_Solvers.L_DS.L_DS import L_DS
from Direction_Solvers.Gurobi_Settings import Gurobi_Settings


class L_DS_GurobiVersion(L_DS, Gurobi_Settings):

    def __init__(self, gurobi_verbose, gurobi_feas_tol):
        L_DS.__init__(self)
        Gurobi_Settings.__init__(self, gurobi_verbose, gurobi_feas_tol)

    def computeDirection(self, problem, Jac, x_p, time_limit):

        m = len(Jac)

        if np.isinf(Jac).any() or np.isnan(Jac).any():
            return np.zeros(problem.n), 0

        model = Model('L-stationarity Direction -- Modified Primal Problem')
        model.setParam('OutputFlag', self._gurobi_verbose)
        model.setParam('FeasibilityTol', self._gurobi_feas_tol)
        model.setParam('IntFeasTol', self._gurobi_feas_tol)
        model.setParam('TimeLimit', max(time_limit, 0))

        t = model.addMVar(1, lb=-np.inf, ub=0., name='t')
        z = model.addMVar(problem.n, lb=problem.lb, ub=problem.ub, name='z')
        delta = model.addMVar(problem.n, vtype=GRB.BINARY, name='delta')

        obj = t - problem.L * (x_p @ z) + problem.L/2 * (z @ z) + problem.L/2 * (x_p @ x_p)
        model.setObjective(obj)

        for j in range(m):
            model.addConstr(Jac[j, :] @ z <= t + Jac[j, :] @ x_p, name='Constraint on Jacobian of Function {}'.format(j))
        for i in range(problem.n):
            model.addSOS(GRB.SOS_TYPE1, [z[i], delta[i]], [1, 1])
        model.addConstr(np.ones(problem.n) @ delta >= problem.n - problem.s, name='Cardinality constraint')

        t.start = 0.
        for i in range(problem.n):
            z[i].start = x_p[i]
            delta[i].start = 0 if abs(x_p[i]) > 0 else 1

        model.update()
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            sol = np.array([s.x for s in model.getVars()])
            z_p = sol[1: problem.n + 1]
            theta_p = model.getObjective().getValue()
        else:
            return np.zeros(problem.n), 0

        return z_p, theta_p
