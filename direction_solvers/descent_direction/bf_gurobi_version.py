import numpy as np
from gurobipy import Model, GRB, GurobiError

from direction_solvers.BF.BF import BF
from direction_solvers.gurobi_settings import Gurobi_Settings


class BF_GurobiVersion(BF, Gurobi_Settings):

    def __init__(self, gurobi_verbose, gurobi_feas_tol):
        BF.__init__(self)
        Gurobi_Settings.__init__(self, gurobi_verbose, gurobi_feas_tol)

    def computeDirection(self, problem, Jac, x_p, time_limit):

        m = len(Jac)

        if np.isinf(Jac).any() or np.isnan(Jac).any():
            return np.zeros(problem.n), 0

        model = Model('Basic Feasibility Direction -- Modified Primal Problem')
        model.setParam('OutputFlag', self._gurobi_verbose)
        model.setParam('FeasibilityTol', self._gurobi_feas_tol)
        model.setParam('IntFeasTol', self._gurobi_feas_tol)
        model.setParam('TimeLimit', max(time_limit, 0))

        t = model.addMVar(1, lb=-np.inf, ub=0., name='t')
        z = model.addMVar(problem.n, lb=problem.lb, ub=problem.ub, name='z')
        delta = model.addMVar(problem.n, vtype=GRB.BINARY, name='delta')

        obj = t - (x_p @ z) + 1/2 * (z @ z) + (x_p @ x_p)
        model.setObjective(obj)

        for j in range(m):
            model.addConstr(Jac[j, :] @ z <= t + Jac[j, :] @ x_p, name='Constraint on Jacobian of Function {}'.format(j))
        for i in range(problem.n):
            model.addSOS(GRB.SOS_TYPE1, [z[i], delta[i]], [1, 1])
            if abs(x_p[i]) > 0:
                model.addConstr(delta[i] == 0)
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
