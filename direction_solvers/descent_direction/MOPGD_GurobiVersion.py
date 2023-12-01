import numpy as np
from gurobipy import Model, GRB

from direction_solvers.MOPGD.MOPGD import MOPGD
from direction_solvers.Gurobi_Settings import Gurobi_Settings


class MOPGD_GurobiVersion(MOPGD, Gurobi_Settings):

    def __init__(self, gurobi_verbose, gurobi_feas_tol, gurobi_method):
        MOPGD.__init__(self)
        Gurobi_Settings.__init__(self, gurobi_verbose, gurobi_feas_tol, gurobi_method=gurobi_method)

    def computeDirection(self, problem, Jac, x_p, time_limit, consider_support=False):

        m, n = Jac.shape

        if np.isinf(Jac).any() or np.isnan(Jac).any():
            return np.zeros(n), 0

        if type(consider_support) == bool:
            if consider_support:
                support = np.where(np.abs(x_p) > 0)[0]
                assert len(support) <= problem.s
        else:
            support = np.array(consider_support)
            assert len(support) <= problem.s

        model = Model("MOPGD")
        model.setParam("OutputFlag", self._gurobi_verbose)
        model.setParam("Method", self._gurobi_method)
        model.setParam("FeasibilityTol", self._gurobi_feas_tol)
        model.setParam("TimeLimit", max(time_limit, 0))

        t = model.addMVar(1, lb=-np.inf, ub=0., name="t")
        z = model.addMVar(n, lb=problem.lb, ub=problem.ub, name="z")
        
        obj = t - (x_p @ z) + 1/2 * (z @ z) + 1/2 * (x_p @ x_p)
        model.setObjective(obj)

        for j in range(m):
            model.addConstr(Jac[j, :] @ z <= t + Jac[j, :] @ x_p, name='Constraint on Jacobian of Function {}'.format(j))

        if type(consider_support) == bool:
            if consider_support:
                for i in range(n):
                    if i not in support:
                        model.addConstr(z[i] - x_p[i] <= 0, name='Upper bound of direction coordinate {}'.format(i))
                        model.addConstr(z[i] - x_p[i] >= 0, name='Lower bound of direction coordinate {}'.format(i))
        else:
            for i in range(n):
                if i not in support:
                    model.addConstr(z[i] - x_p[i] <= 0, name='Upper bound of direction coordinate {}'.format(i))
                    model.addConstr(z[i] - x_p[i] >= 0, name='Lower bound of direction coordinate {}'.format(i))

        t.start = 0.
        for i in range(n):
            z[i].start = float(x_p[i])
        
        model.update()
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            sol = np.array([s.x for s in model.getVars()])
            d_p = sol[1: problem.n + 1] - x_p
            theta_p = model.getObjective().getValue()
        else:
            return np.zeros(n), 0

        return d_p, theta_p
