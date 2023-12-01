import numpy as np
from gurobipy import Model, GRB

from nsma.direction_solvers.descent_direction.dds import DDS

from direction_solvers.gurobi_settings import ExtendedGurobiSettings
from problems.extended_problem import ExtendedProblem


class SubspaceSteepestDescentDSGurobiVersion(DDS, ExtendedGurobiSettings):

    def __init__(self, gurobi_method, gurobi_verbose, gurobi_feasibility_tol):
        DDS.__init__(self)
        ExtendedGurobiSettings.__init__(self, gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

    def compute_direction(self, problem: ExtendedProblem, Jac, x_p=None, subspace_support=None, time_limit=None):
        assert x_p is not None
        assert subspace_support is not None
        assert len(subspace_support) <= problem.s

        m, n = Jac.shape

        if np.isinf(Jac).any() or np.isnan(Jac).any():
            return np.zeros(n), 0

        try:
            model = Model("Subspace Steepest Descent Direction")
            model.setParam("Method", self._gurobi_method)
            model.setParam("OutputFlag", self._gurobi_verbose)
            model.setParam("FeasibilityTol", self._gurobi_feasibility_tol)
            if time_limit is not None:
                model.setParam("TimeLimit", max(time_limit, 0))

            z = model.addMVar(n, lb=-np.inf, ub=np.inf, name="z")
            beta = model.addMVar(1, lb=-np.inf, ub=0., name="beta")

            obj = beta - (x_p @ z) + 1/2 * (z @ z) + 1/2 * (x_p @ x_p)
            model.setObjective(obj)

            for j in range(m):
                model.addConstr(Jac[j, :] @ z <= beta + Jac[j, :] @ x_p, name='Jacobian Constraint n°{}'.format(j))

            for i in range(n):
                if i not in subspace_support:
                    model.addConstr(z[i] - x_p[i] <= 0, name='Subspace Constraint Upper Bound n°{}'.format(i))
                    model.addConstr(z[i] - x_p[i] >= 0, name='Subspace Constraint Lower Bound n°{}'.format(i))

            model.update()

            for i in range(n):
                z[i].start = float(x_p[i])
            beta.start = 0.

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                sol = model.getVars()

                d_p = np.array([s.x for s in sol][:n]) - x_p
                theta_p = model.getObjective().getValue()

            else:
                return np.zeros(n), 0

        except AttributeError:
            return np.zeros(n), 0

        return d_p, theta_p
