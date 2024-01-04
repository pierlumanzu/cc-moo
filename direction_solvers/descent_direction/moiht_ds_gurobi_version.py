import numpy as np
from gurobipy import Model, GRB

from nsma.direction_solvers.descent_direction.dds import DDS

from direction_solvers.gurobi_settings import ExtendedGurobiSettings
from problems.extended_problem import ExtendedProblem


class MOIHTDSGurobiVersion(DDS, ExtendedGurobiSettings):

    def __init__(self, gurobi_method: int, gurobi_verbose: bool, gurobi_feasibility_tol: float):
        DDS.__init__(self)
        ExtendedGurobiSettings.__init__(self, gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

    def compute_direction(self, problem: ExtendedProblem, Jac: np.array, x_p: np.array = None, L: float = None, time_limit: float = None):
        assert x_p is not None
        assert L is not None

        m, n = Jac.shape

        if np.isinf(Jac).any() or np.isnan(Jac).any():
            return np.zeros(n), 0

        try:
            model = Model('MOIHT Direction')
            model.setParam('OutputFlag', self._gurobi_verbose)
            model.setParam('FeasibilityTol', self._gurobi_feasibility_tol)
            model.setParam('IntFeasTol', self._gurobi_feasibility_tol)
            if time_limit is not None:
                model.setParam("TimeLimit", max(time_limit, 0))

            z = model.addMVar(n, lb=-np.inf, ub=np.inf, name='z')
            beta = model.addMVar(1, lb=-np.inf, ub=0., name='beta')
            delta = model.addMVar(n, vtype=GRB.BINARY, name='delta')

            obj = beta - L * (x_p @ z) + L/2 * (z @ z) + L/2 * (x_p @ x_p)
            model.setObjective(obj)

            for j in range(m):
                model.addConstr(Jac[j, :] @ z <= beta + Jac[j, :] @ x_p, name='Jacobian Constraint n°{}'.format(j))

            for i in range(n):
                model.addSOS(GRB.SOS_TYPE1, [z[i], delta[i]], [1, 1])
            model.addConstr(np.ones(n) @ delta >= n - problem.s, name='Zero Norm Constraint')

            model.update()

            for i in range(n):
                z[i].start = float(x_p[i])
                delta[i].start = 0 if abs(x_p[i]) >= problem.sparsity_tol else 1
            beta.start = 0.

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                sol = model.getVars()

                z_p = np.array([s.x for s in sol][:n])
                theta_p = model.getObjective().getValue()

            else:
                return np.zeros(n), 0

        except AttributeError:
            return np.zeros(n), 0

        return z_p, theta_p
