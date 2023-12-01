from nsma.direction_solvers.gurobi_settings import GurobiSettings


class ExtendedGurobiSettings(GurobiSettings):

    def __init__(self, gurobi_method, gurobi_verbose, gurobi_feasibility_tol):

        GurobiSettings.__init__(self, gurobi_method, gurobi_verbose)
        self._gurobi_feasibility_tol = gurobi_feasibility_tol