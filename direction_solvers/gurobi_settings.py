from nsma.direction_solvers.gurobi_settings import GurobiSettings


class ExtendedGurobiSettings(GurobiSettings):

    def __init__(self, gurobi_method: int, gurobi_verbose: bool, gurobi_feasibility_tol: float):

        GurobiSettings.__init__(self, gurobi_method, gurobi_verbose)
        self._gurobi_feasibility_tol = gurobi_feasibility_tol
