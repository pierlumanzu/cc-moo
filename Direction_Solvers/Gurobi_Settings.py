class Gurobi_Settings:

    def __init__(self, gurobi_verbose, gurobi_feas_tol, gurobi_method=None):
        self._gurobi_verbose = gurobi_verbose
        self._gurobi_feas_tol = gurobi_feas_tol
        self._gurobi_method = gurobi_method if gurobi_method is not None else -1