from direction_solvers.descent_direction.feasible_steepest_descent_ds_gurobi_version import FeasibleSteepestDescentDSGurobiVersion
from direction_solvers.descent_direction.moiht_ds_gurobi_version import MOIHTDSGurobiVersion
from direction_solvers.descent_direction.subspace_steepest_descent_ds_gurobi_version import SubspaceSteepestDescentDSGurobiVersion


class DirectionDescentFactory:

    @staticmethod
    def get_direction_calculator(direction_type: str, gurobi_method: int, gurobi_verbose: bool, gurobi_feasibility_tol: float):

        if direction_type == 'Feasible_Steepest_Descent_DS':
            return FeasibleSteepestDescentDSGurobiVersion(gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

        elif direction_type == 'MOIHT_DS':
            return MOIHTDSGurobiVersion(gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

        elif direction_type == 'Subspace_Steepest_Descent_DS':
            return SubspaceSteepestDescentDSGurobiVersion(gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

        else:
            raise NotImplementedError
