from direction_solvers.descent_direction.feasible_steepest_descent_ds_gurobi_version import FeasibleSteepestDescentDSGurobiVersion
from direction_solvers.descent_direction.moiht_ds_gurobi_version import MOIHTDSGurobiVersion
from direction_solvers.descent_direction.subspace_steepest_descent_ds_gurobi_version import SubspaceSteepestDescentDSGurobiVersion


class Direction_Descent_Factory:

    @staticmethod
    def getDirectionCalculator(direction_type, gurobi_method, gurobi_verbose, gurobi_feasibility_tol):

        if direction_type == 'Feasible_Steepest_Descent_DS':
            return FeasibleSteepestDescentDSGurobiVersion(gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

        elif direction_type == 'MOIHT_DS':
            return MOIHTDSGurobiVersion(gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

        elif direction_type == 'Subspace_Steepest_Descent_DS':
            return SubspaceSteepestDescentDSGurobiVersion(gurobi_method, gurobi_verbose, gurobi_feasibility_tol)

        else:
            raise NotImplementedError
