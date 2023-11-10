from Direction_Solvers.descent_direction.L_DS_GurobiVersion import L_DS_GurobiVersion
from Direction_Solvers.descent_direction.BF_GurobiVersion import BF_GurobiVersion
from Direction_Solvers.descent_direction.MOPGD_GurobiVersion import MOPGD_GurobiVersion


class Direction_Descent_Factory:

    @staticmethod
    def getDirectionCalculator(direction_type, gurobi_verbose, gurobi_feas_tol, gurobi_method=None):

        if direction_type == 'L_DS':
            return L_DS_GurobiVersion(gurobi_verbose, gurobi_feas_tol)
        elif direction_type == 'BF':
            return BF_GurobiVersion(gurobi_verbose, gurobi_feas_tol)
        elif direction_type == 'MOPGD':
            return MOPGD_GurobiVersion(gurobi_verbose, gurobi_feas_tol, gurobi_method)
        else:
            raise NotImplementedError
