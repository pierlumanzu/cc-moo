from direction_solvers.descent_direction.l_ds_gurobi_version import L_DS_GurobiVersion
from direction_solvers.descent_direction.bf_gurobi_version import BF_GurobiVersion
from direction_solvers.descent_direction.mopgd_gurobi_version import MOPGD_GurobiVersion


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
