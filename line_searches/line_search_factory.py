from nsma.line_searches.armijo_type.boundconstrained_front_als import BoundconstrainedFrontALS

from line_searches.armijo_type.mo_als import MOALS


class Line_Search_Factory:

    @staticmethod
    def getLineSearch(line_search_type, alpha_0, delta, beta, min_alpha):

        if line_search_type == 'Boundconstrained_Front_ALS':
            return BoundconstrainedFrontALS(alpha_0, delta, beta, min_alpha)

        elif line_search_type == 'MOALS':
            return MOALS(alpha_0, delta, beta, min_alpha)

        else:
            raise NotImplementedError
