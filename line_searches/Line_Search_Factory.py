from line_searches.armijo_type.F_MOALS import F_MOALS
from line_searches.armijo_type.MOALS import MOALS


class Line_Search_Factory:

    @staticmethod
    def getLineSearch(line_search_type, alpha_0, delta, beta, min_alpha):

        if line_search_type == 'F-MOALS':
            return F_MOALS(alpha_0, delta, beta, min_alpha)

        elif line_search_type == 'MOALS':
            return MOALS(alpha_0, delta, beta, min_alpha)

        else:
            raise NotImplementedError
