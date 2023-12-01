from algorithms.gradient_based.F_MOIHT import F_MOIHT
from algorithms.gradient_based.moiht import MOIHT
from algorithms.gradient_based.F_MOSPD import F_MOSPD
from algorithms.gradient_based.mospd import MOSPD
from algorithms.gradient_based.F_MOSPD_MOIHT import F_MOSPD_MOIHT
from algorithms.gradient_based.MOSPD_MOIHT import MOSPD_MOIHT
from algorithms.gradient_based.F_MIP import F_MIP
from algorithms.gradient_based.MIP import MIP
from algorithms.gradient_based.F_GSS import F_GSS
from algorithms.gradient_based.GSS import GSS


class Algorithm_Factory:
    @staticmethod
    def get_algorithm(algorithm_name, **kwargs):

        general_settings = kwargs['general_settings']

        algorithms_settings = kwargs['algorithms_settings']

        if algorithm_name == 'F-MOIHT':
            F_MOIHT_settings = algorithms_settings[algorithm_name]
            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']
            Ref_settings = kwargs['Ref_settings']
            sparsity_settings = kwargs['sparsity_settings']

            algorithm = F_MOIHT(general_settings['max_iter'],
                                general_settings['max_time'],
                                general_settings['max_f_evals'],
                                general_settings['verbose'],
                                general_settings['verbose_interspace'],
                                general_settings['plot_pareto_front'],
                                general_settings['plot_pareto_solutions'],
                                general_settings['plot_dpi'],
                                sparsity_settings['sparsity_tol'],
                                F_MOIHT_settings['theta_tol'],
                                DDS_settings['verbose'],
                                DDS_settings['method'],
                                DDS_settings['feas_tol'],
                                ALS_settings,
                                Ref_settings)

        elif algorithm_name == 'MOIHT':
            MOIHT_settings = algorithms_settings[algorithm_name]
            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']
            Ref_settings = kwargs['Ref_settings']
            sparsity_settings = kwargs['sparsity_settings']

            return MOIHT(general_settings['max_iter'],
                         general_settings['max_time'],
                         general_settings['max_f_evals'],
                         general_settings['verbose'],
                         general_settings['verbose_interspace'],
                         general_settings['plot_pareto_front'],
                         general_settings['plot_pareto_solutions'],
                         general_settings['plot_dpi'],
                         sparsity_settings['sparsity_tol'],
                         MOIHT_settings['theta_tol'],
                         DDS_settings['verbose'],
                         DDS_settings['method'],
                         DDS_settings['feas_tol'],
                         ALS_settings,
                         Ref_settings)

        elif algorithm_name == 'MOSPD':
            MOSPD_settings = algorithms_settings[algorithm_name]
            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']
            Ref_settings = kwargs['Ref_settings']
            sparsity_settings = kwargs['sparsity_settings']

            return MOSPD(general_settings['max_iter'],
                         general_settings['max_time'],
                         general_settings['max_f_evals'],
                         general_settings['verbose'],
                         general_settings['verbose_interspace'],
                         general_settings['plot_pareto_front'],
                         general_settings['plot_pareto_solutions'],
                         general_settings['plot_dpi'],
                         sparsity_settings['sparsity_tol'],
                         MOSPD_settings['xy_diff'],
                         MOSPD_settings['max_inner_iter_count'],
                         MOSPD_settings['max_mopgd_iters'],
                         MOSPD_settings['tau_0'],
                         MOSPD_settings['tau_0_inc_factor'],
                         MOSPD_settings['tau_inc_factor'],
                         MOSPD_settings['epsilon_0'],
                         MOSPD_settings['epsilon_0_dec_factor'],
                         MOSPD_settings['epsilon_dec_factor'],
                         DDS_settings['verbose'],
                         DDS_settings['method'],
                         DDS_settings['feas_tol'],
                         ALS_settings,
                         Ref_settings)

        elif algorithm_name == 'F-MOSPD':
            MOSPD_settings = algorithms_settings['MOSPD']
            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']
            Ref_settings = kwargs['Ref_settings']
            sparsity_settings = kwargs['sparsity_settings']

            return F_MOSPD(general_settings['max_iter'],
                           general_settings['max_time'],
                           general_settings['max_f_evals'],
                           general_settings['verbose'],
                           general_settings['verbose_interspace'],
                           general_settings['plot_pareto_front'],
                           general_settings['plot_pareto_solutions'],
                           general_settings['plot_dpi'],
                           sparsity_settings['sparsity_tol'],
                           MOSPD_settings['xy_diff'],
                           MOSPD_settings['max_inner_iter_count'],
                           MOSPD_settings['max_mopgd_iters'],
                           MOSPD_settings['tau_0'],
                           MOSPD_settings['tau_0_inc_factor'],
                           MOSPD_settings['tau_inc_factor'],
                           MOSPD_settings['epsilon_0'],
                           MOSPD_settings['epsilon_0_dec_factor'],
                           MOSPD_settings['epsilon_dec_factor'],
                           DDS_settings['verbose'],
                           DDS_settings['method'],
                           DDS_settings['feas_tol'],
                           ALS_settings,
                           Ref_settings)

        elif algorithm_name == 'MOSPD-MOIHT':
            MOSPD_settings = algorithms_settings['MOSPD']
            MOIHT_settings = algorithms_settings['MOIHT']
            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']
            Ref_settings = kwargs['Ref_settings']
            sparsity_settings = kwargs['sparsity_settings']

            return MOSPD_MOIHT(general_settings['max_iter'],
                               general_settings['max_time'],
                               general_settings['max_f_evals'],
                               general_settings['verbose'],
                               general_settings['verbose_interspace'],
                               general_settings['plot_pareto_front'],
                               general_settings['plot_pareto_solutions'],
                               general_settings['plot_dpi'],
                               sparsity_settings['sparsity_tol'],
                               MOIHT_settings['theta_tol'],
                               MOSPD_settings['xy_diff'],
                               MOSPD_settings['max_inner_iter_count'],
                               MOSPD_settings['max_mopgd_iters'],
                               MOSPD_settings['tau_0'],
                               MOSPD_settings['tau_0_inc_factor'],
                               MOSPD_settings['tau_inc_factor'],
                               MOSPD_settings['epsilon_0'],
                               MOSPD_settings['epsilon_0_dec_factor'],
                               MOSPD_settings['epsilon_dec_factor'],
                               DDS_settings['verbose'],
                               DDS_settings['method'],
                               DDS_settings['feas_tol'],
                               ALS_settings,
                               Ref_settings)

        elif algorithm_name == 'F-MOSPD-MOIHT':
            MOSPD_settings = algorithms_settings['MOSPD']
            MOIHT_settings = algorithms_settings['MOIHT']
            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']
            Ref_settings = kwargs['Ref_settings']
            sparsity_settings = kwargs['sparsity_settings']

            return F_MOSPD_MOIHT(general_settings['max_iter'],
                                 general_settings['max_time'],
                                 general_settings['max_f_evals'],
                                 general_settings['verbose'],
                                 general_settings['verbose_interspace'],
                                 general_settings['plot_pareto_front'],
                                 general_settings['plot_pareto_solutions'],
                                 general_settings['plot_dpi'],
                                 sparsity_settings['sparsity_tol'],
                                 MOIHT_settings['theta_tol'],
                                 MOSPD_settings['xy_diff'],
                                 MOSPD_settings['max_inner_iter_count'],
                                 MOSPD_settings['max_mopgd_iters'],
                                 MOSPD_settings['tau_0'],
                                 MOSPD_settings['tau_0_inc_factor'],
                                 MOSPD_settings['tau_inc_factor'],
                                 MOSPD_settings['epsilon_0'],
                                 MOSPD_settings['epsilon_0_dec_factor'],
                                 MOSPD_settings['epsilon_dec_factor'],
                                 DDS_settings['verbose'],
                                 DDS_settings['method'],
                                 DDS_settings['feas_tol'],
                                 ALS_settings,
                                 Ref_settings)

        elif algorithm_name == 'GSS':
            GSS_settings = algorithms_settings[algorithm_name]
            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']
            Ref_settings = kwargs['Ref_settings']
            sparsity_settings = kwargs['sparsity_settings']

            return GSS(general_settings['max_iter'],
                       general_settings['max_time'],
                       general_settings['max_f_evals'],
                       general_settings['verbose'],
                       general_settings['verbose_interspace'],
                       general_settings['plot_pareto_front'],
                       general_settings['plot_pareto_solutions'],
                       general_settings['plot_dpi'],
                       sparsity_settings['sparsity_tol'],
                       GSS_settings['x_diff'],
                       GSS_settings['L_BFGS_verbosity_level'],
                       GSS_settings['L_BFGS_gtol'],
                       DDS_settings['verbose'],
                       DDS_settings['method'],
                       DDS_settings['feas_tol'],
                       Ref_settings, ALS_settings)

        elif algorithm_name == 'F-GSS':
            GSS_settings = algorithms_settings['GSS']
            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']
            Ref_settings = kwargs['Ref_settings']
            sparsity_settings = kwargs['sparsity_settings']

            return F_GSS(general_settings['max_iter'],
                         general_settings['max_time'],
                         general_settings['max_f_evals'],
                         general_settings['verbose'],
                         general_settings['verbose_interspace'],
                         general_settings['plot_pareto_front'],
                         general_settings['plot_pareto_solutions'],
                         general_settings['plot_dpi'],
                         sparsity_settings['sparsity_tol'],
                         GSS_settings['x_diff'],
                         GSS_settings['L_BFGS_verbosity_level'],
                         GSS_settings['L_BFGS_gtol'],
                         DDS_settings['verbose'],
                         DDS_settings['method'],
                         DDS_settings['feas_tol'],
                         Ref_settings, ALS_settings)

        elif algorithm_name == 'MIP':
            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']
            Ref_settings = kwargs['Ref_settings']
            sparsity_settings = kwargs['sparsity_settings']

            return MIP(general_settings['max_iter'],
                       general_settings['max_time'],
                       general_settings['max_f_evals'],
                       general_settings['verbose'],
                       general_settings['verbose_interspace'],
                       general_settings['plot_pareto_front'],
                       general_settings['plot_pareto_solutions'],
                       general_settings['plot_dpi'],
                       sparsity_settings['sparsity_tol'],
                       DDS_settings['verbose'],
                       DDS_settings['method'],
                       DDS_settings['feas_tol'],
                       Ref_settings,
                       ALS_settings)

        elif algorithm_name == 'F-MIP':
            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']
            Ref_settings = kwargs['Ref_settings']
            sparsity_settings = kwargs['sparsity_settings']

            return F_MIP(general_settings['max_iter'],
                         general_settings['max_time'],
                         general_settings['max_f_evals'],
                         general_settings['verbose'],
                         general_settings['verbose_interspace'],
                         general_settings['plot_pareto_front'],
                         general_settings['plot_pareto_solutions'],
                         general_settings['plot_dpi'],
                         sparsity_settings['sparsity_tol'],
                         DDS_settings['verbose'],
                         DDS_settings['method'],
                         DDS_settings['feas_tol'],
                         Ref_settings,
                         ALS_settings)

        else:
            raise NotImplementedError

        return algorithm
