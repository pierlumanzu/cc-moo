import argparse
import sys


def getArgs():

    parser = argparse.ArgumentParser(description='Algorithms for Multi-Objective Optimization')

    parser.add_argument('--algs', type=str, help='Algorithms', nargs='+', choices=['F-MOIHT', 'F-MOSPD', 'F-MOSPD-MOIHT', 'F-MIP', 'F-GSS', 'MOIHT', 'MOSPD', 'MOSPD-MOIHT', 'MIP', 'GSS'])

    parser.add_argument('--prob_type', type=str, help='Problems to evaluate', choices=['LR', 'QP'])

    parser.add_argument('--dat_prob_paths', help='Dataset/Problem path', type=str, nargs='+')

    parser.add_argument('--L', help='L-stationarity parameter', type=float, default=[-1], nargs='+')

    parser.add_argument('--L_inc_factor', help='L increment factor', type=float, default=1.1)

    parser.add_argument('--s', help='Solutions cardinality', type=int, nargs='+')

    parser.add_argument('--seeds', help='Seeds', nargs='+', type=int, default=[16007])

    parser.add_argument('--max_iter', help='Maximum number of iterations', default=None, type=int)

    parser.add_argument('--max_time', help='Maximum number of elapsed minutes per problem', default=None, type=float)

    parser.add_argument('--max_f_evals', help='Maximum number of function evaluations', default=None, type=int)

    parser.add_argument('--verbose', help='Verbose during the iterations', action='store_true', default=False)

    parser.add_argument('--verbose_interspace', help='Used interspace in the verbose (Requirements: verbose activated)', default=20, type=int)

    parser.add_argument('--plot_pareto_front', help='Plot Pareto front', action='store_true', default=False)

    parser.add_argument('--plot_pareto_solutions', help='Plot Pareto solutions (Requirements: plot_pareto_front activated; n in [2, 3])', action='store_true', default=False)

    parser.add_argument('--general_export', help='Export fronts (including plots), execution times and arguments files', action='store_true', default=False)

    parser.add_argument('--export_pareto_solutions', help='Export pareto solutions, including the plots if n in [2, 3] (Requirements: general_export activated)', action='store_true', default=False)

    parser.add_argument('--plot_dpi', help='DPI of the saved plots (Requirements: general_export activated)', default=100, type=int)

    ####################################################
    ### ONLY FOR Gurobi ###
    ####################################################

    parser.add_argument('--Gurobi_method', help='Gurobi parameter -- Used method', default=1, type=int)

    parser.add_argument('--Gurobi_verbose', help='Gurobi parameter -- Verbose during the Gurobi iterations', action='store_true', default=False)

    parser.add_argument('--Gurobi_feas_tol', help='Gurobi parameter -- Feasibility tolerance', default=1e-7, type=float)

    ####################################################
    ### ONLY FOR Sparsity ###
    ####################################################

    parser.add_argument('--sparse_tol', help='Sparsity parameter -- Tolerance on sparsity', default=1e-7, type=float)

    ####################################################
    ### ONLY FOR F-MOIHT ###
    ####################################################

    parser.add_argument('--F_MOIHT_theta_for_stationarity', help='F-MOIHT parameter -- Theta for Pareto stationarity', default=-1e-7, type=float)

    ####################################################
    ### ONLY FOR MOIHT ###
    ####################################################

    parser.add_argument('--MOIHT_theta_for_stationarity', help='MOIHT parameter -- Theta for Pareto stationarity', default=-1e-7, type=float)

    ####################################################
    ### ONLY FOR MOSPD ###
    ####################################################

    parser.add_argument('--MOSPD_xy_diff', help='MOSPD parameter -- Difference between current x and y', default=1e-3, type=float)

    parser.add_argument('--MOSPD_max_inner_iter_count', help='MOSPD parameter -- Maximum number of iterations in the inner loop', default=20, type=int)

    parser.add_argument('--MOSPD_max_mopgd_iters', help='MOSPD parameter -- Maximum number of mopgd iterations', default=10, type=int)

    parser.add_argument('--MOSPD_tau_0', help='MOSPD parameter -- Initial value for tau', default=1, type=float)

    parser.add_argument('--MOSPD_tau_0_inc_factor', help='MOSPD parameter -- Increment factor for initial value of tau', default=1e5, type=float)

    parser.add_argument('--MOSPD_tau_inc_factor', help='MOSPD parameter -- Increment factor for tau', default=1.3, type=float)

    parser.add_argument('--MOSPD_epsilon_0', help='MOSPD parameter -- Initial value for epsilon', default=-1e-5, type=float)

    parser.add_argument('--MOSPD_epsilon_0_dec_factor', help='MOSPD parameter -- Decrement factor for initial value of epsilon', default=1e-2, type=float)

    parser.add_argument('--MOSPD_epsilon_dec_factor', help='MOSPD parameter -- Decrement factor for epsilon', default=0.9, type=float)

    ####################################################
    ### ONLY FOR GSS ###
    ####################################################

    parser.add_argument('--GSS_x_diff', help='GSS parameter -- Difference between current x and previous x', default=1e-3, type=float)

    parser.add_argument('--GSS_L_BFGS_verbosity_level', help='GSS parameter -- Set verbosity level of L-BFGS', default=0, type=int)

    parser.add_argument('--GSS_L_BFGS_gtol', help='GSS parameter -- Set tolerance on gradient of L-BFGS', default=1e-7, type=float)

    ####################################################
    ### ONLY FOR Refiner_FPGA ###
    ####################################################

    parser.add_argument('--R_FPGA_theta_for_stationarity', help='Refiner_FPGA parameter -- Theta for Pareto stationarity', default=-1e-7, type=float)

    ####################################################
    ### ONLY FOR Refiner_MOPG ###
    ####################################################

    parser.add_argument('--R_MOPG_theta_for_stationarity', help='Refiner_MOPG parameter -- Theta for Pareto stationarity', default=-1e-7, type=float)

    ####################################################
    ### ONLY FOR ArmijoTypeLineSearch ###
    ####################################################

    parser.add_argument('--ALS_alpha_0', help='ALS parameter -- Initial step size', default=1, type=float)

    parser.add_argument('--ALS_delta', help='ALS parameter -- Coefficient for step size contraction', default=0.5, type=float)

    parser.add_argument('--ALS_beta', help='ALS parameter -- Beta', default=1.0e-4, type=float)

    parser.add_argument('--ALS_min_alpha', help='ALS parameter -- Min alpha', default=1.0e-14, type=float)

    return parser.parse_args(sys.argv[1:])

