import os


def printParameters(args):
    if args.verbose:
        print()
        print('Parameters')
        print()

        for key in args.__dict__.keys():
            print(key.ljust(args.verbose_interspace), args.__dict__[key])
        print()


def checkArgs(args):

    for L in args.L:
        assert L > 0 or L == - 1
    assert args.L_inc_factor > 1

    for s in args.s:
        assert s > 0

    for seed in args.seeds:
        assert seed > 0

    if args.max_iter is not None:
        assert args.max_iter > 0
    if args.max_time is not None:
        assert args.max_time > 0
    if args.max_f_evals is not None:
        assert args.max_f_evals > 0

    assert args.verbose_interspace >= 1
    assert args.plot_dpi >= 1

    assert args.Gurobi_feas_tol > 0

    assert args.sparse_tol > 0

    assert args.F_MOIHT_theta_for_stationarity <= 0

    assert args.MOIHT_theta_for_stationarity <= 0

    assert args.MOSPD_xy_diff >= 0
    assert args.MOSPD_max_inner_iter_count > 0
    assert args.MOSPD_max_mopgd_iters > 0
    assert args.MOSPD_tau_0 >= 0
    assert args.MOSPD_tau_0_inc_factor > args.MOSPD_tau_inc_factor
    assert args.MOSPD_tau_inc_factor > 1
    assert args.MOSPD_epsilon_0 <= 0
    assert 0 <= args.MOSPD_epsilon_0_dec_factor < args.MOSPD_epsilon_dec_factor
    assert 0 < args.MOSPD_epsilon_dec_factor < 1

    assert args.GSS_x_diff >= 0
    assert args.GSS_L_BFGS_verbosity_level >= 0
    assert args.GSS_L_BFGS_gtol >= 0

    assert args.R_FPGA_theta_for_stationarity <= 0

    assert args.R_MOPG_theta_for_stationarity <= 0

    assert args.ALS_alpha_0 > 0
    assert 0 < args.ALS_delta < 1
    assert 0 < args.ALS_beta < 1
    assert args.ALS_min_alpha > 0


def argsPreprocessing(args):
    checkArgs(args)

    algorithms_names = []

    if 'F-MOIHT' in args.algs:
        algorithms_names.append('F-MOIHT')

    if 'F-MOSPD' in args.algs:
        algorithms_names.append('F-MOSPD')

    if 'F-MOSPD-MOIHT' in args.algs:
        algorithms_names.append('F-MOSPD-MOIHT')

    if 'F-MIP' in args.algs:
        algorithms_names.append('F-MIP')

    if 'F-GSS' in args.algs:
        algorithms_names.append('F-GSS')

    if 'MOIHT' in args.algs:
        algorithms_names.append('MOIHT')

    if 'MOSPD' in args.algs:
        algorithms_names.append('MOSPD')

    if 'MOSPD-MOIHT' in args.algs:
        algorithms_names.append('MOSPD-MOIHT')

    if 'GSS' in args.algs:
        algorithms_names.append('GSS')

    if 'MIP' in args.algs:
        algorithms_names.append('MIP')

    if len(algorithms_names) == 0:
        raise Exception('You must insert a set of algorithms')

    general_settings = {'seeds': args.seeds,
                        'max_iter': args.max_iter,
                        'max_time': args.max_time,
                        'max_f_evals': args.max_f_evals,
                        'verbose': args.verbose,
                        'verbose_interspace': args.verbose_interspace,
                        'plot_pareto_front': args.plot_pareto_front,
                        'plot_pareto_solutions': args.plot_pareto_solutions,
                        'general_export': args.general_export,
                        'export_pareto_solutions': args.export_pareto_solutions,
                        'plot_dpi': args.plot_dpi}

    prob_settings = {'prob_type': args.prob_type,
                     'dat_prob_paths': args.dat_prob_paths,
                     'L': args.L,
                     'L_inc_factor': args.L_inc_factor,
                     's': args.s}

    sparsity_settings = {'sparse_tol': args.sparse_tol}

    F_MOIHT_settings = {'theta_for_stationarity': args.F_MOIHT_theta_for_stationarity}

    MOIHT_settings = {'theta_for_stationarity': args.MOIHT_theta_for_stationarity}

    MOSPD_settings = {'xy_diff': args.MOSPD_xy_diff,
                      'max_inner_iter_count': args.MOSPD_max_inner_iter_count,
                      'max_mopgd_iters': args.MOSPD_max_mopgd_iters,
                      'tau_0': args.MOSPD_tau_0,
                      'tau_0_inc_factor': args.MOSPD_tau_0_inc_factor,
                      'tau_inc_factor': args.MOSPD_tau_inc_factor,
                      'epsilon_0': args.MOSPD_epsilon_0,
                      'epsilon_0_dec_factor': args.MOSPD_epsilon_0_dec_factor,
                      'epsilon_dec_factor': args.MOSPD_epsilon_dec_factor}

    GSS_settings = {'x_diff': args.GSS_x_diff,
                    'L_BFGS_verbosity_level': args.GSS_L_BFGS_verbosity_level,
                    'L_BFGS_gtol': args.GSS_L_BFGS_gtol}

    algorithms_settings = {'F-MOIHT': F_MOIHT_settings,
                           'MOIHT': MOIHT_settings,
                           'MOSPD': MOSPD_settings,
                           'GSS': GSS_settings}

    R_FPGA_settings = {'theta_for_stationarity': args.R_FPGA_theta_for_stationarity}

    R_MOPG_settings = {'theta_for_stationarity': args.R_MOPG_theta_for_stationarity}

    Ref_settings = {'FPGA': R_FPGA_settings,
                    'MOPG': R_MOPG_settings}

    DDS_settings = {'verbose': args.Gurobi_verbose,
                    'feas_tol': args.Gurobi_feas_tol,
                    'method': args.Gurobi_method}

    ALS_settings = {'alpha_0': args.ALS_alpha_0,
                    'delta': args.ALS_delta,
                    'beta': args.ALS_beta,
                    'min_alpha': args.ALS_min_alpha}

    return algorithms_names, prob_settings, general_settings, algorithms_settings, DDS_settings, ALS_settings, Ref_settings, sparsity_settings


def argsFileCreation(date, num_trials, args):
    if args.general_export:
        args_file = open(os.path.join('Execution_Outputs', date, str(num_trials), 'params.csv'), 'w')
        for key in args.__dict__.keys():
            args_file.write('{};{}\n'.format(key, args.__dict__[key]))
        args_file.close()


