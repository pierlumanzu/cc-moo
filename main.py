import numpy as np
from datetime import datetime
import tensorflow as tf
import glob
import os

from algorithms.algorithm_utils.Graphical_Plot import Graphical_Plot
from algorithms.Algorithm_Factory import Algorithm_Factory

from problems.Problem_Factory import Problem_Factory

from general_utils.Args_Utils import print_parameters, args_preprocessing, args_file_creation
from general_utils.Management_Utils import folder_initialization, execution_time_file_initialization, write_in_execution_time_file, write_results_in_csv_file, save_plots
from general_utils.Pareto_Utils import pointsInitialization, pointsPostprocessing
from general_utils.Progress_Bar import Progress_Bar

from parser_management import get_args

tf.compat.v1.disable_eager_execution()

args = get_args()
print_parameters(args)
algorithms_names, prob_settings, general_settings, algorithms_settings, DDS_settings, ALS_settings, Ref_settings, sparsity_settings = args_preprocessing(args)

if prob_settings['prob_type'] == 'QP':
    assert len(prob_settings['dat_prob_paths']) == 1
    if 'pkl' not in prob_settings['dat_prob_paths'][0]:
        dat_prob_paths = glob.glob(os.path.join(prob_settings['dat_prob_paths'][0], '*.pkl'))
        dat_prob_paths_new = []
        for dat_prob_path in dat_prob_paths:
            dat_prob_path = dat_prob_path.replace('_a' if '_a' in dat_prob_path else '_b', '')
            if dat_prob_path not in dat_prob_paths_new:
                dat_prob_paths_new.append(dat_prob_path)
        dat_prob_paths = dat_prob_paths_new
    else:
        dat_prob_paths = prob_settings['dat_prob_paths']
else:
    dat_prob_paths = prob_settings['dat_prob_paths']

print('N° algorithms: ', len(algorithms_names))
print('N° problems: ', len(dat_prob_paths) * len(prob_settings['L']) * len(prob_settings['s']))
print('N° Seeds: ', len(general_settings['seeds']))
print()

date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if general_settings['verbose']:
    progress_bar = Progress_Bar(len(algorithms_names) * len(dat_prob_paths) * len(prob_settings['L']) * len(prob_settings['s']) * len(general_settings['seeds']))
    progress_bar.showBar()

for seed in general_settings['seeds']:
    print()
    print('Seed:', str(seed))

    if general_settings['general_export']:
        folder_initialization(seed, date, algorithms_names)
        args_file_creation(date, seed, args)
        execution_time_file_initialization(seed, date, algorithms_names)

    for L in prob_settings['L']:
        print()
        print('L: ', str(L) if L > 0 else 'Not Specified')

        for s in prob_settings['s']:
            print()
            print('s: ', s)

            for dat_prob_path in dat_prob_paths:

                for index_algorithm, algorithm_name in enumerate(algorithms_names):
                    print()
                    print('Algorithm: ', algorithm_name)

                    session = tf.compat.v1.Session()
                    with session.as_default():

                        problem_instance = Problem_Factory.get_problem(prob_settings['prob_type'],
                                                                       dat_prob_path=dat_prob_path,
                                                                       s=s,
                                                                       L=L if L > 0 else None,
                                                                       L_inc_factor=prob_settings['L_inc_factor'])

                        print()
                        if prob_settings['prob_type'] == 'LR':
                            print('Dataset: ', problem_instance.name())
                        elif prob_settings['prob_type'] == 'QP':
                            print('Problem: ', problem_instance.name())

                        print()
                        print('N: ', str(problem_instance.n))
                        print()

                        np.random.seed(seed=seed)
                        initial_p_list, initial_f_list, n_initial_points = pointsInitialization(problem_instance, algorithm_name, seed=seed)

                        algorithm = Algorithm_Factory.get_algorithm(algorithm_name,
                                                                    general_settings=general_settings,
                                                                    algorithms_settings=algorithms_settings,
                                                                    DDS_settings=DDS_settings,
                                                                    ALS_settings=ALS_settings,
                                                                    Ref_settings=Ref_settings,
                                                                    sparsity_settings=sparsity_settings)

                        problem_instance.evaluateFunctions(initial_p_list[0, :])
                        problem_instance.evaluateFunctionsJacobian(initial_p_list[0, :])

                        if algorithm_name not in ['GSS', 'MIP', 'F-MIP', 'F-GSS']:
                            p_list, f_list, elapsed_time = algorithm.search(initial_p_list, initial_f_list, problem_instance)
                        else:
                            p_list, f_list, elapsed_time = algorithm.search(initial_p_list, initial_f_list, problem_instance, lams=np.array([2 ** ((2*i - 2 * problem_instance.n + 1) / 2) for i in range(2 * problem_instance.n)], dtype=float))

                        filtered_p_list, filtered_f_list = pointsPostprocessing(p_list, f_list, problem_instance)

                        if general_settings['plot_pareto_front']:
                            graphical_plot = Graphical_Plot(general_settings['plot_pareto_solutions'], general_settings['plot_dpi'])
                            graphical_plot.showFigure(filtered_p_list, filtered_f_list, hold_still=True)
                            graphical_plot.closeFigure()

                        if general_settings['general_export']:
                            write_in_execution_time_file(seed, date, algorithm_name, problem_instance, elapsed_time)
                            write_results_in_csv_file(filtered_p_list, filtered_f_list, seed, date, algorithm_name, problem_instance, export_pareto_solutions=general_settings['export_pareto_solutions'])
                            save_plots(filtered_p_list, filtered_f_list, seed, date, algorithm_name, problem_instance, general_settings['export_pareto_solutions'], general_settings['plot_dpi'])

                        if general_settings['verbose']:
                            progress_bar.incrementCurrentValue()
                            progress_bar.showBar()

                        tf.compat.v1.reset_default_graph()
                        session.close()
