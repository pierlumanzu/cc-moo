import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable printing of tensorflow information and warnings.

import numpy as np
from datetime import datetime
import tensorflow as tf
import glob

from nsma.algorithms.algorithm_utils.graphical_plot import GraphicalPlot

from algorithms.algorithm_factory import Algorithm_Factory
from problems.problem_factory import Problem_Factory
from general_utils.args_utils import print_parameters, args_preprocessing, args_file_creation
from general_utils.management_utils import folder_initialization, execution_time_file_initialization, write_in_execution_time_file, write_results_in_csv_file, save_plots
from general_utils.pareto_utils import pointsInitialization, pointsPostprocessing
from general_utils.progress_bar import ProgressBarWrapper
from parser_management import get_args


if __name__ == '__main__':

    tf.compat.v1.disable_eager_execution()

    args = get_args()

    print_parameters(args)
    single_point_methods_names, refiner, prob_settings, general_settings, sparsity_settings, algorithms_settings, DDS_settings, ALS_settings = args_preprocessing(args)

    if prob_settings['prob_type'] == 'QP':

        if os.path.isdir(prob_settings['prob_path']):
            prob_paths = glob.glob(os.path.join(prob_settings['prob_path'][0], '*.pkl'))
            for idx_prob_path in range(len(prob_paths)):
                prob_paths[idx_prob_path] = prob_paths[idx_prob_path].replace('_a' if '_a' in prob_paths[idx_prob_path] else '_b', '')
            prob_paths = list(set(prob_paths))
        
        else:
            assert os.path.exists(prob_settings['prob_path'].replace('_a' if '_a' in prob_settings['prob_path'] else '_b', '_b' if '_a' in prob_settings['prob_path'] else '_a'))
            prob_paths = [prob_settings['prob_path'].replace('_a' if '_a' in prob_settings['prob_path'] else '_b', '')]
    
    else:
        raise NotImplementedError

    print('N° Single Point Methods: ', len(single_point_methods_names))
    print('Refiner: ', refiner)
    print('N° Problems: ', len(prob_paths))
    print('N° Upper Bounds for Cardinality Constraint: ', len(sparsity_settings['s']))
    print('N° Seeds: ', len(general_settings['seeds']))
    print()


    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if general_settings['verbose']:
        progress_bar = ProgressBarWrapper(len(single_point_methods_names) * len(prob_paths) * len(sparsity_settings['s']) * len(general_settings['seeds']))
        progress_bar.show_bar()

    for seed in general_settings['seeds']:
        
        print()
        print('Seed: ', str(seed))

        if general_settings['general_export']:
            folder_initialization(date, seed, single_point_methods_names, refiner)
            args_file_creation(date, seed, args)
            execution_time_file_initialization(date, seed, single_point_methods_names, refiner)

        for L in prob_settings['L']:
            print()
            print('L: ', str(L) if L > 0 else 'Not Specified')

            for prob_path in prob_paths:
                for s in sparsity_settings['s']:
                    for idx_single_point_method, single_point_method_name in enumerate(single_point_methods_names):

                        session = tf.compat.v1.Session()
                        with session.as_default():

                            problem_instance = Problem_Factory.get_problem(prob_settings['prob_type'],
                                                                           prob_path=prob_path,
                                                                           s=s,
                                                                           sparsity_tol=sparsity_settings['sparsity_tol'])
                            
                            if not idx_single_point_method:
                                print()
                                print('Problem Type: ', problem_instance.family_name())
                                print('Problem Path: ', problem_instance.name())
                                print('Problem Dimensionality: ', problem_instance.n)
                                print('Upper Bound for Cardinality Constraint: ', sparsity_settings['s'])

                            algorithm = Algorithm_Factory.get_algorithm(single_point_method_name,
                                                                        general_settings=general_settings,
                                                                        algorithms_settings=algorithms_settings,
                                                                        refiner=refiner,
                                                                        DDS_settings=DDS_settings,
                                                                        ALS_settings=ALS_settings)

                            print()
                            print('Single Point Method: ', single_point_method_name)
                            print('Refiner: ', refiner)

                            np.random.seed(seed=seed)
                            initial_p_list, initial_f_list, n_initial_points = pointsInitialization(problem_instance, seed)

                            problem_instance.evaluate_functions(initial_p_list[0, :])
                            problem_instance.evaluate_functions_jacobian(initial_p_list[0, :])       

                            p_list, f_list, elapsed_time = algorithm.search(initial_p_list, initial_f_list, problem_instance)
                            
                            final_p_list, final_f_list = pointsPostprocessing(p_list, f_list, problem_instance)

                            if general_settings['plot_pareto_front']:
                                graphical_plot = GraphicalPlot(general_settings['plot_pareto_solutions'], general_settings['plot_dpi'])
                                graphical_plot.show_figure(final_p_list, final_f_list, hold_still=True)
                                graphical_plot.close_figure()

                            if general_settings['general_export']:
                                write_in_execution_time_file(date, seed, single_point_method_name, refiner, problem_instance, elapsed_time)
                                write_results_in_csv_file(date, seed, single_point_method_name, refiner, problem_instance, final_p_list, final_f_list, export_pareto_solutions=general_settings['export_pareto_solutions'])
                                save_plots(date, seed, single_point_method_name, refiner, problem_instance, final_p_list, final_f_list, general_settings['export_pareto_solutions'], general_settings['plot_dpi'])

                            if general_settings['verbose']:
                                progress_bar.increment_current_value()
                                progress_bar.show_bar()

                            tf.compat.v1.reset_default_graph()
                            session.close()
