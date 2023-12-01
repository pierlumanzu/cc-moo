import os
import numpy as np

from nsma.algorithms.algorithm_utils.graphical_plot import GraphicalPlot

from problems.extended_problem import ExtendedProblem

def make_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def folder_initialization(date: str, seed: int, single_methods_names: list, approach: str):
    assert os.path.exists(os.path.join('Execution_Outputs'))

    folders = ['Execution_Times', 'Csv', 'Plot']

    path = os.path.join('Execution_Outputs', date)
    make_folder(path)

    path = os.path.join(path, str(seed))
    make_folder(path)

    for index_folder, folder in enumerate(folders):
        make_folder(os.path.join(path, folder))
        if index_folder >= 1:
            for single_method_name in single_methods_names:
                make_folder(os.path.join(path, folder, '{}-{}'.format(single_method_name, approach)))


def execution_time_file_initialization(date: str, seed: int, single_methods_names: list, approach: str):
    for single_method_name in single_methods_names:
        execution_time_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Execution_Times', '{}-{}.txt'.format(single_method_name, approach)), 'w')
        execution_time_file.close()


def write_in_execution_time_file(date: str, seed: int, single_method_name: str, approach: str, problem_instance: ExtendedProblem, elapsed_time: float):
    execution_time_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Execution_Times', '{}-{}.txt'.format(single_method_name, approach)), 'a')
    execution_time_file.write('Problem: ' + problem_instance.name() + '    N: ' + str(problem_instance.n) + '    Time: ' + str(elapsed_time) + '\n')
    execution_time_file.close()


def write_results_in_csv_file(date: str, seed: int, single_method_name: str, approach: str, problem_instance: ExtendedProblem, p_list: np.array, f_list: np.array, export_pareto_solutions=False):
    assert len(p_list) == len(f_list)
    n = p_list.shape[1]

    f_list_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Csv', '{}-{}'.format(single_method_name, approach), '{}_{}_{}_{}_pareto_front.csv'.format(problem_instance.name(), n, str(round(np.max(problem_instance.L), 3)), str(problem_instance.s))), 'w')
    if len(f_list):
        for i in range(f_list.shape[0]):
            f_list_file.write(';'.join([str(el) for el in f_list[i, :]]) + '\n')
    f_list_file.close()

    if export_pareto_solutions:
        p_list_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Csv', '{}-{}'.format(single_method_name, approach), '{}_{}_{}_{}_pareto_solutions.csv'.format(problem_instance.name(), n, str(round(np.max(problem_instance.L), 3)), str(problem_instance.s))), 'w')
        if len(p_list):
            for i in range(p_list.shape[0]):
                p_list_file.write(';'.join([str(el) for el in p_list[i, :]]) + '\n')
        p_list_file.close()


def save_plots(date: str, seed: int, single_method_name: str, approach: str, problem_instance: ExtendedProblem, p_list: np.array, f_list: np.array, export_pareto_solutions, plot_dpi):
    assert len(p_list) == len(f_list)

    graphical_plot = Graphical_Plot(export_pareto_solutions, plot_dpi)
    graphical_plot.saveFigure(p_list, f_list, os.path.join('Execution_Outputs', date, str(seed), 'Plot'), algorithm_name, problem_instance)
