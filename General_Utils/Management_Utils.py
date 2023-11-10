import os
import numpy as np

from Algorithms.Algorithm_Utils.Graphical_Plot import Graphical_Plot


def makeFolder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def folderInitialization(seed, date, algorithms_names):
    assert os.path.exists(os.path.join('Execution_Outputs'))

    folders = ['Execution_Times', 'Csv', 'Plot']

    path = os.path.join('Execution_Outputs', date)
    makeFolder(path)

    path = os.path.join(path, str(seed))
    makeFolder(path)

    for index_folder, folder in enumerate(folders):
        makeFolder(os.path.join(path, folder))
        if index_folder >= 1:
            for algorithm_name in algorithms_names:
                makeFolder(os.path.join(path, folder, algorithm_name))


def executionTimeFileInitialization(seed, date, algorithms_names):
    for algorithm_name in algorithms_names:
        execution_time_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Execution_Times', '{}.txt'.format(algorithm_name)), 'w')
        execution_time_file.close()


def writeInExecutionTimeFile(seed, date, algorithm_name, problem_instance, elapsed_time):
    execution_time_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Execution_Times', '{}.txt'.format(algorithm_name)), 'a')
    execution_time_file.write('Problem: ' + problem_instance.name() + '    N: ' + str(problem_instance.n) + '    Time: ' + str(elapsed_time) + '\n')
    execution_time_file.close()


def writeResultsInCsvFile(p_list, f_list, seed, date, algorithm_name, problem, export_pareto_solutions=False):
    assert len(p_list) == len(f_list)
    n = p_list.shape[1]

    f_list_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Csv', algorithm_name, '{}_{}_{}_{}_pareto_front.csv'.format(problem.name(), n, str(round(np.max(problem.L), 3)), str(problem.s))), 'w')
    if len(f_list):
        for i in range(f_list.shape[0]):
            f_list_file.write(';'.join([str(el) for el in f_list[i, :]]) + '\n')
    f_list_file.close()

    if export_pareto_solutions:
        p_list_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Csv', algorithm_name, '{}_{}_{}_{}_pareto_solutions.csv'.format(problem.name(), n, str(round(np.max(problem.L), 3)), str(problem.s))), 'w')
        if len(p_list):
            for i in range(p_list.shape[0]):
                p_list_file.write(';'.join([str(el) for el in p_list[i, :]]) + '\n')
        p_list_file.close()


def savePlots(p_list, f_list, seed, date, algorithm_name, problem_instance, export_pareto_solutions, plot_dpi):
    assert len(p_list) == len(f_list)

    graphical_plot = Graphical_Plot(export_pareto_solutions, plot_dpi)
    graphical_plot.saveFigure(p_list, f_list, os.path.join('Execution_Outputs', date, str(seed), 'Plot'), algorithm_name, problem_instance)
