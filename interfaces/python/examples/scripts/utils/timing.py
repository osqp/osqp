"""
Timing statistics utils
"""
import numpy as np
import os
import pandas as pd

class Statistics(object):
    """
    Solver statistics
    """
    def __init__(self, x):
        self.x = x
        self.mean = np.mean(x)
        self.median = np.median(x)
        self.max = np.max(x)
        self.min = np.min(x)
        self.total = np.sum(x)


def gen_stats_array_vec(statistics_name, stats):
    if statistics_name == 'median':
        out_vec = np.array([x.median for x in stats if x.median != 0])
        stat_list = [x.median for x in stats]
    elif statistics_name == 'mean':
        out_vec = np.array([x.mean for x in stats if x.mean != 0])
        stat_list = [x.mean for x in stats]
    elif statistics_name == 'total':
        out_vec = np.array([x.total for x in stats if x.total != 0])
        stat_list = [x.total for x in stats]
    elif statistics_name == 'max':
        out_vec = np.array([x.max for x in stats if x.max != 0])
        stat_list = [x.max for x in stats]

    idx_vec = np.array([stat_list.index(x) for x in out_vec if x in stat_list])
    
    return out_vec, idx_vec


def compute_statistics_dataframe(statistics_name, timings_dict, n_vec):
    '''
    Compute statistics dataframe from timings dictionary
    '''

    df = pd.DataFrame({'n': n_vec}) 

    for (solver_name, solver_timings) in timings_dict.items():
        stats_array, idx_val = gen_stats_array_vec(statistics_name,
                                                  solver_timings)
        df['%s %s' % (solver_name, statistics_name)] = stats_array

    return df


def store_timings(example_name, timings_dict, n_vec, stats_name):
    #  comparison_table = pd.DataFrame(timings_dict)
    #  comparison_table = comparison_table[cols]  # Sort table columns

    comparison_table = compute_statistics_dataframe(stats_name, 
                                                    timings_dict,
                                                    n_vec)
    
    data_dir = 'scripts/%s/data' % example_name
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
  
    comparison_table.to_csv('%s/timings_%s.csv' % (data_dir, stats_name),
                            index=False)

     # Converting results to latex table and storing them to a file
    formatter = lambda x: '%1.2f' % x
    latex_table = comparison_table.to_latex(header=True, index=False,
                                            float_format=formatter)
    f = open('%s/timings_%s.tex' % (data_dir, stats_name), 'w')
    f.write(latex_table)
    f.close()
