"""
Timing statistics utils
"""
import numpy as np


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
