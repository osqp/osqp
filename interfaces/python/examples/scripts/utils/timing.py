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
        out_vec = np.array([x.median for x in stats])
    elif statistics_name == 'mean':
        out_vec = np.array([x.mean for x in stats])
    elif statistics_name == 'total':
        out_vec = np.array([x.total for x in stats])
    elif statistics_name == 'max':
        out_vec = np.array([x.max for x in stats])

    return out_vec
