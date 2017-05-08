import scipy.sparse as spa
import numpy as np
from collections import namedtuple


def load_pendulum_data():
    '''
    Load MPC pendulum example
    https://github.com/ferreau/mpcBenchmarking/blob/master/benchmarks/Benchmark_pendulum.m
    '''

    # Problem setup (sampling interval Ts = 0.05s)
    problem = namedtuple("problem", "A B R Q QN umin umax xmin xmax " +
                                    "T tmin tmax N x0 name")
    problem.A = spa.csc_matrix([[1.001, -0.05, -0.001],
                                [-0.05, 1.003, 0.05],
                                [-0.001, 0.05, 1.001]])
    problem.B = spa.csc_matrix([[0.], [0.001], [0.05]])
    problem.R = spa.diags([0.1])
    problem.Q = spa.eye(3)
    problem.QN = spa.csc_matrix([[113.4637, -95.4971, -21.0556],
                                 [-95.4971, 99.5146, 23.3756],
                                 [-21.0556, 23.3756, 12.1868]])
    problem.umin = np.array([-1.25])
    problem.umax = np.array([1.25])
    problem.xmin = []
    problem.xmax = []
    problem.T = []                  # terminal constraints
    problem.tmin = []
    problem.tmax = []
    problem.x0 = 0.6*np.ones(3)
    problem.name = 'pendulum'

    return problem
