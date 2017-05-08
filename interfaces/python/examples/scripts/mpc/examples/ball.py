import scipy.sparse as spa
import numpy as np
from collections import namedtuple


def load_ball_data():
    '''
    Load MPC ball on plate example
    https://github.com/ferreau/mpcBenchmarking/blob/master/benchmarks/Benchmark_ballOnPlate.m
    '''

    # Problem setup
    problem = namedtuple("problem", "A B R Q QN umin umax xmin xmax " +
                                    "T tmin tmax N x0 name")
    problem.A = spa.csc_matrix([[1., 0.01],
                                [0., 1.]])
    problem.B = spa.csc_matrix([[-0.0004],
                                [-0.0701]])
    problem.R = spa.eye(1)
    problem.Q = spa.diags([100, 10])
    problem.QN = spa.diags([100, 10])
    problem.umin = np.array([-0.0524])
    problem.umax = np.array([0.0524])
    problem.xmin = np.array([-0.2, -0.1])
    problem.xmax = np.array([0.01, 0.1])
    problem.T = []
    problem.tmin = []
    problem.tmax = []
    problem.x0 = np.array([-0.05, 0])
    problem.name = 'ball'

    return problem
