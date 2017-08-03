import scipy.sparse as spa
import numpy as np
from collections import namedtuple
from .mpc_problem import MPCProblem


def load_ball_data():
    '''
    Load MPC ball on plate example
    https://github.com/ferreau/mpcBenchmarking/blob/master/benchmarks/Benchmark_ballOnPlate.m
    '''

    # Problem setup
    A = spa.csc_matrix([[1., 0.01],
                        [0., 1.]])
    B = spa.csc_matrix([[-0.0004],
                        [-0.0701]])
    R = spa.eye(1)
    Q = spa.diags([100, 10])
    QN = spa.diags([100, 10])
    umin = np.array([-0.0524])
    umax = np.array([0.0524])
    xmin = np.array([-0.2, -0.1])
    xmax = np.array([0.01, 0.1])
    # T = []
    # tmin = []
    # tmax = []
    x0 = np.array([-0.05, 0])
    name = 'ball'

    return MPCProblem(name,
                      A=A, B=B,
                      R=R, Q=Q, QN=QN,
                      umin=umin, umax=umax,
                      xmin=xmin, xmax=xmax,
                      x0=x0)
