import scipy.sparse as spa
import numpy as np
from .mpc_problem import MPCProblem


def load_pendulum_data():
    '''
    Load MPC pendulum example
    https://github.com/ferreau/mpcBenchmarking/blob/master/benchmarks/Benchmark_pendulum.m
    '''

    # Problem setup (sampling interval Ts = 0.05s)
    A = spa.csc_matrix([[1.001, -0.05, -0.001],
                        [-0.05, 1.003, 0.05],
                        [-0.001, 0.05, 1.001]])
    B = spa.csc_matrix([[0.], [0.001], [0.05]])
    R = spa.diags([0.1])
    Q = spa.eye(3)
    QN = spa.csc_matrix([[113.4637, -95.4971, -21.0556],
                         [-95.4971, 99.5146, 23.3756],
                         [-21.0556, 23.3756, 12.1868]])
    umin = np.array([-1.25])
    umax = np.array([1.25])
    x0 = 0.6*np.ones(3)
    name = 'pendulum'

    return MPCProblem(name,
                      A=A, B=B,
                      R=R, Q=Q, QN=QN,
                      umin=umin, umax=umax,
                      x0=x0)
