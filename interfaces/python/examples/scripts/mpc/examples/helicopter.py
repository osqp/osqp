import scipy.sparse as spa
import scipy.linalg as spla
import numpy as np
from collections import namedtuple
from .mpc_problem import MPCProblem


def load_helicopter_data():
    '''
    Load MPC helicopter example
    https://github.com/ferreau/mpcBenchmarking/blob/master/benchmarks/Benchmark_helicopter.m
    '''

    # Problem setup
    A = spa.csc_matrix([[0.99, 0., 0.01, 0., 0., 0.],
                        [0., 0.99, 0., 0.01, 0., 0.],
                        [0., 0., 0.99, 0., 0., 0.],
                        [0., 0., 0., 0.99, 0., 0.],
                        [0.01, 0., 0., 0., 0.99, 0.],
                        [0., 0.01, 0., 0., 0., 0.99]])
    B = spa.csc_matrix([[0., 0.],
                        [0.0001, -0.0001],
                        [0.0019, -0.0019],
                        [0.0132, -0.0132],
                        [0., 0.],
                        [0., 0.]])
    R = 0.001*spa.eye(2)
    Q = spa.diags([100, 100, 10, 10, 400, 200])
    QN = spla.solve_discrete_are(A.todense(), B.todense(),
                                 Q.todense(), R.todense())
    QN = spa.csc_matrix((QN + QN.T) / 2)
    umin = -np.ones(2)
    umax = 3*np.ones(2)
    xmin = -np.array([np.inf, np.inf, 0.44, 0.6, np.inf, np.inf])
    xmax = np.array([np.inf, np.inf, 0.44, 0.6, np.inf, np.inf])
    T = spa.eye(6)
    tmin = -np.ones(6)
    tmax = np.ones(6)
    x0 = np.array([0.5, 0.5, 0., 0., 0., 0.])
    name = 'helicopter'

    return MPCProblem(name,
                      A=A, B=B,
                      R=R, Q=Q, QN=QN,
                      umin=umin, umax=umax,
                      xmin=xmin, xmax=xmax,
                      T=T, tmin=tmin, tmax=tmax,
                      x0=x0)
