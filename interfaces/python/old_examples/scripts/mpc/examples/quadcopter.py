import scipy.sparse as spa
import numpy as np
from .mpc_problem import MPCProblem


def load_quadcopter_data():
    '''
    Load MPC helicopter example
    https://github.com/ferreau/mpcBenchmarking/blob/master/benchmarks/Benchmark_quadcopter.m
    '''

    # Problem setup
    A = spa.csc_matrix([
                        [1.,      0.,     0., 0., 0., 0., 0.1,
                         0.,     0.,  0.,     0.,     0.],
                        [0.,      1.,     0., 0., 0., 0., 0.,
                         0.1,    0.,  0.,     0.,     0.],
                        [0.,      0.,     1., 0., 0., 0., 0.,
                         0.,     0.1, 0.,     0.,     0.],
                        [0.0488,  0.,     0., 1., 0., 0., 0.0016,
                         0.,     0.,  0.0992, 0.,     0.],
                        [0.,     -0.0488, 0., 0., 1., 0., 0., -0.0016, 0.,
                         0.,     0.0992, 0.],
                        [0.,      0.,     0., 0., 0., 1., 0.,
                         0.,     0.,  0.,     0.,     0.0992],
                        [0.,      0.,     0., 0., 0., 0., 1.,
                         0.,     0.,  0.,     0.,     0.],
                        [0.,      0.,     0., 0., 0., 0., 0.,
                         1.,     0.,  0.,     0.,     0.],
                        [0.,      0.,     0., 0., 0., 0., 0.,
                         0.,     1.,  0.,     0.,     0.],
                        [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,
                         0.,  0.9846, 0.,     0.],
                        [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488,
                         0.,  0.,     0.9846, 0.],
                        [0.,      0.,     0., 0., 0., 0., 0.,
                         0.,     0.,  0.,     0.,     0.9846]])

    B = spa.csc_matrix([
                        [0.,      -0.0726,  0.,     0.0726],
                        [-0.0726,  0.,      0.0726, 0.],
                        [-0.0152,  0.0152, -0.0152, 0.0152],
                        [-0.,     -0.0006, -0.,     0.0006],
                        [0.0006,   0.,     -0.0006, 0.0000],
                        [0.0106,   0.0106,  0.0106, 0.0106],
                        [0,       -1.4512,  0.,     1.4512],
                        [-1.4512,  0.,      1.4512, 0.],
                        [-0.3049,  0.3049, -0.3049, 0.3049],
                        [-0.,     -0.0236,  0.,     0.0236],
                        [0.0236,   0.,     -0.0236, 0.],
                        [0.2107,   0.2107,  0.2107, 0.2107]])

    [nx, nu] = B.shape

    # Constraints
    u0 = np.array([10.59, 0, 0, 0])
    umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
    umax = np.array([13., 13., 13., 13.]) - u0
    xmin = np.array([-np.pi / 6, -np.pi / 6, -np.inf, -np.inf, -np.inf, -1.,
                     -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    xmax = np.array([np.pi / 6, np.pi / 6, np.inf, np.inf, np.inf, np.inf,
                     np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

    # Objective function
    Q = spa.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
    QN = Q
    R = 0.1 * spa.eye(4)

    # Initial and reference states
    x0 = np.zeros(12)
    xr = np.array([0., 0., 1., 0., 0., 0., 0., 0.,  0., 0., 0., 0.])
    name = 'quadcopter'

    return MPCProblem(name,
                      A=A, B=B,
                      R=R, Q=Q, QN=QN,
                      umin=umin, umax=umax,
                      xmin=xmin, xmax=xmax,
                      x0=x0, xr=xr)
