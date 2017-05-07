import scipy.sparse as spa
import numpy as np


def load_helicopter_data():
    '''
    Load MPC helicopter example
    https://github.com/ferreau/mpcBenchmarking/blob/master/benchmarks/Benchmark_helicopter.m
    '''

    # Problem setup
    problem = namedtuple("problem", "A B R Q QN umin umax xmin xmax " +
                                    "T tmin tmax N x0 name")
    problem.A = spa.csc_matrix([[0.99, 0., 0.01, 0., 0., 0.],
                                [0., 0.99, 0., 0.01, 0., 0.],
                                [0., 0., 0.99, 0., 0., 0.],
                                [0., 0., 0., 0.99, 0., 0.],
                                [0.01, 0., 0., 0., 0.99, 0.],
                                [0., 0.01, 0., 0., 0., 0.99]])
    problem.B = spa.csc_matrix([[0., 0.],
                                [0.0001, -0.0001],
                                [0.0019, -0.0019],
                                [0.0132, -0.0132],
                                [0., 0.],
                                [0., 0.]])
    problem.R = 0.001*spa.eye(2)
    problem.Q = spa.diags([100, 100, 10, 10, 400, 200])
    problem.QN = spla.solve_discrete_are(
                                problem.A.todense(), problem.B.todense(),
                                problem.Q.todense(), problem.R.todense())
    problem.umin = -np.ones(2)
    problem.umax = 3*np.ones(2)
    problem.xmin = -np.array([np.inf, np.inf, 0.44, 0.6, np.inf, np.inf])
    problem.xmax = np.array([np.inf, np.inf, 0.44, 0.6, np.inf, np.inf])
    problem.T = spa.eye(6)
    problem.tmin = -np.ones(6)
    problem.tmax = np.ones(6)
    problem.x0 = np.array([0.5, 0.5, 0., 0., 0., 0.])
    problem.name = 'helicopter'


    return problem
