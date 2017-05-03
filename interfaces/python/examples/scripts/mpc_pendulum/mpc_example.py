"""
Code for Portfolio example

This script compares:
    - OSQP
    - qpOASES
    - GUROBI
"""

from __future__ import print_function
from __future__ import division

import osqp  # Import osqp
#import qpoases as qpoases  # Import qpoases
import mathprogbasepy as mpbpy  # Mathprogbasepy to benchmark gurobi

# Numerics
import numpy as np
import scipy.sparse as spa

# Pandas
from collections import OrderedDict, namedtuple

# Import examples utilities
from .. import utils


def gen_qp_matrices(k, n, gammas, dens_lvl=0.5):
    """
    Generate QP matrices for portfolio optimization problem
    """

    # Generate data
    F = spa.random(n, k, density=dens_lvl, format='csc')
    D = spa.diags(np.random.rand(n) * np.sqrt(k), format='csc')
    mu = np.random.randn(n)

    # Construct the problem
    #       minimize	x' D x + y' I y - (1/gamma) * mu' x
    #       subject to  1' x = 1
    #                   F' x = y
    #                   0 <= x <= 1
    P = spa.block_diag((2*D, 2*spa.eye(k)), format='csc')
    A = spa.vstack([
            spa.hstack([spa.csc_matrix(np.ones((1, n))),
                       spa.csc_matrix((1, k))]),
            spa.hstack([F.T, -spa.eye(k)])
        ]).tocsc()
    l = np.hstack([1., np.zeros(k)])   # Linear constraints
    u = np.hstack([1., np.zeros(k)])
    lx = np.zeros(n)   # Bounds
    ux = np.ones(n)

    # Create linear cost vectors
    q = np.empty((k + n, 0))
    for gamma in gammas:
        q = np.column_stack((q, np.append(-mu / gamma, np.zeros(k))))

    qp_matrices = utils.QPmatrices(P, q, A, l, u, lx, ux)

    # Return QP matrices
    return qp_matrices


def solve_loop(qp_matrices, solver='osqp'):
    """
    Solve MPC loop
    """
    # Shorter name for qp_matrices
    qp = qp_matrices

    # Get dimensions
    n = len(qp.lx)
    k = len(qp.l) - 1

    print('\nSolving MPC pendulum problem for solver %s' % solver)

    # Get number of problems to solve
    n_prob = qp.q.shape[1]

    # Initialize time vector
    time = np.zeros(n_prob)

    # Initialize number of iterations vector
    niter = np.zeros(n_prob)

    if solver == 'osqp':
        # Construct qp matrices
        Aosqp = spa.vstack((qp.A,
                            spa.hstack((spa.eye(n), spa.csc_matrix((n, k)))
                                       ))).tocsc()
        losqp = np.append(qp.l, qp.lx)
        uosqp = np.append(qp.u, qp.ux)

        # Setup OSQP
        m = osqp.OSQP()
        m.setup(qp.P, qp.q[:, 0], Aosqp, losqp, uosqp,
                auto_rho=False,
                rho=0.1,
                polish=False,
                verbose=False)

        for i in range(n_prob):
            q = qp.q[:, i]

            # Update linear cost
            m.update(q=q)

            # Solve
            results = m.solve()
            x = results.x
            y = results.y
            status = results.info.status_val
            niter[i] = results.info.iter
            time[i] = results.info.run_time

            # Check if status correct
            if status != m.constant('OSQP_SOLVED'):
                import ipdb; ipdb.set_trace()
                raise ValueError('OSQP did not solve the problem!')

    elif solver == 'osqp_coldstart':
        # Construct qp matrices
        Aosqp = spa.vstack((qp.A,
                            spa.hstack((spa.eye(n), spa.csc_matrix((n, k)))
                                       ))).tocsc()
        losqp = np.append(qp.l, qp.lx)
        uosqp = np.append(qp.u, qp.ux)

        # Setup OSQP
        m = osqp.OSQP()
        m.setup(qp.P, qp.q[:, 0], Aosqp, losqp, uosqp,
                warm_start=False,
                auto_rho=False,
                rho=0.1,
                polish=False,
                verbose=True)

        for i in range(n_prob):
            q = qp.q[:, i]

            # Update linear cost
            m.update(q=q)

            # Solve
            results = m.solve()
            x = results.x
            y = results.y
            status = results.info.status_val
            niter[i] = results.info.iter
            time[i] = results.info.run_time

            # Check if status correct
            if status != m.constant('OSQP_SOLVED'):
                import ipdb; ipdb.set_trace()
                raise ValueError('OSQP did not solve the problem!')

    elif solver == 'qpoases':

        n_dim = qp.P.shape[0]  # Number of variables
        m_dim = qp.A.shape[0]  # Number of constraints without bounds

        # Initialize qpoases and set options
        qpoases_m = qpoases.PyQProblem(n_dim, m_dim)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qpoases_m.setOptions(options)

        # Construct bounds for qpoases
        lx = np.append(qp.lx, -np.inf * np.ones(k))
        ux = np.append(qp.ux, np.inf * np.ones(k))

        # Setup matrix P and A
        P = np.ascontiguousarray(qp.P.todense())
        A = np.ascontiguousarray(qp.A.todense())

        for i in range(n_prob):

            # Get linera cost as contiguous array
            q = np.ascontiguousarray(qp.q[:, i])

            # Reset cpu time
            qpoases_cpu_time = np.array([20.])

            # Reset number of of working set recalculations
            nWSR = np.array([1000])

            if i == 0:
                # First iteration
                res_qpoases = qpoases_m.init(P, q, A,
                                             np.ascontiguousarray(lx),
                                             np.ascontiguousarray(ux),
                                             np.ascontiguousarray(qp.l),
                                             np.ascontiguousarray(qp.u),
                                             nWSR, qpoases_cpu_time)
            else:
                # Solve new hot started problem
                res_qpoases = qpoases_m.hotstart(q,
                                                 np.ascontiguousarray(lx),
                                                 np.ascontiguousarray(ux),
                                                 np.ascontiguousarray(qp.l),
                                                 np.ascontiguousarray(qp.u),
                                                 nWSR,
                                                 qpoases_cpu_time)

            if res_qpoases != 0:
                raise ValueError('qpoases did not solve the problem!')

            # Save time
            time[i] = qpoases_cpu_time[0]

            # Save number of iterations
            niter[i] = nWSR[0]

    elif solver == 'gurobi':

        # Construct qp matrices
        Agurobi = spa.vstack((qp.A,
                              spa.hstack((spa.eye(n), spa.csc_matrix((n, k)))
                                         ))).tocsc()
        lgurobi = np.append(qp.l, qp.lx)
        ugurobi = np.append(qp.u, qp.ux)

        for i in range(n_prob):

            # Get linera cost as contiguous array
            q = qp.q[:, i]

            # Solve with gurobi
            prob = mpbpy.QuadprogProblem(qp.P, q, Agurobi, lgurobi, ugurobi)
            res = prob.solve(solver=mpbpy.GUROBI, verbose=False)

            # Save time
            time[i] = res.cputime

            # Save number of iterations
            niter[i] = res.total_iter

    else:
        raise ValueError('Solver not understood')

    # Return statistics
    return utils.Statistics(time), utils.Statistics(niter)


def run_mpc_pendulum_example():
    '''
    Solve problems
    '''

    # Problem setup
    problem = namedtuple("problem", "A B R Q P umin umax N x0")
    problem.A = spa.csc_matrix([[1.001, -0.05, -0.001],
                                [-0.05, 1.003, 0.05],
                                [-0.001, 0.05, 1.001]])
    problem.B = spa.csc_matrix([[0.], [0.001], [0.05]])

    # Simulation steps
    nsim = 10

    # Define statistics for osqp, qpoases and gurobi
    osqp_timing = []
    osqp_iter = []

    osqp_coldstart_timing = []
    osqp_coldstart_iter = []
    # qpoases_timing = []
    # qpoases_iter = []
    gurobi_iter = []
    gurobi_timing = []

    for i in range(len(n_vec)):
        # Generate QP
        qp_matrices = gen_qp_matrices(k_vec[i], n_vec[i], gammas)

        # Solve loop with osqp
        timing, niter = solve_loop(qp_matrices, 'osqp')
        osqp_timing.append(timing)
        osqp_iter.append(niter)

        # Solve loop with osqp (coldstart)
        timing, niter = solve_loop(qp_matrices, 'osqp_coldstart')
        osqp_coldstart_timing.append(timing)
        osqp_coldstart_iter.append(niter)

        # # Solving loop with qpoases
        # timing, niter = solve_loop(qp_matrices, 'qpoases')
        # qpoases_timing.append(timing)
        # qpoases_iter.append(niter)

        # Solve loop with gurobi
        timing, niter = solve_loop(qp_matrices, 'gurobi')
        gurobi_timing.append(timing)
        gurobi_iter.append(niter)

    solver_timings = OrderedDict([('OSQP (warm start)', osqp_timing),
                                  ('OSQP (cold start)',
                                   osqp_coldstart_timing),
                                  ('qpOASES', qpoases_timing),
                                  ('GUROBI', gurobi_timing)])

    utils.generate_plot('portfolio', 'time', 'median', n_vec,
                        solver_timings,
                        fig_size=0.9)
    utils.generate_plot('portfolio', 'time', 'total', n_vec,
                        solver_timings,
                        fig_size=0.9)
