"""
Code for Huber example

This script compares:
    - OSQP
    - qpOASES
    - GUROBI
"""

from __future__ import print_function
from __future__ import division

import osqp  # Import osqp
# import qpoases as qpoases  # Import qpoases
import mathprogbasepy as mpbpy  # Mathprogbasepy to benchmark gurobi

# Numerics
import numpy as np
import scipy.sparse as spa

from collections import OrderedDict

# Import examples utilities
from .. import utils


def gen_qp_matrices(m, n, dens_lvl=0.5):
    """
    Generate QP matrices for Huber fitting problem
    """

    # Generate data
    A = spa.random(m, n, density=dens_lvl, format='csc')
    x_true = np.random.randn(n) / np.sqrt(n)
    ind95 = (np.random.rand(m) < 0.95).astype(float)
    b = A.dot(x_true) + np.multiply(0.5*np.random.randn(m), ind95) \
                      + np.multiply(10.*np.random.rand(m), 1. - ind95)

    # Construct the problem
    #       minimize	1/2 u.T * u + np.ones(m).T * v
    #       subject to  -u - v <= Ax - b <= u + v
    #                   0 <= u <= 1
    #                   v >= 0
    Im = spa.eye(m)
    P = spa.block_diag((spa.csc_matrix((n, n)), Im,
                        spa.csc_matrix((m, m))), format='csc')
    q = np.append(np.zeros(m + n), np.ones(m))
    A = spa.vstack([
            spa.hstack([A, Im, Im]),
            spa.hstack([A, -Im, -Im])]).tocsc()
    l = np.hstack([b, -np.inf*np.ones(m)])  # Linear constraints
    u = np.hstack([np.inf*np.ones(m), b])
    lx = np.zeros(2*m)                      # Bounds on (u,v)
    ux = np.hstack([np.ones(m), np.inf*np.ones(m)])

    qp_matrices = utils.QPmatrices(P, q, A, l, u, lx, ux)

    # Return QP matrices
    return qp_matrices


def solve_problem(qp_matrices, solver='osqp'):
    """
    Solve Huber fitting problem
    """
    # Shorter name for qp_matrices
    qp = qp_matrices

    # Get dimensions
    m = int(len(qp.lx) / 2)
    n = len(qp.q) - 2*m

    print('\nSolving Huber fitting problem' +
          'for n = %d (parameters) and solver %s' % (n, solver))

    if solver == 'osqp':
        # Construct qp matrices
        Im = spa.eye(m)
        Om = spa.csc_matrix((m, m))
        Omn = spa.csc_matrix((m, n))

        Aosqp = spa.vstack([
                    qp.A,
                    spa.hstack([Omn, Im, Om]),
                    spa.hstack([Omn, Om, Im]),
                ]).tocsc()
        losqp = np.hstack([qp.l, np.zeros(m), np.zeros(m)])
        uosqp = np.hstack([qp.u, np.ones(m), np.inf*np.ones(m)])

        # Setup OSQP
        m = osqp.OSQP()
        m.setup(qp.P, qp.q, Aosqp, losqp, uosqp,
                auto_rho=True,
                polish=False,
                verbose=False)

        # Solve
        results = m.solve()
        x = results.x
        status = results.info.status_val
        niter = results.info.iter
        time = results.info.run_time

        # Check if status correct
        if status != m.constant('OSQP_SOLVED'):
            import ipdb; ipdb.set_trace()
            raise ValueError('OSQP did not solve the problem!')

        # DEBUG
        # solve with gurobi
        # import mathprogbasepy as mpbpy
        prob = mpbpy.QuadprogProblem(qp.P, qp.q, Aosqp, losqp, uosqp)
        res = prob.solve(solver=mpbpy.GUROBI, verbose=False)
        print('Norm difference OSQP-GUROBI %.3e' %
              np.linalg.norm(x - res.x))
        # import ipdb; ipdb.set_trace()

    elif solver == 'qpoases':

        n_dim = qp.P.shape[0]  # Number of variables
        m_dim = qp.A.shape[0]  # Number of constraints without bounds

        # Initialize qpoases and set options
        qpoases_m = qpoases.PyQProblem(n_dim, m_dim)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qpoases_m.setOptions(options)

        # Construct bounds for qpoases
        lx = np.append(-np.inf*np.ones(n), qp.lx)
        ux = np.append(np.inf*np.ones(n), qp.ux)

        # Setup matrix P and A
        P = np.ascontiguousarray(qp.P.todense())
        A = np.ascontiguousarray(qp.A.todense())

        # Reset cpu time
        qpoases_cpu_time = np.array([20.])

        # Reset number of working set recalculations
        nWSR = np.array([1000])

        # Solve
        res_qpoases = qpoases_m.init(P, np.ascontiguousarray(qp.q), A,
                                     np.ascontiguousarray(lx),
                                     np.ascontiguousarray(ux),
                                     np.ascontiguousarray(qp.l),
                                     np.ascontiguousarray(qp.u),
                                     nWSR, qpoases_cpu_time)

        if res_qpoases != 0:
            raise ValueError('qpoases did not solve the problem!')

        # Save time and number of iterations
        time = qpoases_cpu_time[0]
        niter = nWSR[0]

    elif solver == 'gurobi':

        # Construct qp matrices
        Im = spa.eye(m)
        Om = spa.csc_matrix((m, m))
        Omn = spa.csc_matrix((m, n))

        Agurobi = spa.vstack([
                    qp.A,
                    spa.hstack([Omn, Im, Om]),
                    spa.hstack([Omn, Om, Im]),
                ]).tocsc()
        lgurobi = np.hstack([qp.l, np.zeros(m), np.zeros(m)])
        ugurobi = np.hstack([qp.u, np.ones(m), np.inf*np.ones(m)])

        # Solve with gurobi
        prob = mpbpy.QuadprogProblem(qp.P, qp.q, Agurobi, lgurobi, ugurobi)
        res = prob.solve(solver=mpbpy.GUROBI, verbose=False)
        niter = res.total_iter
        time = res.cputime

    else:
        raise ValueError('Solver not understood')

    # Return statistics
    return utils.Statistics(time), utils.Statistics(niter)


def run_huber_example():
    '''
    Solve problems
    '''

    # Reset random seed for repeatibility
    np.random.seed(1)

    # Parameter dimension
    n_vec = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

    # Factors
    m_vec = n_vec * 10

    # Define statistics for osqp, qpoases and gurobi
    osqp_timing = []
    osqp_iter = []
    # qpoases_timing = []
    # qpoases_iter = []
    gurobi_iter = []
    gurobi_timing = []

    for i in range(len(n_vec)):
        # Generate QP
        qp_matrices = gen_qp_matrices(m_vec[i], n_vec[i])

        # Solve loop with osqp
        timing, niter = solve_problem(qp_matrices, 'osqp')
        osqp_timing.append(timing)
        osqp_iter.append(niter)

        # # Solving loop with qpoases
        # timing, niter = solve_problem(qp_matrices, 'qpoases')
        # qpoases_timing.append(timing)
        # qpoases_iter.append(niter)

        # Solve loop with gurobi
        timing, niter = solve_problem(qp_matrices, 'gurobi')
        gurobi_timing.append(timing)
        gurobi_iter.append(niter)

    solvers = OrderedDict([('OSQP', osqp_timing),
                          #('qpOASES', qpoases_timing),
                          ('GUROBI', gurobi_timing)])

    utils.generate_plot('huber', 'median', n_vec, solvers,
                        fig_size=0.9)
    utils.generate_plot('huber', 'total', n_vec, solvers,
                        fig_size=0.9)
