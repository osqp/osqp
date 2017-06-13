"""
Code for Huber example
"""

from __future__ import print_function
from __future__ import division

import osqp  # Import osqp
import qpoases as qpoases  # Import qpoases
import mathprogbasepy as mpbpy  # Mathprogbasepy to benchmark gurobi
import cvxpy

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
    A_huber = spa.random(m, n, density=dens_lvl, format='csc')
    x_true = np.random.randn(n) / np.sqrt(n)
    ind95 = (np.random.rand(m) < 0.95).astype(float)
    b_huber = A_huber.dot(x_true) + np.multiply(0.5*np.random.randn(m), ind95) \
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
            spa.hstack([A_huber, Im, Im]),
            spa.hstack([A_huber, -Im, -Im])]).tocsc()
    l = np.hstack([b_huber, -np.inf*np.ones(m)])  # Linear constraints
    u = np.hstack([np.inf*np.ones(m), b_huber])
    lx = np.zeros(2*m)                      # Bounds on (u,v)
    ux = np.hstack([np.ones(m), np.inf*np.ones(m)])

    qp_matrices = utils.QPmatrices(P, q, A, l, u, lx, ux)

    # Add further matrices for CVXPY modeling
    qp_matrices.A_huber = A_huber
    qp_matrices.b_huber = b_huber

    # Return QP matrices
    return qp_matrices


def solve_problem(qp_matrices, n_prob, solver='osqp'):
    """
    Solve Huber fitting problem
    """
    # Shorter name for qp_matrices
    qp = qp_matrices

    # Get dimensions
    m = int(len(qp.lx) / 2)
    n = len(qp.q) - 2*m

    print('n = %d and solver %s' % (n, solver))

    # Initialize time vector
    time = np.zeros(n_prob)

    # Initialize number of iterations vector
    niter = np.zeros(n_prob)

    if solver == 'osqp':
        # Construct qp matrices
        Aosqp = spa.vstack([
                    qp.A,
                    spa.hstack([spa.csc_matrix((2*m, n)), spa.eye(2*m)])
                ]).tocsc()
        losqp = np.hstack([qp.l, qp.lx])
        uosqp = np.hstack([qp.u, qp.ux])


        for i in range(n_prob):
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
            niter[i] = results.info.iter
            time[i] = results.info.run_time

            # Check if status correct
            if status != m.constant('OSQP_SOLVED'):
                import ipdb; ipdb.set_trace()
                raise ValueError('OSQP did not solve the problem!')

            # DEBUG
            # solve with gurobi
            # import mathprogbasepy as mpbpy
            # prob = mpbpy.QuadprogProblem(qp.P, qp.q, Aosqp, losqp, uosqp)
            # res = prob.solve(solver=mpbpy.GUROBI, verbose=False)
            # print('Norm difference OSQP-GUROBI %.3e' %
            #       np.linalg.norm(x - res.x))
            # import ipdb; ipdb.set_trace()

    elif solver == 'qpoases':


        for i in range(n_prob):
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
            qpoases_cpu_time = np.array([10.])

            # Reset number of working set recalculations
            nWSR = np.array([1000])

            # Solve
            res_qpoases = qpoases_m.init(P, np.ascontiguousarray(qp.q), A,
                                         np.ascontiguousarray(lx),
                                         np.ascontiguousarray(ux),
                                         np.ascontiguousarray(qp.l),
                                         np.ascontiguousarray(qp.u),
                                         nWSR, qpoases_cpu_time)

            # if res_qpoases != 0:
            #     raise ValueError('qpoases did not solve the problem!')

            # Save time and number of iterations
            time[i] = qpoases_cpu_time[0]
            niter[i] = nWSR[0]

    elif solver == 'gurobi':

        for i in range(n_prob):

            # Construct qp matrices
            Agurobi = spa.vstack([
                        qp.A,
                        spa.hstack([spa.csc_matrix((2*m, n)), spa.eye(2*m)])
                    ]).tocsc()
            lgurobi = np.hstack([qp.l, qp.lx])
            ugurobi = np.hstack([qp.u, qp.ux])

            # Solve with gurobi
            prob = mpbpy.QuadprogProblem(qp.P, qp.q, Agurobi, lgurobi, ugurobi)
            res = prob.solve(solver=mpbpy.GUROBI, verbose=False)
            niter[i] = res.total_iter
            time[i] = res.cputime

    elif solver == 'mosek':

        for i in range(n_prob):

            # Construct qp matrices
            Amosek = spa.vstack([
                        qp.A,
                        spa.hstack([spa.csc_matrix((2*m, n)), spa.eye(2*m)])
                    ]).tocsc()
            lmosek = np.hstack([qp.l, qp.lx])
            umosek = np.hstack([qp.u, qp.ux])

            # Solve with mosek
            prob = mpbpy.QuadprogProblem(qp.P, qp.q, Amosek, lmosek, umosek)
            res = prob.solve(solver=mpbpy.MOSEK, verbose=False)
            niter[i] = res.total_iter
            time[i] = res.cputime

    elif solver == 'ecos':
        for i in range(n_prob):

            # Model with CVXPY
            #       minimize	1/2 u.T * u + np.ones(m).T * v
            #       subject to  -u - v <= Ax - b <= u + v
            #                   0 <= u <= 1
            #                   v >= 0
            n_var = qp.A_huber.shape[1]
            m_var = qp.b_huber.shape[0]
            x = cvxpy.Variable(n_var)
            u = cvxpy.Variable(m_var)
            v = cvxpy.Variable(m_var)

            objective = cvxpy.Minimize(.5 * cvxpy.quad_form(u, spa.eye(m_var))
                                       + np.ones(m_var) * v)
            constraints = [-u - v <= qp.A_huber * x - qp.b_huber,
                           qp.A_huber * x - qp.b_huber <= u + v,
                           0 <= u, u <= 1,
                           v >= 0]
            problem = cvxpy.Problem(objective, constraints)
            problem.solve(solver=cvxpy.ECOS, verbose=False)


            # DEBUG: solve with MOSEK
            # Amosek = spa.vstack([
            #             qp.A,
            #             spa.hstack([spa.csc_matrix((2*m, n)), spa.eye(2*m)])
            #         ]).tocsc()
            # lmosek = np.hstack([qp.l, qp.lx])
            # umosek = np.hstack([qp.u, qp.ux])
            #
            # # Solve with mosek
            # prob = mpbpy.QuadprogProblem(qp.P, qp.q, Amosek, lmosek, umosek)
            # res = prob.solve(solver=mpbpy.MOSEK, verbose=False)
            # x_mosek = res.x[:n_var]

            # Obtain time and number of iterations
            time[i] = problem.solver_stats.setup_time + \
                problem.solver_stats.solve_time

            niter[i] = problem.solver_stats.num_iters

    else:
        raise ValueError('Solver not understood')

    # Return statistics
    return utils.Statistics(time), utils.Statistics(niter)


def run_huber_example():
    '''
    Solve problems
    '''

    print("Huber example")
    print("--------------------")
    
    # Reset random seed for repeatibility
    np.random.seed(1)

    # Parameter dimension
    n_vec = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    # n_vec = np.array([100, 200, 300])
    # n_vec = np.array([10])

    # Data dimension
    m_vec = n_vec * 10

    # Number of problems for each dimension
    n_prob = 10

    # Define statistics for osqp, qpoases and gurobi
    osqp_timing = []
    osqp_iter = []
    qpoases_timing = []
    qpoases_iter = []
    gurobi_iter = []
    gurobi_timing = []
    mosek_iter = []
    mosek_timing = []
    ecos_iter = []
    ecos_timing = []

    for i in range(len(n_vec)):
        # Generate QP
        qp_matrices = gen_qp_matrices(m_vec[i], n_vec[i])

        # Solve loop with osqp
        timing, niter = solve_problem(qp_matrices, n_prob, 'osqp')
        osqp_timing.append(timing)
        osqp_iter.append(niter)

        # Solving loop with qpoases (It cannot solve the problems!)
        # timing, niter = solve_problem(qp_matrices, n_prob, 'qpoases')
        # qpoases_timing.append(timing)
        # qpoases_iter.append(niter)

        # Solve loop with gurobi
        timing, niter = solve_problem(qp_matrices, n_prob, 'gurobi')
        gurobi_timing.append(timing)
        gurobi_iter.append(niter)

        # Solve loop with mosek
        timing, niter = solve_problem(qp_matrices, n_prob, 'mosek')
        mosek_timing.append(timing)
        mosek_iter.append(niter)

        # Solve loop with ecos
        timing, niter = solve_problem(qp_matrices, n_prob, 'ecos')
        ecos_timing.append(timing)
        ecos_iter.append(niter)

    solver_timings = OrderedDict([('OSQP', osqp_timing),
                                #   ('qpOASES', qpoases_timing),
                                  ('GUROBI', gurobi_timing),
                                  ('MOSEK', mosek_timing),
                                  ('ECOS', ecos_timing)])

    utils.generate_plot('huber', 'time', 'mean', n_vec, solver_timings,
                        fig_size=0.9)
