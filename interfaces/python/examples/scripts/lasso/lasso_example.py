"""
Code for lasso example
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

# Pandas
import pandas as pd
from collections import OrderedDict

# Import examples utilities
from .. import utils


def gen_qp_matrices(n, m, lambdas, dens_lvl=0.5):
    """
    Generate QP matrices for lasso optimization problem
    """

    # Generate data
    Ad = spa.random(m, n, density=dens_lvl)
    x_true = np.multiply((np.random.rand(n) > 0.5).astype(float),
                         np.random.randn(n)) / np.sqrt(n)
    bd = Ad.dot(x_true) + np.random.randn(m)

    #       minimize	y' * y + lambda * 1' * t
    #       subject to  y = Ax - b
    #                   -t <= x <= t
    P = spa.block_diag((spa.csc_matrix((n, n)), 2*spa.eye(m),
                          spa.csc_matrix((n, n))), format='csc')
    In = spa.eye(n)
    Onm = spa.csc_matrix((n, m))
    A = spa.vstack([spa.hstack([Ad, -spa.eye(m), spa.csc_matrix((m, n))]),
                     spa.hstack([In, Onm, -In]),
                     spa.hstack([In, Onm, In])]).tocsc()
    l = np.hstack([bd, -np.inf * np.ones(n), np.zeros(n)])
    u = np.hstack([bd, np.zeros(n), np.inf * np.ones(n)])

    lx = -np.inf * np.ones(2 * n + m)
    ux = np.inf * np.ones(2 * n + m)

    # Create linear cost vectors
    q = np.empty((m + 2 * n, 0))
    for lambda_i in lambdas:
        q = np.column_stack((q, np.append(np.zeros(m + n), lambda_i*np.ones(n))
                             ))

    qp_matrices = utils.QPmatrices(P, q, A, l, u, lx, ux)

    # Add further matrices for CVXPY modeling
    qp_matrices.lambdas = lambdas
    qp_matrices.A_lasso = Ad
    qp_matrices.b_lasso = bd
    qp_matrices.n = n
    qp_matrices.m = m
    # qp_matrices.nnzA = A.nnz
    # qp_matrices.nnzP = P.nnz


    # Return QP matrices
    return qp_matrices


def solve_loop(qp_problem, solver='osqp', osqp_settings=None):
    """
    Solve lasso optimization loop for all lambdas
    """
    # Shorter name for qp_problem
    qp = qp_problem

    # Extract features
    n = qp.n

    print('n = %d and solver %s' %
          (n, solver))

    # Get number of problems to solve
    n_prob = qp.q_vec.shape[1]

    # Initialize time vector
    time = np.zeros(n_prob)

    # Initialize number of iterations vector
    niter = np.zeros(n_prob)

    if solver == 'osqp':

        # Setup OSQP
        m = osqp.OSQP()
        m.setup(qp.P, qp.q_vec[:, 0], qp.A, qp.l, qp.u, **osqp_settings)

        for i in range(n_prob):
            qp.q = qp.q_vec[:, i]

            # Update linear cost
            m.update(q=qp.q)

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

            if not qp.is_optimal(x, y):
                raise ValueError('Returned solution not optimal!')
            # DEBUG
            # solve with gurobi
            # prob = mpbpy.QuadprogProblem(qp.P, q, Aosqp, losqp, uosqp)
            # res = prob.solve(solver=mpbpy.GUROBI, verbose=False)
            # print('Norm difference OSQP-GUROBI %.3e' %
            #       np.linalg.norm(x - res.x))
            # import ipdb; ipdb.set_trace()

    elif solver == 'osqp_coldstart':

        # Setup OSQP
        m = osqp.OSQP()
        m.setup(qp.P, qp.q_vec[:, 0], qp.A, qp.l, qp.u, warm_start=False, **osqp_settings)

        for i in range(n_prob):
            qp.q = qp.q_vec[:, i]

            # Update linear cost
            m.update(q=qp.q)

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

            if not qp.is_optimal(x, y):
                raise ValueError('Returned solution not optimal!')

            # DEBUG
            # solve with gurobi
            # prob = mpbpy.QuadprogProblem(qp.P, q, qp.A, qp.l, qp.u)
            # res = prob.solve(solver=mpbpy.GUROBI, verbose=False)
            # print('Norm difference OSQP-GUROBI %.3e' %
            #       np.linalg.norm(x - res.x))
            # import ipdb; ipdb.set_trace()

        # DEBUG print iterations per value of lambda
        # lambda_vals = np.logspace(-2, 2, 101)[::-1]
        #
        # import matplotlib.pylab as plt
        # plt.figure()
        # ax = plt.gca()
        # plt.plot(lambda_vals, niter)
        # ax.set_xlabel(r'$\lambda$')
        # ax.set_ylabel(r'iter')
        # plt.show(block=False)

        # import ipdb; ipdb.set_trace()

    elif solver == 'osqp_no_caching':

        for i in range(n_prob):

            # Setup OSQP
            m = osqp.OSQP()
            qp.q = qp.q_vec[:, i]
            m.setup(qp.P, qp.q, qp.A, qp.l, qp.u, **osqp_settings)

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

            if not qp.is_optimal(x, y):
                raise ValueError('Returned solution not optimal!')

            # DEBUG
            # solve with gurobi
            # prob = mpbpy.QuadprogProblem(qp.P, q, Aosqp, losqp, uosqp)
            # res = prob.solve(solver=mpbpy.GUROBI, verbose=False)
            # print('Norm difference OSQP-GUROBI %.3e' %
            #       np.linalg.norm(x - res.x))
            # import ipdb; ipdb.set_trace()

    elif solver == 'qpoases':

        n_dim = qp.P.shape[0]  # Number of variables
        m_dim = qp.A.shape[0]  # Number of constraints without bounds

        # Initialize qpoases and set options
        qpoases_m = qpoases.PyQProblem(n_dim, m_dim)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qpoases_m.setOptions(options)

        # Setup matrix P and A
        P = np.ascontiguousarray(qp.P.todense())
        A = np.ascontiguousarray(qp.A.todense())

        for i in range(n_prob):

            # Get linera cost as contiguous array
            qp.q = np.ascontiguousarray(qp.q_vec[:, i])

            # Reset cpu time
            qpoases_cpu_time = np.array([10.])

            # Reset number of of working set recalculations
            nWSR = np.array([1000])

            if i == 0:
                # First iteration
                res_qpoases = qpoases_m.init(P, qp.q, A,
                                             np.ascontiguousarray(qp.lx),
                                             np.ascontiguousarray(qp.ux),
                                             np.ascontiguousarray(qp.l),
                                             np.ascontiguousarray(qp.u),
                                             nWSR, qpoases_cpu_time)
            else:
                # Solve new hot started problem
                res_qpoases = qpoases_m.hotstart(qp.q,
                                                 np.ascontiguousarray(qp.lx),
                                                 np.ascontiguousarray(qp.ux),
                                                 np.ascontiguousarray(qp.l),
                                                 np.ascontiguousarray(qp.u),
                                                 nWSR,
                                                 qpoases_cpu_time)

            # qpoases solution
            x = np.zeros(n_dim)
            y = np.zeros(n_dim + m_dim)
            qpoases_m.getPrimalSolution(x)
            qpoases_m.getDualSolution(y)
            y = y[n_dim:]

            if not qp.is_optimal(x, y):
                raise ValueError('Returned solution not optimal!')


            # # DEBUG Solve with gurobi
            # qpoases solution
            # sol_qpoases = np.zeros(n + k)
            # qpoases_m.getPrimalSolution(sol_qpoases)
            # import mathprogbasepy as mpbpy
            # Agrb = spa.vstack((qp.A,
            #                     spa.hstack((spa.eye(n), spa.csc_matrix((n, k)))
            #                                ))).tocsc()
            # lgrb = np.append(qp.l, qp.lx)
            # ugrb = np.append(qp.u, qp.ux)
            # prob = mpbpy.QuadprogProblem(spa.csc_matrix(qp.P), q,
            #                              Agrb, lgrb, ugrb)
            # res = prob.solve(solver=mpbpy.GUROBI, verbose=True)
            # print("Norm difference x qpoases - GUROBI = %.4f" %
            #       np.linalg.norm(sol_qpoases - res.x))
            # print("Norm difference objval qpoases - GUROBI = %.4f" %
            #       abs(qpoases_m.getObjVal() - res.obj_val))
            # import ipdb; ipdb.set_trace()

            # if res_qpoases != 0:
            #     raise ValueError('qpoases did not solve the problem!')

            # Save time
            time[i] = qpoases_cpu_time[0]

            # Save number of iterations
            niter[i] = nWSR[0]

    elif solver == 'gurobi':

        for i in range(n_prob):

            # Get linera cost as contiguous array
            qp.q = qp.q_vec[:, i]

            # Solve with gurobi
            prob = mpbpy.QuadprogProblem(qp.P, qp.q, qp.A, qp.l, qp.u)
            res = prob.solve(solver=mpbpy.GUROBI, verbose=False)

            # Save time
            time[i] = res.cputime

            # Save number of iterations
            niter[i] = res.total_iter

    elif solver == 'mosek':

        for i in range(n_prob):

            # Get linera cost as contiguous array
            qp.q = qp.q_vec[:, i]

            # Solve with mosek
            prob = mpbpy.QuadprogProblem(qp.P, qp.q, qp.A, qp.l, qp.u)
            res = prob.solve(solver=mpbpy.MOSEK, verbose=False)

            # Save time
            time[i] = res.cputime

            # Save number of iterations
            niter[i] = res.total_iter

    elif solver == 'ecos':

        n_var = qp_problem.A_lasso.shape[1]
        m_var = qp_problem.A_lasso.shape[0]

        lambda_i = cvxpy.Parameter(sign="positive")
        x = cvxpy.Variable(n_var)
        y = cvxpy.Variable(m_var)
        t = cvxpy.Variable(n_var)

        objective = cvxpy.Minimize(cvxpy.quad_form(y, spa.eye(m_var))
                                   + lambda_i * (np.ones(n_var) * t))
        constraints = [y == qp_problem.A_lasso * x - qp_problem.b_lasso,
                       -t <= x, x <= t]
        problem = cvxpy.Problem(objective, constraints)

        for i in range(n_prob):
            if n_var <= 60:  # (problem becomes too big otherwise):

                # Model with CVXPY
                #       minimize	y' * y + lambda * 1' * t
                #       subject to  y = Ax - b
                #                   -t <= x <= t
                lambda_i.value = qp_problem.lambdas[i]
                problem.solve(solver=cvxpy.ECOS, verbose=False)

                # DEBUG: Solve with MOSEK
                # q = qp.q[:, i]

                # Solve with mosek
                # prob = mpbpy.QuadprogProblem(qp.P, qp.q_vec[:, i],
                #                              qp.A, qp.l, qp.u)
                # res = prob.solve(solver=mpbpy.MOSEK, verbose=False)
                # x_mosek = res.x[:n_var]
                # import ipdb; ipdb.set_trace()

                # Check if solved
                x_ecos = np.concatenate((x.value.A1, y.value.A1, t.value.A1))
                y_ecos = np.concatenate((-constraints[0].dual_value.A1,
                                         constraints[2].dual_value.A1,
                                         -constraints[1].dual_value.A1))

                qp.q = qp.q_vec[:, i]
                if not qp.is_optimal(x_ecos, y_ecos):
                    raise ValueError('Returned solution not optimal')

                # Obtain time and number of iterations
                time[i] = problem.solver_stats.setup_time + \
                    problem.solver_stats.solve_time

                niter[i] = problem.solver_stats.num_iters
            else:
                time[i] = 0
                niter[i] = 0

    else:
        raise ValueError('Solver not understood')

    # Return statistics
    return utils.Statistics(time), utils.Statistics(niter)


def run_lasso_example(osqp_settings):
    '''
    Solve problems
    '''


    print("Lasso  example")
    print("--------------------")

    # Reset random seed for repetibility
    np.random.seed(1)

    # Generate lambda parameters and cost vectors
    n_lambda = 11
    lambdas = np.logspace(-1, 1, n_lambda)[::-1]
    # lambdas = np.logspace(-2, 2, n_lambda)

    # Parameters
    #  n_vec = np.array([10, 20, 40, 50, 60, 80, 100])
    # n_vec = np.array([10, 50, 100, 500, 1000])
    n_vec = np.array([10, 20, 30])

    # Points
    m_vec = (n_vec * 100).astype(int)

    # Matrix of dimensions
    dims_mat = np.zeros((4, len(n_vec)))

    # Define statistics
    osqp_timing = []
    osqp_iter = []
    osqp_no_caching_timing = []
    osqp_no_caching_iter = []
    osqp_coldstart_timing = []
    osqp_coldstart_iter = []
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
        qp_matrices = gen_qp_matrices(n_vec[i], m_vec[i], lambdas)

        # Get dimensions
        dims_mat[:, i] = np.array([qp_matrices.n,
                                   qp_matrices.m,
                                   qp_matrices.nnzA,
                                   qp_matrices.nnzP])

        # Solve loop with osqp
        timing, niter = solve_loop(qp_matrices, 'osqp', osqp_settings)
        osqp_timing.append(timing)
        osqp_iter.append(niter)

        # Solve loop with osqp (coldstart)
        timing, niter = solve_loop(qp_matrices, 'osqp_coldstart', osqp_settings)
        osqp_coldstart_timing.append(timing)
        osqp_coldstart_iter.append(niter)

        # Solve loop with osqp (no caching)
        timing, niter = solve_loop(qp_matrices, 'osqp_no_caching',
                osqp_settings)
        osqp_no_caching_timing.append(timing)
        osqp_no_caching_iter.append(niter)

        # # Solving loop with qpoases (qpOASES saturates after smallest problem)
        # timing, niter = solve_loop(qp_matrices, 'qpoases')
        # qpoases_timing.append(timing)
        # qpoases_iter.append(niter)

        # Solve loop with gurobi
        timing, niter = solve_loop(qp_matrices, 'gurobi')
        gurobi_timing.append(timing)
        gurobi_iter.append(niter)
#
        # Solve loop with mosek
        timing, niter = solve_loop(qp_matrices, 'mosek')
        mosek_timing.append(timing)
        mosek_iter.append(niter)

        #  Solve loop with ecos
        timing, niter = solve_loop(qp_matrices, 'ecos')
        ecos_timing.append(timing)
        ecos_iter.append(niter)

    '''
    Define timings and dimensions dictionaries
    '''
    solver_timings = OrderedDict([('OSQP (warm start)', osqp_timing),
                                  ('OSQP (cold start)',
                                   osqp_coldstart_timing),
                                  ('OSQP (no caching)',
                                   osqp_no_caching_timing),
                                  #   ('qpOASES', qpoases_timing),
                                  ('GUROBI', gurobi_timing),
                                  ('MOSEK', mosek_timing),
                                  ('ECOS', ecos_timing)
                                  ])
    #  cols_timings = ['OSQP (warm  start)', 'OSQP (cold start)', 'OSQP (no caching)',
    #          'GUROBI', 'MOSEK']

    dims_dict = {'n': dims_mat[0, :],
                 'm': dims_mat[1, :],
                 'nnzA': dims_mat[2, :],
                 'nnzP': dims_mat[3, :]}
    cols_dims = ['n', 'm', 'nnzA', 'nnzP']

    '''
    Store timings and dimensions
    '''
    utils.store_timings("lasso", solver_timings, n_vec, 'mean')
    utils.store_timings("lasso", solver_timings, n_vec, 'max')
    utils.store_dimensions("lasso", dims_dict, cols_dims)

    '''
    Store plots
    '''

    fig_size = None  # Adapt for talk plots

    utils.generate_plot('lasso', 'time', 'median', n_vec,
                        solver_timings,
                        fig_size=fig_size)
    utils.generate_plot('lasso', 'time', 'total', n_vec,
                        solver_timings,
                        fig_size=fig_size)
    utils.generate_plot('lasso', 'time', 'mean', n_vec,
                        solver_timings,
                        fig_size=fig_size)
