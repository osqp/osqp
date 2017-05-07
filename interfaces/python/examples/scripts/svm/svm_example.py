"""
Code for SVM example
"""

from __future__ import print_function
from __future__ import division

import osqp  # Import osqp
# import qpoases as qpoases  # Import qpoases
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
    Generate QP matrices for SVM problem
    """

    # Generate data
    N = int(m / 2)
    gamma = 1.0
    b_svm = np.append(np.ones(N), -np.ones(N))
    A_upp = spa.random(N, n, density=dens_lvl)
    A_low = spa.random(N, n, density=dens_lvl)
    A_svm = spa.vstack([
            A_upp / np.sqrt(n) + (A_upp != 0.).astype(float) / n,
            A_low / np.sqrt(n) - (A_low != 0.).astype(float) / n
        ]).tocsc()

    # Construct the problem
    #       minimize	 x.T * x + gamma 1.T * t
    #       subject to  t >= diag(b) A x + 1
    #                   t >= 0
    P = spa.block_diag((spa.eye(n), spa.csc_matrix((m, m))), format='csc')
    q = np.append(np.zeros(n), (gamma/2)*np.ones(m))
    A = spa.hstack([spa.diags(b_svm).dot(A_svm), -spa.eye(m)]).tocsc()
    l = -np.inf*np.ones(m)  # Linear constraints
    u = -np.ones(m)
    lx = np.zeros(m)        # Bounds on t
    ux = np.inf*np.ones(m)

    qp_matrices = utils.QPmatrices(P, q, A, l, u, lx, ux)

    # Add further matrices for CVXPY modeling
    qp_matrices.A_svm = A_svm
    qp_matrices.b_svm = b_svm
    qp_matrices.gamma = gamma


    # Return QP matrices
    return qp_matrices


def solve_problem(qp_matrices, n_prob, solver='osqp'):
    """
    Solve SVM problem
    """
    # Shorter name for qp_matrices
    qp = qp_matrices

    # Initialize time vector
    time = np.zeros(n_prob)

    # Initialize number of iterations vector
    niter = np.zeros(n_prob)

    # Get dimensions
    m = len(qp.lx)
    n = len(qp.q) - m

    print('\nSolving SVM problem ' +
          'for n = %d (parameters) and solver %s' % (n, solver))

    if solver == 'osqp':
        # Construct qp matrices
        Aosqp = spa.vstack([
                    qp.A,
                    spa.hstack([spa.csc_matrix((m, n)), spa.eye(m)]),
                ]).tocsc()
        losqp = np.hstack([qp.l, qp.lx])
        uosqp = np.hstack([qp.u, qp.ux])

        for i in range(n_prob):
            # Setup OSQP
            m = osqp.OSQP()
            m.setup(qp.P, qp.q, Aosqp, losqp, uosqp,
                    rho=0.01,
                    auto_rho=False,
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
            time[i] = qpoases_cpu_time[0]
            niter[i] = nWSR[0]

    elif solver == 'gurobi':

        for i in range(n_prob):
            # Construct qp matrices
            Agurobi = spa.vstack([
                        qp.A,
                        spa.hstack([spa.csc_matrix((m, n)), spa.eye(m)]),
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
                        spa.hstack([spa.csc_matrix((m, n)), spa.eye(m)]),
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
            # Construct the problem
            #       minimize	 x.T * x + gamma 1.T * t
            #       subject to  t >= diag(b) A x + 1
            #                   t >= 0
            n_var = qp.A_svm.shape[1]
            m_var = qp.A_svm.shape[0]
            x = cvxpy.Variable(n_var)
            t = cvxpy.Variable(m_var)

            objective = cvxpy.Minimize(cvxpy.quad_form(x, spa.eye(n_var))
                                       + qp.gamma * np.ones(m_var) * t)
            constraints = [t >= spa.diags(qp.b_svm).dot(qp.A_svm) * x + 1,
                           t >= 0]

            problem = cvxpy.Problem(objective, constraints)
            problem.solve(solver=cvxpy.ECOS, verbose=False)


            # Obtain time and number of iterations
            time[i] = problem.solver_stats.setup_time + \
                problem.solver_stats.solve_time

            niter[i] = problem.solver_stats.num_iters

            # # DEBUG: solve with mosek
            # Amosek = spa.vstack([
            #             qp.A,
            #             spa.hstack([spa.csc_matrix((m, n)), spa.eye(m)]),
            #           ]).tocsc()
            # lmosek = np.hstack([qp.l, qp.lx])
            # umosek = np.hstack([qp.u, qp.ux])
            #
            # # Solve with mosek
            # prob = mpbpy.QuadprogProblem(qp.P, qp.q, Amosek, lmosek, umosek)
            # res = prob.solve(solver=mpbpy.MOSEK, verbose=False)
            # x_mosek = res.x[:n_var]
            #
            # import ipdb; ipdb.set_trace()


    else:
        raise ValueError('Solver not understood')

    # Return statistics
    return utils.Statistics(time), utils.Statistics(niter)


def run_svm_example():
    '''
    Solve problems
    '''

    # Reset random seed for repeatibility
    np.random.seed(1)

    # Parameter dimension
    n_vec = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    # n_vec = np.array([100, 200, 1000])

    # Factors
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

        # # Solving loop with qpoases
        # timing, niter = solve_problem(qp_matrices, 'qpoases')
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
                                  #('qpOASES', qpoases_timing),
                                  ('GUROBI', gurobi_timing),
                                  ('MOSEK', mosek_timing),
                                  ('ECOS', ecos_timing)])

    utils.generate_plot('svm', 'time', 'mean', n_vec, solver_timings,
                        fig_size=0.9)
