"""
Run equality constrained QP example
"""


from __future__ import print_function
from __future__ import division

import osqp  # Import osqp
import qpoases as qpoases  # Import qpoases
import mathprogbasepy as mpbpy  # Mathprogbasepy to benchmark gurobi

# Numerics
import numpy as np
import scipy.sparse as spa

# Pandas
import pandas as pd
from collections import OrderedDict

# Import examples utilities
from .. import utils


def gen_qp_matrices(n, m, dens_lvl=0.5):
    """
    Generate QP matrices for the quality constrained QP problem
    """

    # Construct problem
    #       minimize	1/2 x' P x + q'*x
    #       subject to  A * x = b
    #
    P = spa.random(n, n, density=dens_lvl, format='csc')
    P = P.dot(P.T).tocsc()
    q = np.random.randn(n)

    # scal_cost = np.linalg.norm(q)
    # P /= scal_cost
    # q /= scal_cost

    A = spa.random(m, n, density=dens_lvl, format='csc')
    l = np.random.randn(m)
    u = np.copy(l)

    # scal_constraints = np.linalg.norm(l)
    # A /= scal_constraints
    # l /= scal_constraints
    # u /= scal_constraints

    lx = -np.inf * np.ones(n)
    ux = np.inf * np.ones(n)

    qp_matrices = utils.QPmatrices(P, q, A, l, u, lx, ux)

    # Return QP matrices
    return qp_matrices


def solve_problem(qp_matrices, n_prob, solver='osqp'):
    """
    Solve equality constrained optimization
    """
    # Shorter name for qp_matrices
    qp = qp_matrices

    # Get dimensions
    n = len(qp.q)
    m = len(qp.l)

    print('\nSolving equality constrained QP problem for n = %d and solver %s' %
          (n, solver))

    # Initialize time vector
    time = np.zeros(n_prob)

    # Initialize number of iterations vector
    niter = np.zeros(n_prob)

    if solver == 'osqp':

        for i in range(n_prob):
            # Setup OSQP
            m = osqp.OSQP()
            m.setup(qp.P, qp.q, qp.A, qp.l, qp.u,
                    rho=40,   # Set high rho to enforce feasibility
                    auto_rho=False,
                    polish=False,
                    verbose=False)

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
                # raise ValueError('OSQP did not solve the problem!')

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

        for i in range(n_prob):

            # Initialize qpoases and set options
            qpoases_m = qpoases.PyQProblem(n_dim, m_dim)
            options = qpoases.PyOptions()
            options.printLevel = qpoases.PyPrintLevel.NONE
            qpoases_m.setOptions(options)

            # Reset cpu time
            qpoases_cpu_time = np.array([60.])

            # Reset number of of working set recalculations
            nWSR = np.array([10000])

            # First iteration
            res_qpoases = qpoases_m.init(np.ascontiguousarray(qp.P.todense()),
                                         np.ascontiguousarray(qp.q),
                                         np.ascontiguousarray(qp.A.todense()),
                                         np.ascontiguousarray(qp.lx),
                                         np.ascontiguousarray(qp.ux),
                                         np.ascontiguousarray(qp.l),
                                         np.ascontiguousarray(qp.u),
                                         nWSR, qpoases_cpu_time)


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

            if res_qpoases != 0:
                raise ValueError('qpoases did not solve the problem!')

            # Save time
            time[i] = qpoases_cpu_time[0]

            # Save number of iterations
            niter[i] = nWSR[0]

    elif solver == 'gurobi':
        for i in range(n_prob):

            # Solve with gurobi
            prob = mpbpy.QuadprogProblem(qp.P, qp.q, qp.A, qp.l, qp.u)
            res = prob.solve(solver=mpbpy.GUROBI, verbose=False)

            # Save time
            time[i] = res.cputime

            # Save number of iterations
            niter[i] = res.total_iter

    else:
        raise ValueError('Solver not understood')

    # Return statistics
    return utils.Statistics(time), utils.Statistics(niter)




def run_eq_qp_example():
    '''
    Solve problems
    '''

    # Reset random seed for repetibility
    np.random.seed(1)

    # Dimensions
    n_vec = np.array([100, 200, 300])

    # Constraints
    m_vec = (n_vec).astype(int)

    # Number of problems to be solved for each dimension
    n_prob = 10

    # Define statistics for osqp and qpoases
    osqp_timing = []
    osqp_iter = []
    osqp_coldstart_timing = []
    osqp_coldstart_iter = []
    qpoases_timing = []
    qpoases_iter = []
    gurobi_iter = []
    gurobi_timing = []

    for i in range(len(n_vec)):
        # Generate QP
        qp_matrices = gen_qp_matrices(n_vec[i], m_vec[i])

        # Solve loop with osqp
        timing, niter = solve_problem(qp_matrices, n_prob, 'osqp')
        osqp_timing.append(timing)
        osqp_iter.append(niter)

        # Solving loop with qpoases
        timing, niter = solve_problem(qp_matrices, n_prob, 'qpoases')
        qpoases_timing.append(timing)
        qpoases_iter.append(niter)

        # Solve loop with gurobi
        timing, niter = solve_problem(qp_matrices, n_prob, 'gurobi')
        gurobi_timing.append(timing)
        gurobi_iter.append(niter)

    solver_timings = OrderedDict([('OSQP', osqp_timing),
                                  ('qpOASES', qpoases_timing),
                                  ('GUROBI', gurobi_timing)])

    utils.generate_plot('eq_qp', 'time', 'mean', n_vec,
                        solver_timings,
                        fig_size=0.9)

    # solver_max_iter = OrderedDict([('OSQP (warm start)', osqp_iter),
    #                                ('OSQP (cold start)', osqp_coldstart_iter)])
    # utils.generate_plot('qp_qp', 'iter', 'max', n_vec,
    #                     solver_max_iter,
    #                     fig_size=0.9)
