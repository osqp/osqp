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
import scripts.utils as utils


class PortfolioExample(utils.Example):
    """
    Portfolio example class
    """

    def __init__(self, n_vec, solvers, parameter):
        """
        Initialize example class

        Arguments:
            name (str): Name of the example
            n_vec (ndarray): Vector of dimension n (different sizes of problem)
            solvers (list): List of solvers
            parameter (ndarray): Parameter to iterate over
                                 during loops
        """
        self.name = "portfolio"
        self.n_vec = n_vec
        self.solvers = solvers
        self.parameter = parameter

    def gen_qp_matrices(self, n, dens_lvl=0.5):
        """
        Generate QP matrices for portfolio optimization problem
        """

        # Factors
        k = (n / 100).astype(int)

        # Generate data
        F = spa.random(n, k, density=dens_lvl,
                       data_rvs=np.random.randn, format='csc')
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
                spa.hstack([F.T, -spa.eye(k)]),
                spa.hstack((spa.eye(n), spa.csc_matrix((n, k))))
            ]).tocsc()
        l = np.hstack([1., np.zeros(k), np.zeros(n)])   # Linear constraints
        u = np.hstack([1., np.zeros(k), np.ones(n)])


        lx = np.hstack([np.zeros(n), -np.inf * np.ones(k)])   # Bounds
        ux = np.hstack([np.ones(n), np.inf * np.ones(k)])

        # Create linear cost vectors
        q = np.empty((k + n, 0))
        for gamma in self.parameter:
            q = np.column_stack((q, np.append(-mu / gamma, np.zeros(k))))

        qp_matrices = utils.QPmatrices(P, q, A, l, u, lx, ux)

        # Add further matrices for CVXPY modeling
        qp_matrices.n = n
        qp_matrices.k = k
        qp_matrices.F = F
        qp_matrices.D = D
        qp_matrices.mu = mu

        # Other stuff for
        qp_matrices.A_nobounds = spa.vstack([
                spa.hstack([spa.csc_matrix(np.ones((1, n))),
                            spa.csc_matrix((1, k))]),
                spa.hstack([F.T, -spa.eye(k)]),
                ]).tocsc()
        qp_matrices.l_nobounds = np.hstack([1., np.zeros(k)])
        qp_matrices.u_nobounds = np.hstack([1., np.zeros(k)])

        # Return QP matrices
        return qp_matrices

    def gen_cvxpy_problem(self, qp):
        # Construct the problem
        #       minimize	x' D x + y' I y - (1/gamma) * mu' x
        #       subject to  1' x = 1
        #                   F' x = y
        #                   0 <= x <= 1

        # gamma parameter
        gamma = cvxpy.Parameter(sign="positive")

        n_var = qp.F.shape[0]
        m_var = qp.F.shape[1]
        x = cvxpy.Variable(n_var)
        y = cvxpy.Variable(m_var)

        objective = cvxpy.Minimize(cvxpy.quad_form(x, qp.D) +
                                   cvxpy.quad_form(y, spa.eye(m_var)) +
                                   - 1 / gamma * (qp.mu * x))
        constraints = [np.ones(n_var) * x == 1,
                       qp.F.T * x == y,
                       0 <= x, x <= 1]
        problem = cvxpy.Problem(objective, constraints)

        return problem, gamma, (x, y)

    def solve_problem(self, qp_matrices, solver='osqp', osqp_settings=None):
        """
        Solve portfolio optimization loop for all gammas
        """
        # Shorter name for qp_matrices
        qp = qp_matrices

        print('n = %d and solver %s' %
              (qp.n, solver))

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

                # Check if status correct
                if status != m.constant('OSQP_SOLVED'):
                    print('OSQP did not solve the problem!')
                else:
                    niter[i] = results.info.iter
                    time[i] = results.info.run_time

                    if results.info.status_polish == -1:
                            print('Polish failed!')

                if not qp.is_optimal(x, y):
                    print('Returned solution not optimal!')

        elif solver == 'osqp_coldstart':
            # Setup OSQP
            m = osqp.OSQP()
            m.setup(qp.P, qp.q_vec[:, 0], qp.A, qp.l, qp.u,
                    warm_start=False, **osqp_settings)

            for i in range(n_prob):
                qp.q = qp.q_vec[:, i]

                # Update linear cost
                m.update(q=qp.q)

                # Solve
                results = m.solve()
                x = results.x
                y = results.y
                status = results.info.status_val
                # Check if status correct
                if status != m.constant('OSQP_SOLVED'):
                    print('OSQP did not solve the problem!')
                else:
                    niter[i] = results.info.iter
                    time[i] = results.info.run_time

                    if results.info.status_polish == -1:
                            print('Polish failed!')

                    if not qp.is_optimal(x, y):
                        print('Returned solution not optimal!')

        elif solver == 'osqp_no_caching':

            for i in range(n_prob):

                qp.q = qp.q_vec[:, i]

                # Setup OSQP
                m = osqp.OSQP()
                m.setup(qp.P, qp.q_vec[:, i], qp.A, qp.l, qp.u,
                        **osqp_settings)

                # Solve
                results = m.solve()
                x = results.x
                y = results.y
                status = results.info.status_val
                # Check if status correct
                if status != m.constant('OSQP_SOLVED'):
                    print('OSQP did not solve the problem!')
                else:
                    niter[i] = results.info.iter
                    time[i] = results.info.run_time

                    if results.info.status_polish == -1:
                            print('Polish failed!')

                    if not qp.is_optimal(x, y):
                        print('Returned solution not optimal!')

        elif solver == 'qpoases':

            n_dim = qp.P.shape[0]  # Number of variables
            m_dim = qp.A_nobounds.shape[0]  # Number of constraints without bounds

            # Initialize qpoases and set options
            qpoases_m = qpoases.PyQProblem(n_dim, m_dim)
            options = qpoases.PyOptions()
            options.printLevel = qpoases.PyPrintLevel.NONE
            qpoases_m.setOptions(options)

            # Setup matrix P and A
            P = np.ascontiguousarray(qp.P.todense())
            A = np.ascontiguousarray(qp.A_nobounds.todense())

            for i in range(n_prob):

                # Get linera cost as contiguous array
                qp.q = np.ascontiguousarray(qp.q_vec[:, i])

                # Reset cpu time
                qpoases_cpu_time = np.array([20.])

                # Reset number of of working set recalculations
                nWSR = np.array([10000])

                if i == 0:
                    # First iteration
                    res_qpoases = qpoases_m.init(P, qp.q, A,
                                                 np.ascontiguousarray(qp.lx),
                                                 np.ascontiguousarray(qp.ux),
                                                 np.ascontiguousarray(qp.l_nobounds),
                                                 np.ascontiguousarray(qp.u_nobounds),
                                                 nWSR, qpoases_cpu_time)
                else:
                    # Solve new hot started problem
                    res_qpoases = qpoases_m.hotstart(qp.q,
                                                     np.ascontiguousarray(qp.lx),
                                                     np.ascontiguousarray(qp.ux),
                                                     np.ascontiguousarray(qp.l_nobounds),
                                                     np.ascontiguousarray(qp.u_nobounds),
                                                     nWSR,
                                                     qpoases_cpu_time)

                # qpoases solution
                x = np.zeros(n_dim)
                y = np.zeros(n_dim + m_dim)
                qpoases_m.getPrimalSolution(x)
                qpoases_m.getDualSolution(y)
                y = np.append(-y[n_dim:], -y[:qp.n])

                # import ipdb; ipdb.set_trace()

                if res_qpoases != 0:
                    print('qpoases did not solve the problem!')
                else:
                    # Save time and number of iterations
                    time[i] = qpoases_cpu_time[0]
                    niter[i] = nWSR[0]

                    if not qp.is_optimal(x, y):
                        print('Returned solution not optimal!')

        elif solver == 'gurobi':

            for i in range(n_prob):

                # Get linera cost as contiguous array
                qp.q = qp.q_vec[:, i]

                # Solve with gurobi
                prob = mpbpy.QuadprogProblem(qp.P, qp.q, qp.A, qp.l, qp.u)
                res = prob.solve(solver=mpbpy.GUROBI, verbose=False)

                if res.status != 'optimal' and \
                        res.status != 'optimal inaccurate':
                    print('GUROBI did not solve the problem!')
                else:

                    niter[i] = res.total_iter
                    time[i] = res.cputime

                    if not qp.is_optimal(res.x, res.y):
                        print('Returned solution not optimal!')

        elif solver == 'mosek':

            for i in range(n_prob):

                # Get linera cost as contiguous array
                qp.q = qp.q_vec[:, i]

                # Solve with mosek
                prob = mpbpy.QuadprogProblem(qp.P, qp.q, qp.A, qp.l, qp.u)
                res = prob.solve(solver=mpbpy.MOSEK, verbose=False)

                if res.status != 'optimal' and \
                        res.status != 'optimal inaccurate':
                    print('MOSEK did not solve the problem!')
                else:
                    niter[i] = res.total_iter
                    time[i] = res.cputime

                    if not qp.is_optimal(res.x, res.y):
                        print('Returned solution not optimal!')

        elif solver == 'ecos':

            problem, gamma, variables = self.gen_cvxpy_problem(qp)
            for i in range(n_prob):
                gamma.value = self.parameter[i]
                problem.solve(solver=cvxpy.ECOS, verbose=False)

                # Check if solved
                (x, y) = variables
                cons = problem.constraints
                x_ecos = np.concatenate((x.value.A1,
                                         y.value.A1))
                y_ecos = np.concatenate(([cons[0].dual_value],
                                         cons[1].dual_value.A1,
                                         cons[3].dual_value.A1 -
                                         cons[2].dual_value.A1))

                qp.q = qp.q_vec[:, i]
                if problem.status != 'optimal' and \
                        problem.status != 'optimal inaccurate':
                    print('ECOS did not solve the problem!')
                else:
                    # Obtain time and number of iterations
                    time[i] = problem.solver_stats.setup_time + \
                        problem.solver_stats.solve_time

                    niter[i] = problem.solver_stats.num_iters

                    if not qp.is_optimal(x_ecos, y_ecos):
                        print('Returned solution not optimal')

        else:
            raise ValueError('Solver not understood')

        # Return statistics
        return utils.Statistics(time), utils.Statistics(niter)
