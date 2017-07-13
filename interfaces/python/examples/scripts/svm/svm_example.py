"""
Code for SVM example
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


# Import examples utilities
import scripts.utils as utils


class SVMExample(utils.Example):
    """
    SVM QP example
    """

    def __init__(self, n_vec, solvers):
        """
        Initialize example class

        Arguments:
            name (str): Name of the example
            n_vec (ndarray): Vector of dimension n (different sizes of problem)
            solvers (list): List of solvers
        """
        self.name = "svm"
        self.n_vec = n_vec
        self.solvers = solvers

    def gen_qp_matrices(self, n, dens_lvl=0.5):
        """
        Generate QP matrices for SVM problem
        """
        # Same dimension n and m
        m = 10 * n

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
        A = spa.vstack([spa.hstack([spa.diags(b_svm).dot(A_svm),
                                    -spa.eye(m)]),
                        spa.hstack([spa.csc_matrix((m, n)), spa.eye(m)])
                        ]).tocsc()
        l = np.hstack([-np.inf*np.ones(m), np.zeros(m)])
        u = np.hstack([-np.ones(m), np.inf*np.ones(m)])

        # Variable bounds
        lx = np.append(-np.inf*np.ones(n), np.zeros(m))
        ux = np.append(np.inf*np.ones(n), np.inf*np.ones(m))

        qp_matrices = utils.QPmatrices(P, q, A, l, u, lx, ux)

        # Add further matrices for CVXPY modeling
        qp_matrices.n = n
        qp_matrices.A_svm = A_svm
        qp_matrices.b_svm = b_svm
        qp_matrices.gamma = gamma

        # add further variables for qpOASES
        qp_matrices.A_nobounds = spa.hstack([spa.diags(b_svm).dot(A_svm),
                                             -spa.eye(m)]).tocsc()
        qp_matrices.l_nobounds = -np.inf*np.ones(m)
        qp_matrices.u_nobounds = -np.ones(m)

        # Return QP matrices
        return qp_matrices

    def gen_cvxpy_problem(self, qp):
        # Construct the problem
        #       minimize	 x.T * x + gamma 1.T * t
        #       subject to  t >= diag(b) A x + 1
        #                   t >= 0
        n_var = qp.A_svm.shape[1]
        m_var = qp.A_svm.shape[0]
        x = cvxpy.Variable(n_var)
        t = cvxpy.Variable(m_var)

        objective = cvxpy.Minimize(.5 * cvxpy.quad_form(x, spa.eye(n_var))
                                   + .5 * qp.gamma * np.ones(m_var) * t)
        constraints = [t >= spa.diags(qp.b_svm).dot(qp.A_svm) * x + 1,
                       t >= 0]

        problem = cvxpy.Problem(objective, constraints)

        return problem, (x, t)

    def solve_problem(self, qp_matrices, solver='osqp',
                      osqp_settings=None, n_prob=10):
        """
        Solve SVM problem
        """
        # Shorter name for qp_matrices
        qp = qp_matrices

        # Initialize time vector
        time = np.zeros(n_prob)

        # Initialize number of iterations vector
        niter = np.zeros(n_prob)

        n = qp.n
        # m = len(qp.lx)
        # n = len(qp.q) - m

        print('n = %d and solver %s' % (n, solver))

        if solver == 'osqp':

            for i in range(n_prob):
                # Setup OSQP
                m = osqp.OSQP()
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
                    raise ValueError('OSQP did not solve the problem!')
                if not qp.is_optimal(x, y):
                    raise ValueError('Returned solution not optimal!')

        elif solver == 'qpoases':

            for i in range(n_prob):
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

                # Reset cpu time
                qpoases_cpu_time = np.array([20.])

                # Reset number of working set recalculations
                nWSR = np.array([1000])

                # Solve
                res_qpoases = qpoases_m.init(P, np.ascontiguousarray(qp.q), A,
                                             np.ascontiguousarray(qp.lx),
                                             np.ascontiguousarray(qp.ux),
                                             np.ascontiguousarray(qp.l_nobounds),
                                             np.ascontiguousarray(qp.u_nobounds),
                                             nWSR, qpoases_cpu_time)
                # Check qpoases solution
                x = np.zeros(n_dim)
                y = np.zeros(n_dim + m_dim)
                qpoases_m.getPrimalSolution(x)
                qpoases_m.getDualSolution(y)

                # Get dual variable canonical QP
                y = np.append(-y[n_dim:], -y[qp.n:n_dim])

                if res_qpoases != 0:
                    raise ValueError('qpoases did not solve the problem!')

                if not qp.is_optimal(x, y):
                    raise ValueError('Returned solution not optimal!')

                # Save time and number of iterations
                time[i] = qpoases_cpu_time[0]
                niter[i] = nWSR[0]

        elif solver == 'gurobi':

            for i in range(n_prob):

                # Solve with gurobi
                prob = mpbpy.QuadprogProblem(qp.P, qp.q, qp.A, qp.l, qp.u)
                res = prob.solve(solver=mpbpy.GUROBI, verbose=False)
                niter[i] = res.total_iter
                time[i] = res.cputime

                if not qp.is_optimal(res.x, res.y):
                    raise ValueError('Returned solution not optimal!')

        elif solver == 'mosek':

            for i in range(n_prob):

                # Solve with mosek
                prob = mpbpy.QuadprogProblem(qp.P, qp.q, qp.A, qp.l, qp.u)
                res = prob.solve(solver=mpbpy.MOSEK, verbose=False)
                niter[i] = res.total_iter
                time[i] = res.cputime

                if not qp.is_optimal(res.x, res.y):
                    raise ValueError('Returned solution not optimal!')

        elif solver == 'ecos':

            problem, variables = self.gen_cvxpy_problem(qp)

            for i in range(n_prob):

                problem.solve(solver=cvxpy.ECOS, verbose=False)

                # Check if solved correctly
                (x, t) = variables
                cns = problem.constraints
                x_ecos = np.concatenate((x.value.A1,
                                         t.value.A1))
                y_ecos = np.concatenate((cns[0].dual_value.A1,
                                         -cns[1].dual_value.A1))

                prob = mpbpy.QuadprogProblem(qp.P, qp.q, qp.A, qp.l, qp.u)
                res = prob.solve(solver=mpbpy.MOSEK, verbose=False)

                if not qp.is_optimal(x_ecos, y_ecos):
                    raise ValueError('Returned solution not optimal')

                # Obtain time and number of iterations
                time[i] = problem.solver_stats.setup_time + \
                    problem.solver_stats.solve_time

                niter[i] = problem.solver_stats.num_iters

        else:
            raise ValueError('Solver not understood')

        # Return statistics
        return utils.Statistics(time), utils.Statistics(niter)
