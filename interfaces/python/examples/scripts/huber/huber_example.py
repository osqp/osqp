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

# Import examples utilities
import scripts.utils as utils


class HuberExample(utils.Example):
    """
    Huber example class
    """

    def __init__(self, n_vec, solvers):
        """
        Initialize example class

        Arguments:
            name (str): Name of the example
            n_vec (ndarray): Vector of dimension n (different sizes of problem)
            solvers (list): List of solvers

        """
        self.name = "huber"
        self.n_vec = n_vec
        self.solvers = solvers

    def gen_qp_matrices(self, n, dens_lvl=0.5):
        """
        Generate QP matrices for Huber fitting problem
        """

        # 10 more datapoints than features
        m = int(n * 100)

        # Generate data
        A_huber = spa.random(m, n, density=dens_lvl,
                             data_rvs=np.random.randn, format='csc')
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
                spa.hstack([A_huber, -Im, -Im]),
                spa.hstack([spa.csc_matrix((2*m, n)), spa.eye(2*m)])
                ]).tocsc()
        l = np.hstack([b_huber,
                       -np.inf*np.ones(m),
                       np.zeros(2*m)])
        u = np.hstack([np.inf*np.ones(m),
                       b_huber,
                       np.ones(m),
                       np.inf*np.ones(m)])

        # Bounds on (u,v)
        lx = np.hstack([-np.inf * np.ones(n), np.zeros(2*m)])
        ux = np.hstack([np.inf * np.ones(n), np.ones(m), np.inf*np.ones(m)])

        qp_matrices = utils.QPmatrices(P, q, A, l, u, lx, ux)

        # Add further details related to the problem
        qp_matrices.A_huber = A_huber
        qp_matrices.b_huber = b_huber
        qp_matrices.A_nobounds = spa.vstack([
                spa.hstack([A_huber, Im, Im]),
                spa.hstack([A_huber, -Im, -Im])
                ]).tocsc()
        qp_matrices.l_nobounds = np.hstack([b_huber,
                                            -np.inf*np.ones(m)])
        qp_matrices.u_nobounds = np.hstack([np.inf*np.ones(m),
                                            b_huber])
        qp_matrices.m = m
        qp_matrices.n = n

        # Return QP matrices
        return qp_matrices

    def gen_cvxpy_problem(self, n_var, m_var, qp):
        # Model with CVXPY
        #       minimize	1/2 u.T * u + np.ones(m).T * v
        #       subject to  -u - v <= Ax - b <= u + v
        #                   0 <= u <= 1
        #                   v >= 0
        x = cvxpy.Variable(n_var)
        u = cvxpy.Variable(m_var)
        v = cvxpy.Variable(m_var)

        objective = cvxpy.Minimize(.5 * cvxpy.quad_form(u, spa.eye(m_var))
                                   + np.ones(m_var) * v)
        constraints = [-u - v <= qp.A_huber * x - qp.b_huber,
                       qp.A_huber * x - qp.b_huber <= u + v,
                       0. <= u, u <= 1.,
                       v >= 0.]
        problem = cvxpy.Problem(objective, constraints)

        return problem, (x, u, v)

    def solve_problem(self, qp_matrices, solver='osqp',
                      osqp_settings=None, n_prob=10):
        """
        Solve Huber fitting problem
        """
        # Shorter name for qp_matrices
        qp = qp_matrices

        # Get dimensions
        m = qp_matrices.m
        n = qp_matrices.n

        print('n = %d and solver %s' % (n, solver))

        # Initialize time vector
        time = np.zeros(n_prob)

        # Initialize number of iterations vector
        niter = np.zeros(n_prob)

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

            for i in range(n_prob):
                n_dim = qp.P.shape[0]  # Number of variables
                m_dim = qp.A_nobounds.shape[0]  # Number of constraints without bounds

                # Initialize qpoases and set options
                qpoases_m = qpoases.PyQProblem(n_dim, m_dim)
                options = qpoases.PyOptions()
                options.printLevel = qpoases.PyPrintLevel.NONE
                qpoases_m.setOptions(options)

                # Construct bounds for qpoases
                # lx = np.append(-np.inf*np.ones(n), qp.lx)
                # ux = np.append(np.inf*np.ones(n), qp.ux)

                # Setup matrix P and A
                P = np.ascontiguousarray(qp.P.todense())
                A = np.ascontiguousarray(qp.A_nobounds.todense())

                # Reset cpu time
                qpoases_cpu_time = np.array([20.])

                # Reset number of working set recalculations
                nWSR = np.array([10000])

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
                    print('qpoases did not solve the problem!')
                else:
                    # Save time and number of iterations
                    time[i] = qpoases_cpu_time[0]
                    niter[i] = nWSR[0]

                    if not qp.is_optimal(x, y):
                        print('Returned solution not optimal!')



        elif solver == 'gurobi':

            for i in range(n_prob):

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

            problem, variables = self.gen_cvxpy_problem(n, m, qp)

            for i in range(n_prob):

                # Solve with ECOS (via CVXPY)
                problem.solve(solver=cvxpy.ECOS, verbose=False)

                # Check if solved
                (x, u, v) = variables
                cns = problem.constraints
                x_ecos = np.concatenate((x.value.A1,
                                         u.value.A1,
                                         v.value.A1))
                y_ecos = np.concatenate((-cns[0].dual_value.A1,
                                         cns[1].dual_value.A1,
                                         cns[3].dual_value.A1 -
                                         cns[2].dual_value.A1,
                                         -cns[4].dual_value.A1))

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
