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


# Utilities
import scripts.utils as utils


class LassoExample(utils.Example):
    """
    Lasso example class
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
        self.name = "lasso"
        self.n_vec = n_vec
        self.solvers = solvers
        self.parameter = parameter

    def gen_qp_matrices(self, n, dens_lvl=0.5):
        """
        Generate QP matrices for lasso optimization problem
        """

        # 100 more datapoints than features
        m = n * 100

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
        for lambda_i in self.parameter:
            q = np.column_stack((q, np.append(np.zeros(m + n),
                                              lambda_i*np.ones(n))
                                 ))

        qp_matrices = utils.QPmatrices(P, q, A, l, u, lx, ux)

        # Add further data related to this problem
        qp_matrices.A_lasso = Ad
        qp_matrices.b_lasso = bd
        qp_matrices.n = n

        # Return QP matrices
        return qp_matrices

    def gen_cvxpy_problem(self, n, m, qp_matrices):

        lambda_i = cvxpy.Parameter(sign="positive")
        x = cvxpy.Variable(n)
        y = cvxpy.Variable(m)
        t = cvxpy.Variable(n)

        objective = cvxpy.Minimize(cvxpy.quad_form(y, spa.eye(m))
                                   + lambda_i * (np.ones(n) * t))
        constraints = [y == qp_matrices.A_lasso * x - qp_matrices.b_lasso,
                       -t <= x, x <= t]
        problem = cvxpy.Problem(objective, constraints)
        return problem, lambda_i, (x, y, t)

    def solve_problem(self, qp_matrices, solver='osqp', osqp_settings=None):

        # Shorter name for qp_matrices
        qp = qp_matrices

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
                    raise ValueError('OSQP did not solve the problem!')

                if not qp.is_optimal(x, y):
                    raise ValueError('Returned solution not optimal!')

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
                niter[i] = results.info.iter
                time[i] = results.info.run_time

                # Check if status correct
                if status != m.constant('OSQP_SOLVED'):
                    raise ValueError('OSQP did not solve the problem!')

                if not qp.is_optimal(x, y):
                    raise ValueError('Returned solution not optimal!')

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
                    raise ValueError('OSQP did not solve the problem!')

                if not qp.is_optimal(x, y):
                    raise ValueError('Returned solution not optimal!')

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
                qpoases_cpu_time = np.array([20.])

                # Reset number of of working set recalculations
                nWSR = np.array([10000])

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
                    res_qpoases = \
                        qpoases_m.hotstart(qp.q,
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
                y = -y[n_dim:]

                if res_qpoases != 0:
                    raise ValueError('qpoases did not solve the problem!')

                if not qp.is_optimal(x, y):
                    raise ValueError('Returned solution not optimal!')

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

                if not qp.is_optimal(res.x, res.y):
                    raise ValueError('Returned solution not optimal!')

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

                if not qp.is_optimal(res.x, res.y):
                    raise ValueError('Returned solution not optimal!')

                # Save time
                time[i] = res.cputime

                # Save number of iterations
                niter[i] = res.total_iter

        elif solver == 'ecos':

            n_var = qp.A_lasso.shape[1]
            m_var = qp.A_lasso.shape[0]

            problem, lambda_i, variables = self.gen_cvxpy_problem(n_var,
                                                                  m_var, qp)

            for i in range(n_prob):
                # if n_var <= 60:  # (problem becomes too big otherwise):

                    # Model with CVXPY
                    #       minimize	y' * y + lambda * 1' * t
                    #       subject to  y = Ax - b
                    #                   -t <= x <= t
                    lambda_i.value = self.parameter[i]
                    problem.solve(solver=cvxpy.ECOS, verbose=False)


                    # Check if solved
                    (x, y, t) = variables
                    constraints = problem.constraints
                    x_ecos = np.concatenate((x.value.A1,
                                             y.value.A1,
                                             t.value.A1))
                    y_ecos = np.concatenate((-constraints[0].dual_value.A1,
                                             constraints[2].dual_value.A1,
                                             -constraints[1].dual_value.A1))

                    qp.q = qp.q_vec[:, i]
                    if problem.status != cvxpy.OPTIMAL:
                        raise ValueError("ECOS did not solve the problem")
                    if not qp.is_optimal(x_ecos, y_ecos):
                        raise ValueError('Returned solution not optimal')

                    # Obtain time and number of iterations
                    time[i] = problem.solver_stats.setup_time + \
                        problem.solver_stats.solve_time

                    niter[i] = problem.solver_stats.num_iters
                # else:
                    # time[i] = 0
                    # niter[i] = 0

        else:
            raise ValueError('Solver not understood')

        # Return statistics
        return utils.Statistics(time), utils.Statistics(niter)
