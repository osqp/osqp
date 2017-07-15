"""
Run equality constrained QP example
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

class EqqpExample(utils.Example):
    """
    Equality constrained QP class
    """

    def __init__(self, n_vec, solvers):
        """
        Initialize example class

        Arguments:
            name (str): Name of the example
            n_vec (ndarray): Vector of dimension n (different sizes of problem)
            solvers (list): List of solvers
        """
        self.name = "eq_qp"
        self.n_vec = n_vec
        self.solvers = solvers

    def gen_qp_matrices(self, n, dens_lvl=0.5):
        """
        Generate QP matrices for the quality constrained QP problem
        """
        # Same dimension n and m
        m = n

        # Construct problem
        #       minimize	1/2 x' P x + q'*x
        #       subject to  A * x = b
        #
        P = spa.random(n, n, density=dens_lvl, format='csc')
        P = .1 * P.dot(P.T).tocsc()
        q = np.random.randn(n)

        A = spa.random(m, n, density=dens_lvl, format='csc')
        l = np.random.randn(m)
        u = np.copy(l)

        lx = -np.inf * np.ones(n)
        ux = np.inf * np.ones(n)

        qp_matrices = utils.QPmatrices(P, q, A, l, u, lx, ux)

        # Add further details related to the problem
        qp_matrices.n = n

        # Return QP matrices
        return qp_matrices

    def gen_cvxpy_problem(self, n, qp):
        x_var = cvxpy.Variable(n)
        objective = cvxpy.Minimize(.5 * cvxpy.quad_form(x_var, qp.P) +
                                   qp.q * x_var)
        constraints = [qp.A * x_var <= qp.u, qp.A * x_var >= qp.l]
        # constraints = [qp.A * x_var == qp.u]
        problem = cvxpy.Problem(objective, constraints)
        return problem, x_var

    def solve_problem(self, qp_matrices, solver='osqp',
                      osqp_settings=None, n_prob=10):
        """
        Solve equality constrained optimization
        """
        # Shorter name for qp_matrices
        qp = qp_matrices

        # Get dimensions
        n = len(qp.q)
        m = len(qp.l)

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

                    if not qp.is_optimal(x, y):
                        print('Returned solution not optimal!')

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
                qpoases_cpu_time = np.array([20.])

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

                if res_qpoases != 0:
                    print('qpoases did not solve the problem!')
                else:
                    # Get qpoases solution
                    x = np.zeros(n_dim)
                    y = np.zeros(n_dim + m_dim)
                    qpoases_m.getPrimalSolution(x)
                    qpoases_m.getDualSolution(y)
                    y = -y[n_dim:]

                    # Save time
                    time[i] = qpoases_cpu_time[0]

                    # Save number of iterations
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
                    # Save time
                    time[i] = res.cputime

                    # Save number of iterations
                    niter[i] = res.total_iter

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
                    # Save time
                    time[i] = res.cputime

                    # Save number of iterations
                    niter[i] = res.total_iter

                    if not qp.is_optimal(res.x, res.y):
                        print('Returned solution not optimal!')

        elif solver == 'ecos':

            problem, x = self.gen_cvxpy_problem(n, qp)

            for i in range(n_prob):

                # Solve with ECOS (via CVXPY)
                problem.solve(solver=cvxpy.ECOS, verbose=False)

                # Check if solved
                constraints = problem.constraints
                x_ecos = x.value.A1
                y_ecos = constraints[0].dual_value.A1 - \
                    constraints[1].dual_value.A1

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
