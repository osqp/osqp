"""
Code for MPC example
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

# Load MPC examples data
from .examples.ball import load_ball_data
from .examples.helicopter import load_helicopter_data
from .examples.pendulum import load_pendulum_data
from .examples.quadcopter import load_quadcopter_data


class MPCExample(utils.Example):

    def __init__(self, n_vec, solvers, problem_name):
        """
        Initialize example class

        Arguments:
            name (str): Name of the example
            n_vec (ndarray): Vector of dimension n (different sizes of problem)
            solvers (list): List of solvers
            parameter (ndarray): Parameter to iterate over
                                 during loops
        """
        self.name = "mpc"
        self.n_vec = n_vec
        self.solvers = solvers
        self.problem_name = problem_name

        # Load the example
        if problem_name == 'pendulum':
            self.problem = load_pendulum_data()
        elif problem_name == 'helicopter':
            self.problem = load_helicopter_data()
        elif problem_name == 'quadcopter':
            self.problem = load_quadcopter_data()
        elif problem_name == 'ball':
            self.problem = load_ball_data()
        else:
            self.problem = load_ball_data()  # Default data

    # RHS of linear equality constraint in sparse MPC variant
    def b(self, x, nx, N):
        b = np.zeros((N+1)*nx)
        b[:nx] = -x
        return b

    def gen_qp_matrices(self, N):
        """
        Generate QP matrices for MPC problem
        """
        # Problem
        problem = self.problem

        # Get dimensions
        (nx, nu) = problem.B.shape

        # Objective
        Px = spa.kron(spa.eye(N), problem.Q)
        Pu = spa.kron(spa.eye(N), problem.R)
        P = spa.block_diag([Px, problem.QN, Pu]).tocsc()

        if problem.xr is not None:
            q = np.hstack([np.kron(np.ones(N), -problem.Q.dot(problem.xr)),
                           -problem.QN.dot(problem.xr),
                           np.zeros(N*nu)])
        else:
            q = np.zeros((N+1)*nx + N*nu)

        # Dynamics
        Ax = spa.kron(spa.eye(N+1), -spa.eye(nx)) + \
            spa.kron(spa.eye(N+1, k=-1), problem.A)
        Au = spa.kron(spa.vstack([spa.csc_matrix((1, N)), spa.eye(N)]),
                      problem.B)
        A = spa.hstack([Ax, Au])
        l = self.b(problem.x0, nx, N)
        u = self.b(problem.x0, nx, N)

        # Terminal constraints
        if problem.tmin is not None:
            nt = problem.T.shape[0]
            A = spa.vstack([A, spa.hstack([spa.csc_matrix((nt, N*nx)),
                                           problem.T,
                                           spa.csc_matrix((nt, N*nu))])])
            l = np.append(l, problem.tmin)
            u = np.append(u, problem.tmax)

        # Store values of problem without variable bound (for qpOASES)
        A_nobounds = A.copy()
        l_nobounds = np.copy(l)
        u_nobounds = np.copy(u)

        # Add bounds on x and u (for qpOASES)
        lx = np.array([])
        ux = np.array([])

        # Bounds on x
        if problem.xmin is not None:
            l = np.append(l, np.tile(problem.xmin, N+1))
            u = np.append(u, np.tile(problem.xmax, N+1))
            A = spa.vstack([A,
                            spa.hstack([spa.eye((N+1)*nx),
                                        spa.csc_matrix(((N+1)*nx, N*nu))])
                            ]).tocsc()
            lx = np.append(lx, np.tile(problem.xmin, N+1))
            ux = np.append(ux, np.tile(problem.xmax, N+1))
        else:
            lx = np.append(lx, -np.inf * np.ones(nx * (N+1)))
            ux = np.append(ux, np.inf * np.ones(nx * (N+1)))

        # Bounds on u
        if problem.umin is not None:
            l = np.append(l, np.tile(problem.umin, N))
            u = np.append(u, np.tile(problem.umax, N))
            A = spa.vstack([A,
                            spa.hstack([spa.csc_matrix((N*nu, (N+1)*nx)),
                                        spa.eye(N*nu)])
                            ]).tocsc()
            lx = np.append(lx, np.tile(problem.umin, N))
            ux = np.append(ux, np.tile(problem.umax, N))
        else:
            lx = np.append(lx, -np.inf * np.ones(nu * N))
            ux = np.append(ux, np.inf * np.ones(nu * N))

        # Return QP matrices
        qp_matrices = utils.QPmatrices(P, q, A, l, u, lx, ux)

        # Add additional elements to qp_matrices
        qp_matrices.n = N
        qp_matrices.nsim = 200  # Assume 200 steps of simulation
        qp_matrices.A_nobounds = A_nobounds
        qp_matrices.l_nobounds = l_nobounds
        qp_matrices.u_nobounds = u_nobounds

        return qp_matrices

    def update_initial_state(self, qp, x0):
        """
        Update initial state in qp solver formulation
        """
        nx = self.problem.B.shape[0]
        N = qp.n
        qp.l[:(N+1)*nx] = self.b(x0, nx, N)
        qp.u[:(N+1)*nx] = qp.l[:(N+1)*nx]
        qp.l_nobounds[:(N+1)*nx] = self.b(x0, nx, N)
        qp.u_nobounds[:(N+1)*nx] = qp.l_nobounds[:(N+1)*nx]

    def gen_cvxpy_problem(self, qp):
        (nx, nu) = self.problem.B.shape
        N = qp.n

        # Initial state
        x0 = cvxpy.Parameter(nx)

        # Problem
        p = self.problem

        # variables
        x = cvxpy.Variable(nx, N + 1)
        u = cvxpy.Variable(nu, N)

        # Objective
        cost = .5 * cvxpy.quad_form((x[:, N] - p.xr), p.QN)  # Final stage cost
        for i in range(N):
            cost += .5 * cvxpy.quad_form((x[:, i] - p.xr), p.Q)  # State cost
            cost += .5 * cvxpy.quad_form(u[:, i], p.R)  # Inpout cost
        objective = cvxpy.Minimize(cost)

        # Constraints
        constr = []
        # Linear Dynamics
        constr += [x[:, 0] == x0]
        for i in range(0, N):
            constr += [x[:, i+1] == p.A * x[:, i] + p.B * u[:, i]]

        # Terminal constraints
        if p.tmin is not None:
            constr += [p.tmin <= p.T * x[:, N], p.T * x[:, N] <= p.tmax]

        # State constraints
        if p.xmin is not None:
            for i in range(N + 1):
                constr += [p.xmin <= x[:, i], x[:, i] <= p.xmax]

        # Input constraints
        if p.umin is not None:
            for i in range(N):
                constr += [p.umin <= u[:, i], u[:, i] <= p.umax]

        problem = cvxpy.Problem(objective, constr)

        return problem, x0, (x, u)

    def solve_problem(self, qp_matrices, solver='osqp', osqp_settings=None):
        problem = self.problem
        nsim = qp_matrices.nsim

        """
        Solve MPC loop
        """
        # Shorter name for qp_matrices
        qp = qp_matrices

        # Get dimensions
        (nx, nu) = problem.B.shape
        N = qp_matrices.n

        print('N = %d and solver %s' % (N, solver))

        # Initialize time and number of iterations vectors
        time = np.zeros(nsim)
        niter = np.zeros(nsim)

        # Initialize bounds with initial state
        self.update_initial_state(qp, problem.x0)
        x_sys = np.zeros((nx, nsim+1))
        x_sys[:, 0] = problem.x0
        u_sys = np.zeros((nu, nsim))

        if solver == 'osqp':

            # Setup OSQP
            m = osqp.OSQP()
            m.setup(qp.P, qp.q, qp.A, qp.l, qp.u, **osqp_settings)

            for i in range(nsim):
                # Solve with osqp
                res = m.solve()

                # Save time and number of iterations
                time[i] = res.info.run_time
                niter[i] = res.info.iter

                # Check if status is correct
                status = res.info.status_val

                # Check if status correct
                if status != m.constant('OSQP_SOLVED'):
                    print('OSQP did not solve the problem!')
                    # # Dump file to 'bad_convergence/data'folder
                    # import pickle
                    # problem = {'P': qp.P,
                    #            'q': qp.q,
                    #            'A': qp.A,
                    #            'l': qp.l,
                    #            'u': qp.u}
                    # with open('bad_convergence/data/%s.pickle' % 'helicopter_balanced_residuals', 'wb') as f:
                    #     pickle.dump(problem, f)
                else:
                    niter[i] = res.info.iter
                    time[i] = res.info.run_time

                    if res.info.status_polish == -1:
                            print('Polish failed!')

                    if not qp.is_optimal(res.x, res.y):
                        print('Returned solution not optimal!')

                # Apply first control input to the plant
                u_sys[:, i] = res.x[-N*nu:-(N-1)*nu]

                # x_{k+1} = Ax_{t} + Bu_{t}
                x_sys[:, i+1] = problem.A.dot(x_sys[:, i]) + \
                    problem.B.dot(u_sys[:, i])

                # Update linear constraints
                self.update_initial_state(qp, x_sys[:, i+1])

                # Change l and u
                m.update(l=qp.l, u=qp.u)

        elif solver == 'osqp_coldstart':
            # Setup OSQP
            m = osqp.OSQP()
            m.setup(qp.P, qp.q, qp.A, qp.l, qp.u,
                    warm_start=False, **osqp_settings)

            for i in range(nsim):
                # Solve with osqp
                res = m.solve()

                # Save time and number of iterations
                time[i] = res.info.run_time
                niter[i] = res.info.iter

                # Check if status is correct
                status = res.info.status_val
                # Check if status correct
                if status != m.constant('OSQP_SOLVED'):
                    print('OSQP did not solve the problem!')
                else:
                    niter[i] = res.info.iter
                    time[i] = res.info.run_time

                    if res.info.status_polish == -1:
                            print('Polish failed!')

                    if not qp.is_optimal(res.x, res.y):
                        print('Returned solution not optimal!')

                # Apply first control input to the plant
                u_sys[:, i] = res.x[-N*nu:-(N-1)*nu]

                # x_{k+1} = Ax_{t} + Bu_{t}
                x_sys[:, i+1] = problem.A.dot(x_sys[:, i]) + \
                    problem.B.dot(u_sys[:, i])

                # Update linear constraints
                self.update_initial_state(qp, x_sys[:, i+1])

                # Change l and u
                m.update(l=qp.l, u=qp.u)

        elif solver == 'osqp_no_caching':

            for i in range(nsim):
                # Setup OSQP
                m = osqp.OSQP()
                m.setup(qp.P, qp.q, qp.A, qp.l, qp.u, **osqp_settings)
                # Solve
                res = m.solve()

                # Save time and number of iterations
                time[i] = res.info.run_time
                niter[i] = res.info.iter

                # Check if status is correct
                status = res.info.status_val
                # Check if status correct
                if status != m.constant('OSQP_SOLVED'):
                    print('OSQP did not solve the problem!')
                else:
                    niter[i] = res.info.iter
                    time[i] = res.info.run_time

                    if res.info.status_polish == -1:
                            print('Polish failed!')

                    if not qp.is_optimal(res.x, res.y):
                        print('Returned solution not optimal!')

                # Apply first control input to the plant
                u_sys[:, i] = res.x[-N*nu:-(N-1)*nu]

                # x_{k+1} = Ax_{t} + Bu_{t}
                x_sys[:, i+1] = problem.A.dot(x_sys[:, i]) + \
                    problem.B.dot(u_sys[:, i])

                # Update linear constraints
                self.update_initial_state(qp, x_sys[:, i+1])

                # Change l and u
                m.update(l=qp.l, u=qp.u)

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

            for i in range(nsim):
                # Reset cpu time
                qpoases_cpu_time = np.array([20.])

                # Reset number of of working set recalculations
                nWSR = np.array([10000])

                if i == 0:
                    # First iteration
                    res_qpoases = qpoases_m.init(P, np.ascontiguousarray(qp.q),
                                                 A,
                                                 np.ascontiguousarray(qp.lx),
                                                 np.ascontiguousarray(qp.ux),
                                                 np.ascontiguousarray(qp.l_nobounds),
                                                 np.ascontiguousarray(qp.u_nobounds),
                                                 nWSR, qpoases_cpu_time)
                else:
                    # Solve new hot started problem
                    res_qpoases = qpoases_m.hotstart(np.ascontiguousarray(qp.q),
                                                     np.ascontiguousarray(qp.lx),
                                                     np.ascontiguousarray(qp.ux),
                                                     np.ascontiguousarray(qp.l_nobounds),
                                                     np.ascontiguousarray(qp.u_nobounds),
                                                     nWSR,
                                                     qpoases_cpu_time)

                # Check qpoases solution
                x_qpoases = np.zeros(n_dim)
                y_qpoases = np.zeros(n_dim + m_dim)
                qpoases_m.getPrimalSolution(x_qpoases)
                qpoases_m.getDualSolution(y_qpoases)
                x = x_qpoases
                y = -y_qpoases[n_dim:]

                if problem.xmin is not None and problem.umin is not None:
                    y = np.append(y, -y_qpoases[:n_dim])
                elif problem.xmin is None and problem.umin is not None:
                    y = np.append(y, -y_qpoases[nx * (N+1):n_dim])
                elif problem.xmin is not None and problem.umin is not None:
                    y = np.append(y, -y_qpoases[: nx * (N+1)])

                if res_qpoases != 0:
                    print('qpoases did not solve the problem!')
                else:
                    # Save time and number of iterations
                    time[i] = qpoases_cpu_time[0]
                    niter[i] = nWSR[0]

                    if not qp.is_optimal(x, y):
                        print('Returned solution not optimal!')

                # Apply first control input to the plant
                u_sys[:, i] = x[-N*nu:-(N-1)*nu]

                # x_{k+1} = Ax_{t} + Bu_{t}
                x_sys[:, i+1] = problem.A.dot(x_sys[:, i]) + \
                    problem.B.dot(u_sys[:, i])

                # Update linear constraints
                self.update_initial_state(qp, x_sys[:, i+1])

        elif solver == 'gurobi':
            for i in range(nsim):
                # Solve with gurobi
                prob = mpbpy.QuadprogProblem(qp.P, qp.q, qp.A,
                                             qp.l, qp.u)
                res = prob.solve(solver=mpbpy.GUROBI, verbose=False)

                if res.status != 'optimal' and \
                        res.status != 'optimal inaccurate':
                    print('GUROBI did not solve the problem!')
                else:

                    niter[i] = res.total_iter
                    time[i] = res.cputime

                    if not qp.is_optimal(res.x, res.y):
                        print('Returned solution not optimal!')

                # Apply first control input to the plant
                u_sys[:, i] = res.x[-N*nu:-(N-1)*nu]

                # x_{k+1} = Ax_{t} + Bu_{t}
                x_sys[:, i+1] = problem.A.dot(x_sys[:, i]) + \
                    problem.B.dot(u_sys[:, i])

                # Update linear constraints
                self.update_initial_state(qp, x_sys[:, i+1])

        elif solver == 'mosek':
            for i in range(nsim):
                # Solve with mosek
                prob = mpbpy.QuadprogProblem(qp.P, qp.q, qp.A,
                                             qp.l, qp.u)
                res = prob.solve(solver=mpbpy.MOSEK, verbose=False)

                if res.status != 'optimal' and \
                        res.status != 'optimal inaccurate':
                    print('MOSEK did not solve the problem!')
                else:
                    niter[i] = res.total_iter
                    time[i] = res.cputime

                    if not qp.is_optimal(res.x, res.y):
                        print('Returned solution not optimal!')

                # Apply first control input to the plant
                u_sys[:, i] = res.x[-N*nu:-(N-1)*nu]

                # x_{k+1} = Ax_{t} + Bu_{t}
                x_sys[:, i+1] = problem.A.dot(x_sys[:, i]) + \
                    problem.B.dot(u_sys[:, i])

                # Update linear constraints
                self.update_initial_state(qp, x_sys[:, i+1])

        elif solver == 'ecos':
            cvxpy_prob, x_init, variables = self.gen_cvxpy_problem(qp)
            (x, u) = variables
            for i in range(nsim):

                x_init.value = x_sys[:, i]

                cvxpy_prob.solve(solver=cvxpy.ECOS, verbose=False)

                if cvxpy_prob.status != 'optimal' and \
                        cvxpy_prob.status != 'optimal inaccurate':
                    print('ECOS did not solve the problem!')
                else:
                    # Obtain time and number of iterations
                    time[i] = cvxpy_prob.solver_stats.setup_time + \
                        cvxpy_prob.solver_stats.solve_time

                    niter[i] = cvxpy_prob.solver_stats.num_iters

                # Apply first control input to the plant
                # u_ecos = np.asarray(u.value)
                u_sys[:, i] = u.value[:, 0].A1

                # x_{k+1} = Ax_{t} + Bu_{t}
                x_sys[:, i+1] = problem.A.dot(x_sys[:, i]) + \
                    problem.B.dot(u_sys[:, i])

                # Update linear constraints
                self.update_initial_state(qp, x_sys[:, i+1])

        else:
            raise ValueError('Solver not understood')

        # DEBUG: print state behavior
        # import matplotlib.pylab as plt
        # plt.figure()
        # plt.plot(x_sys.T)
        # plt.show(block=False)
        # import ipdb; ipdb.set_trace()

        # Return statistics
        return utils.Statistics(time), utils.Statistics(niter)
