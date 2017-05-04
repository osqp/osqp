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
# import qpoases as qpoases  # Import qpoases
import mathprogbasepy as mpbpy  # Mathprogbasepy to benchmark gurobi

# Numerics
import numpy as np
import scipy.sparse as spa

# Pandas
from collections import OrderedDict, namedtuple

# Import examples utilities
from .. import utils


# RHS of linear equality constraint in sparse MPC variant
def b(x, nx, N):
    b = np.zeros((N+1)*nx)
    b[:nx] = -x
    return b


def gen_qp_matrices(problem):
    """
    Generate QP matrices for portfolio optimization problem
    """
    # Get dimensions
    (nx, nu) = problem.B.shape
    N = problem.N

    # Objective
    Px = spa.kron(spa.eye(N), problem.Q)
    Pu = spa.kron(spa.eye(N), problem.R)
    P = spa.block_diag([Px, problem.QN, Pu]).tocsc()
    q = np.zeros((N+1)*nx + N*nu)

    # Dynamics
    Ax = spa.kron(spa.eye(N+1), -spa.eye(nx)) + \
         spa.kron(spa.eye(N+1, k=-1), problem.A)
    Au = spa.kron(spa.vstack([spa.csc_matrix((1, N)), spa.eye(N)]), problem.B)
    A = spa.hstack([Ax, Au])
    l = b(problem.x0, nx, N)
    u = b(problem.x0, nx, N)

    # Bounds on u
    lx = np.tile(problem.umin, N)
    ux = np.tile(problem.umax, N)

    # Bounds on x
    if len(problem.xmin) > 0:
        lx = np.append(lx, np.tile(problem.xmin, N+1))
        ux = np.append(ux, np.tile(problem.xmax, N+1))

    # Return QP matrices
    qp_matrices = utils.QPmatrices(P, q, A, l, u, lx, ux)
    return qp_matrices


def solve_loop(qp_matrices, problem, nsim, solver='osqp'):
    """
    Solve MPC loop
    """
    # Shorter name for qp_matrices
    qp = qp_matrices

    # Get dimensions
    (nx, nu) = problem.B.shape
    N = problem.N

    print('\nSolving MPC %s problem for N = %d (horizon) and solver %s' %
          (problem.name, N, solver))

    # Initialize time and number of iterations vectors
    time = np.zeros(nsim)
    niter = np.zeros(nsim)

    if solver == 'osqp':
        # Construct qp matrices
        if len(problem.xmin) > 0:
            # If the problem has state constraints
            Aosqp = spa.vstack([
                        qp.A,
                        spa.eye((N+1)*nx + N*nu),
                      ]).tocsc()
        else:
            Aosqp = spa.vstack([
                        qp.A,
                        spa.hstack([
                            spa.csc_matrix((N*nu, (N+1)*nx)),
                            spa.eye(N*nu)]),
                      ]).tocsc()
        losqp = np.hstack([qp.l, qp.lx])
        uosqp = np.hstack([qp.u, qp.ux])

        # Initial state
        x0 = problem.x0

        # Setup OSQP
        m = osqp.OSQP()
        m.setup(qp.P, qp.q, Aosqp, losqp, uosqp,
                auto_rho=False,
                rho=0.1,
                max_iter=5000,
                polish=False,
                verbose=False)

        for i in range(nsim):
            # Solve with osqp
            res = m.solve()

            # Save time and number of iterations
            time[i] = res.info.run_time
            niter[i] = res.info.iter

            # Check if status is correct
            status = res.info.status_val
            if status != m.constant('OSQP_SOLVED'):
                import ipdb; ipdb.set_trace()
                raise ValueError('OSQP did not solve the problem!')

            # Apply first control input to the plant
            u = res.x[-N*nu:-(N-1)*nu]
            x0 = problem.A.dot(x0) + problem.B.dot(u)

            # Update QP problem
            losqp = np.hstack([b(x0, nx, N), qp.lx])
            uosqp = np.hstack([b(x0, nx, N), qp.ux])
            m.update(l=losqp, u=uosqp)

    elif solver == 'osqp_coldstart':
        # Construct qp matrices
        if len(problem.xmin) > 0:
            # If the problem has state constraints
            Aosqp = spa.vstack([
                        qp.A,
                        spa.eye((N+1)*nx + N*nu),
                      ]).tocsc()
        else:
            Aosqp = spa.vstack([
                        qp.A,
                        spa.hstack([
                            spa.csc_matrix((N*nu, (N+1)*nx)),
                            spa.eye(N*nu)]),
                      ]).tocsc()
        losqp = np.hstack([qp.l, qp.lx])
        uosqp = np.hstack([qp.u, qp.ux])

        # Initial state
        x0 = problem.x0

        # Setup OSQP
        m = osqp.OSQP()
        m.setup(qp.P, qp.q, Aosqp, losqp, uosqp,
                warm_start=False,
                auto_rho=False,
                rho=0.1,
                max_iter=5000,
                polish=False,
                verbose=False)

        for i in range(nsim):
            # Solve with osqp
            res = m.solve()

            # Save time and number of iterations
            time[i] = res.info.run_time
            niter[i] = res.info.iter

            # Check if status is correct
            status = res.info.status_val
            if status != m.constant('OSQP_SOLVED'):
                import ipdb; ipdb.set_trace()
                raise ValueError('OSQP did not solve the problem!')

            # Apply first control input to the plant
            u = res.x[-N*nu:-(N-1)*nu]
            x0 = problem.A.dot(x0) + problem.B.dot(u)

            # Update QP problem
            losqp = np.hstack([b(x0, nx, N), qp.lx])
            uosqp = np.hstack([b(x0, nx, N), qp.ux])
            m.update(l=losqp, u=uosqp)

    elif solver == 'qpoases':

        n_dim = qp.P.shape[0]  # Number of variables
        m_dim = qp.A.shape[0]  # Number of constraints without bounds

        # Initialize qpoases and set options
        qpoases_m = qpoases.PyQProblem(n_dim, m_dim)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qpoases_m.setOptions(options)

        # Construct bounds for qpoases
        lx = np.append(-np.inf * np.ones((N+1)*nx), qp.lx)
        ux = np.append(np.inf * np.ones((N+1)*nx), qp.ux)

        # Setup matrix P and A
        P = np.ascontiguousarray(qp.P.todense())
        A = np.ascontiguousarray(qp.A.todense())

        # Initial state
        x0 = problem.x0

        # RHS of the linear equality constraints
        lqpoases = qp.l
        uqpoases = qp.u

        for i in range(nsim):
            # Reset cpu time
            qpoases_cpu_time = np.array([20.])

            # Reset number of of working set recalculations
            nWSR = np.array([1000])

            if i == 0:
                # First iteration
                res_qpoases = qpoases_m.init(P, np.ascontiguousarray(qp.q), A,
                                             np.ascontiguousarray(lx),
                                             np.ascontiguousarray(ux),
                                             np.ascontiguousarray(lqpoases),
                                             np.ascontiguousarray(uqpoases),
                                             nWSR, qpoases_cpu_time)
            else:
                # Solve new hot started problem
                res_qpoases = qpoases_m.hotstart(np.ascontiguousarray(qp.q),
                                                 np.ascontiguousarray(lx),
                                                 np.ascontiguousarray(ux),
                                                 np.ascontiguousarray(lqpoases),
                                                 np.ascontiguousarray(uqpoases),
                                                 nWSR,
                                                 qpoases_cpu_time)

            if res_qpoases != 0:
                raise ValueError('qpoases did not solve the problem!')

            # Save time and number of iterations
            time[i] = qpoases_cpu_time[0]
            niter[i] = nWSR[0]

            # Get qpoases solution
            sol_qpoases = np.zeros((N+1)*nx + N*nu)
            qpoases_m.getPrimalSolution(sol_qpoases)

            # Apply first control input to the plant
            u = sol_qpoases[-N*nu:-(N-1)*nu]
            x0 = problem.A.dot(x0) + problem.B.dot(u)

            # Update linear equality constraint
            lqpoases = b(x0, nx, N)
            uqpoases = b(x0, nx, N)

    elif solver == 'gurobi':
        # Construct qp matrices
        if len(problem.xmin) > 0:
            # If the problem has state constraints
            Agurobi = spa.vstack([
                        qp.A,
                        spa.eye((N+1)*nx + N*nu),
                      ]).tocsc()
        else:
            Agurobi = spa.vstack([
                        qp.A,
                        spa.hstack([
                            spa.csc_matrix((N*nu, (N+1)*nx)),
                            spa.eye(N*nu)]),
                      ]).tocsc()
        lgurobi = np.hstack([qp.l, qp.lx])
        ugurobi = np.hstack([qp.u, qp.ux])

        # Initial state
        x0 = problem.x0

        for i in range(nsim):
            # Solve with gurobi
            prob = mpbpy.QuadprogProblem(qp.P, qp.q, Agurobi, lgurobi, ugurobi)
            res = prob.solve(solver=mpbpy.GUROBI, verbose=False)

            # Save time and number of iterations
            time[i] = res.cputime
            niter[i] = res.total_iter

            # Check if status is correct
            status = res.status
            if status != 'optimal':
                import ipdb; ipdb.set_trace()
                raise ValueError('Gurobi did not solve the problem!')

            # Apply first control input to the plant
            u = res.x[-N*nu:-(N-1)*nu]
            x0 = problem.A.dot(x0) + problem.B.dot(u)

            # Update QP problem
            lgurobi = np.hstack([b(x0, nx, N), qp.lx])
            ugurobi = np.hstack([b(x0, nx, N), qp.ux])

    else:
        raise ValueError('Solver not understood')

    # Return statistics
    return utils.Statistics(time), utils.Statistics(niter)


def run_mpc_pendulum_example():
    '''
    Solve MPC pendulum problem
    https://github.com/ferreau/mpcBenchmarking/blob/master/benchmarks/Benchmark_pendulum.m
    '''

    # Problem setup (sampling interval Ts = 0.05s)
    problem = namedtuple("problem", "A B R Q QN umin umax xmin xmax N x0 name")
    problem.A = spa.csc_matrix([[1.001, -0.05, -0.001],
                                [-0.05, 1.003, 0.05],
                                [-0.001, 0.05, 1.001]])
    problem.B = spa.csc_matrix([[0.], [0.001], [0.05]])
    problem.R = spa.diags([0.1])
    problem.Q = spa.eye(3)
    problem.QN = spa.csc_matrix([[113.4637, -95.4971, -21.0556],
                                 [-95.4971, 99.5146, 23.3756],
                                 [-21.0556, 23.3756, 12.1868]])
    problem.umin = np.array([-1.25])
    problem.umax = np.array([1.25])
    problem.xmin = []
    problem.xmax = []
    problem.x0 = 0.6*np.ones(3)
    problem.name = 'pendulum'

    # Simulation steps
    nsim = 100

    # Prediction horizon
    N_vec = np.array([10, 20])

    # Define statistics for osqp, qpoases and gurobi
    osqp_timing = []
    osqp_iter = []
    osqp_coldstart_timing = []
    osqp_coldstart_iter = []
    # qpoases_timing = []
    # qpoases_iter = []
    gurobi_iter = []
    gurobi_timing = []

    for i in range(len(N_vec)):
        # Generate QP
        problem.N = N_vec[i]
        qp_matrices = gen_qp_matrices(problem)

        # Solve loop with osqp
        timing, niter = solve_loop(qp_matrices, problem, nsim, 'osqp')
        osqp_timing.append(timing)
        osqp_iter.append(niter)

        # Solve loop with osqp (coldstart)
        timing, niter = solve_loop(qp_matrices, problem, nsim, 'osqp_coldstart')
        osqp_coldstart_timing.append(timing)
        osqp_coldstart_iter.append(niter)

        # # Solving loop with qpoases
        # timing, niter = solve_loop(qp_matrices, 'qpoases')
        # qpoases_timing.append(timing)
        # qpoases_iter.append(niter)

        # Solve loop with gurobi
        timing, niter = solve_loop(qp_matrices, problem, nsim, 'gurobi')
        gurobi_timing.append(timing)
        gurobi_iter.append(niter)

    solver_timings = OrderedDict([('OSQP (warm start)', osqp_timing),
                                  ('OSQP (cold start)', osqp_coldstart_timing),
                                  # ('qpOASES', qpoases_timing),
                                  ('GUROBI', gurobi_timing)])

    utils.generate_plot('mpc', 'time', 'median', N_vec, solver_timings,
                        fig_size=0.9, plot_name='pendulum')
    utils.generate_plot('mpc', 'time', 'total', N_vec, solver_timings,
                        fig_size=0.9, plot_name='pendulum')
    utils.generate_plot('mpc', 'time', 'max', N_vec, solver_timings,
                        fig_size=0.9, plot_name='pendulum')


def run_mpc_helicopter_example():
    '''
    Solve MPC helicopter problem
    https://github.com/ferreau/mpcBenchmarking/blob/master/benchmarks/Benchmark_helicopter.m
    '''

    # Problem setup
    problem = namedtuple("problem", "A B R Q QN umin umax xmin xmax N x0 name")
    problem.A = spa.csc_matrix([[0.99, 0., 0.01, 0., 0., 0.],
                                [0., 0.99, 0., 0.01, 0., 0.],
                                [0., 0., 0.99, 0., 0., 0.],
                                [0., 0., 0., 0.99, 0., 0.],
                                [0.01, 0., 0., 0., 0.99, 0.],
                                [0., 0.01, 0., 0., 0., 0.99]])
    problem.B = spa.csc_matrix([[0., 0.],
                                [0.0001, -0.0001],
                                [0.0019, -0.0019],
                                [0.0132, -0.0132],
                                [0., 0.],
                                [0., 0.]])
    problem.R = 0.001*spa.eye(2)
    problem.Q = spa.diags([100, 100, 10, 10, 400, 200])

    # TBC
