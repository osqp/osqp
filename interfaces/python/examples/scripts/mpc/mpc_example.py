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
from collections import OrderedDict

# Import examples utilities
from .. import utils

# Load MPC examples data
from .examples.ball import load_ball_data
from .examples.helicopter import load_helicopter_data
from .examples.pendulum import load_pendulum_data


# RHS of linear equality constraint in sparse MPC variant
def b(x, nx, N):
    b = np.zeros((N+1)*nx)
    b[:nx] = -x
    return b


def gen_qp_matrices(problem):
    """
    Generate QP matrices for MPC problem
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

    # Terminal constraints
    if len(problem.tmin) > 0:
        nt = problem.T.shape[0]
        A = spa.vstack([A, spa.hstack([spa.csc_matrix((nt, N*nx)),
                                       problem.T,
                                       spa.csc_matrix((nt, N*nu))])])
        l = np.append(l, problem.tmin)
        u = np.append(u, problem.tmax)

    # Bounds on u
    lx = np.tile(problem.umin, N)
    ux = np.tile(problem.umax, N)

    # Bounds on x
    if len(problem.xmin) > 0:
        lx = np.append(np.tile(problem.xmin, N+1), lx)
        ux = np.append(np.tile(problem.xmax, N+1), ux)

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

    print('\nSolving MPC %s example for N = %d (horizon) and solver %s' %
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
                # auto_rho=True,
                auto_rho=False,
                rho=1e-2,
                max_iter=2500,
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

            # Update linear constraints
            if len(problem.tmin) > 0:
                losqp = np.hstack([b(x0, nx, N), problem.tmin, qp.lx])
                uosqp = np.hstack([b(x0, nx, N), problem.tmax, qp.ux])
            else:
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
                #auto_rho=True,
                auto_rho=False,
                rho=0.1,
                max_iter=2500,
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

            # Update linear constraints
            if len(problem.tmin) > 0:
                losqp = np.hstack([b(x0, nx, N), problem.tmin, qp.lx])
                uosqp = np.hstack([b(x0, nx, N), problem.tmax, qp.ux])
            else:
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

            # Update linear constraints
            if len(problem.tmin) > 0:
                lqpoases = np.hstack([b(x0, nx, N), problem.tmin])
                uqpoases = np.hstack([b(x0, nx, N), problem.tmax])
            else:
                lqpoases = np.hstack([b(x0, nx, N)])
                uqpoases = np.hstack([b(x0, nx, N)])

    elif solver == 'gurobi' or solver == 'mosek':
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
            if solver == 'gurobi':
                res = prob.solve(solver=mpbpy.CPLEX, verbose=False)
            else:
                res = prob.solve(solver=mpbpy.MOSEK, verbose=True)

            # Save time and number of iterations
            time[i] = res.cputime
            niter[i] = res.total_iter

            # Check if status is correct
            status = res.status
            if status != 'optimal':
                import ipdb; ipdb.set_trace()
                raise ValueError('Gurobi did not solve the problem!')

            # Plot the computed control action
            if i == 0:
                x_sim = np.zeros((N+1, nx))
                u_sim = np.zeros((N, nu))
                x_sim[0] = x0
                u = res.x[-N*nu:]

                for j in range(N):
                    u_sim[j] = u[j*nu:(j+1)*nu]
                    x_sim[j+1] = problem.A.dot(x_sim[j]) + problem.B.dot(u_sim[j])

                import matplotlib.pylab as plt
                plt.figure(1)
                plt.step(np.arange(N), u_sim.T[0], label='input1')
                plt.step(np.arange(N), u_sim.T[1], label='input2')
                plt.show(block=False)

                plt.figure(2)
                plt.plot(np.arange(N+1), x_sim.T[0], label='x1')
                plt.plot(np.arange(N+1), x_sim.T[1], label='x2')
                plt.plot(np.arange(N+1), x_sim.T[2], label='x3')
                plt.plot(np.arange(N+1), x_sim.T[3], label='x4')
                plt.plot(np.arange(N+1), x_sim.T[4], label='x5')
                plt.plot(np.arange(N+1), x_sim.T[5], label='x6')
                plt.show(block=False)

            # Apply first control input to the plant
            u = res.x[-N*nu:-(N-1)*nu]
            x0 = problem.A.dot(x0) + problem.B.dot(u)

            # Update QP problem
            if len(problem.tmin) > 0:
                lgurobi = np.hstack([b(x0, nx, N), problem.tmin, qp.lx])
                ugurobi = np.hstack([b(x0, nx, N), problem.tmax, qp.ux])
            else:
                lgurobi = np.hstack([b(x0, nx, N), qp.lx])
                ugurobi = np.hstack([b(x0, nx, N), qp.ux])

    else:
        raise ValueError('Solver not understood')

    # Return statistics
    return utils.Statistics(time), utils.Statistics(niter)


def run_mpc_example(example_name):
    """
    Solve MPC example
    """

    # Load the example
    if example_name == 'pendulum':
        print("MPC pendulum example")
        print("--------------------")
        problem = load_pendulum_data()
    elif example_name == 'helicopter':
        print("MPC helicopter example")
        print("----------------------")
        problem = load_helicopter_data()
    elif example_name == 'ball':
        print("MPC ball example")
        print("----------------")
        problem = load_ball_data()
    else:
        problem = load_ball_data()  # Default data

    # Simulation steps
    nsim = 1

    # Prediction horizon
    N_vec = np.array([20])

    # Define statistics for osqp, qpoases and gurobi
    osqp_timing = []
    osqp_iter = []
    osqp_coldstart_timing = []
    osqp_coldstart_iter = []
    # qpoases_timing = []
    # qpoases_iter = []
    gurobi_iter = []
    gurobi_timing = []
    mosek_iter = []
    mosek_timing = []

    for i in range(len(N_vec)):
        # Generate QP
        problem.N = N_vec[i]
        qp_matrices = gen_qp_matrices(problem)

        # # Solve loop with osqp
        # timing, niter = solve_loop(qp_matrices, problem, nsim, 'osqp')
        # osqp_timing.append(timing)
        # osqp_iter.append(niter)

        # # Solve loop with osqp (coldstart)
        # timing, niter = solve_loop(qp_matrices, problem, nsim, 'osqp_coldstart')
        # osqp_coldstart_timing.append(timing)
        # osqp_coldstart_iter.append(niter)

        # # Solving loop with qpoases
        # timing, niter = solve_loop(qp_matrices, 'qpoases')
        # qpoases_timing.append(timing)
        # qpoases_iter.append(niter)

        # Solve loop with gurobi
        timing, niter = solve_loop(qp_matrices, problem, nsim, 'gurobi')
        gurobi_timing.append(timing)
        gurobi_iter.append(niter)

        # # Solve loop with mosek
        # timing, niter = solve_loop(qp_matrices, problem, nsim, 'mosek')
        # mosek_timing.append(timing)
        # mosek_iter.append(niter)

    solver_timings = OrderedDict([
                                  # ('OSQP (warm start)', osqp_timing),
                                  # ('OSQP (cold start)', osqp_coldstart_timing),
                                  # ('qpOASES', qpoases_timing),
                                  ('GUROBI', gurobi_timing),
                                  # ('MOSEK', mosek_timing)
                                  ])
    #
    # utils.generate_plot('mpc', 'time', 'median', N_vec, solver_timings,
    #                     fig_size=0.9, plot_name=problem.name)
    # utils.generate_plot('mpc', 'time', 'total', N_vec, solver_timings,
    #                     fig_size=0.9, plot_name=problem.name)
    # utils.generate_plot('mpc', 'time', 'max', N_vec, solver_timings,
    #                     fig_size=0.9, plot_name=problem.name)
