from __future__ import print_function
from __future__ import division
import numpy as np
import scipy.sparse as sp
from builtins import range
import sys


# For importing python modules from string
import importlib

# Import osqp
import osqp

# Import qpoases
import qpoases

# Plotting
import matplotlib.pylab as plt
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)   # fontsize of the tick labels
plt.rc('ytick', labelsize=15)   # fontsize of the tick labels
plt.rc('legend', fontsize=15)   # legend fontsize
plt.rc('text', usetex=True)     # use latex
plt.rc('font', family='serif')

colors = { 'b': '#1f77b4',
           'g': '#2ca02c',
           'o': '#ff7f0e'}

# Iterations
import tqdm


class QPmatrices(object):
    """
    QP problem matrices

    q_vecs is the matrix containing different linear costs
    """
    def __init__(self, P, q_vecs,
                 A, l, u, n, k):
        self.P = P
        self.q_vecs = q_vecs
        self.A = A
        self.l = l
        self.u = u
        self.n = n
        self.k = k


class Statistics(object):
    """
    Solve statistics
    """
    def __init__(self, x):
        self.x = x
        self.avg = np.mean(x)
        self.median = np.median(x)
        self.max = np.max(x)
        self.min = np.min(x)


def gen_qp_matrices(k, n, gammas):
    """
    Generate QP matrices for portfolio optimization problem
    """
    # Reset random seed for repetibility
    np.random.seed(1)

    # Problem parameters
    # k = 10
    # n = 200
    dens_lvl = 0.5
    version = 'sparse'  # 'dense' or 'sparse'

    # Generate data
    F = sp.random(n, k, density=dens_lvl, format='csc')
    D = sp.diags(np.random.rand(n) * np.sqrt(k), format='csc')
    mu = np.random.randn(n)
    # Write mu vector in the file
    # np.savetxt('portfolio_data.txt', mu)
    gamma = 1

    # Construct the problem
    if version == 'dense':
        #       minimize	x' (F * F' + D) x - (1/gamma) * mu' x
        #       subject to  1' x = 1
        #                   0 <= x <= 1
        P = 2 * (F.dot(F.T) + D)
        A = sp.vstack([np.ones((1, n)), sp.eye(n)]).tocsc()
        l = np.append([1.], np.zeros(n))
        u = np.append([1.], np.ones(n))

        # Create linear cost vectors
        q_vecs = np.empty((n, 0))
        for gamma in gammas:
            q_vecs = np.column_stack((q_vecs, -mu / gamma))


    elif version == 'sparse':
        #       minimize	x' D x + y' I y - (1/gamma) * mu' x
        #       subject to  1' x = 1
        #                   F' x = y
        #                   0 <= x <= 1
        P = sp.block_diag((2*D, 2*sp.eye(k)), format='csc')
        q = np.append(-mu / gamma, np.zeros(k))
        A = sp.vstack([
                sp.hstack([sp.csc_matrix(np.ones((1, n))),
                           sp.csc_matrix((1, k))]),
                sp.hstack([F.T, -sp.eye(k)]),
                sp.hstack([sp.eye(n), sp.csc_matrix((n, k))])
            ]).tocsc()
        l = np.hstack([1., np.zeros(k), np.zeros(n)])
        u = np.hstack([1., np.zeros(k), np.ones(n)])

        # Create linear cost vectors
        q_vecs = np.empty((k + n, 0))
        for gamma in gammas:
            q_vecs = np.column_stack((q_vecs,
                                      np.append(-mu / gamma, np.zeros(k))))

    # Return QP matrices
    return QPmatrices(P, q_vecs, A, l, u, n, k)


def solve_loop(qp_matrices, solver='emosqp'):
    """
    Solve portfolio optimization loop for all gammas
    """

    # Shorter name for qp_matrices
    qp = qp_matrices

    print('\nSolving portfolio problem loop for n = %d and solver %s' %
          (qp.n, solver))

    # Get number of problems to solve
    n_prob = qp.q_vecs.shape[1]

    # Initialize time vector
    time = np.zeros(n_prob)

    # Initialize number of iterations vector
    niter = np.zeros(n_prob)

    if solver == 'emosqp':
        # Pass the data to OSQP
        m = osqp.OSQP()
        m.setup(qp.P, qp.q_vecs[:, 0], qp.A, qp.l, qp.u, rho=0.1, verbose=False)

        # Get extension name
        module_name = 'emosqpn%s' % str(qp.n)

        # Generate the code
        m.codegen("code", python_ext_name=module_name, force_rewrite=True)

        # Import module
        emosqp = importlib.import_module(module_name)

        for i in range(n_prob):
            q = qp.q_vecs[:, i]

            # Update linear cost
            emosqp.update_lin_cost(q)

            # Solve
            x, y, status, niter[i], time[i] = emosqp.solve()

            # Check if status correct
            if status != 1:
                raise ValueError('OSQP did not solve the problem!')

            # DEBUG
            # solve with gurobi
            # import mathprogbasepy as mpbpy
            # prob = mpbpy.QuadprogProblem(qp.P, q, qp.A, qp.l, qp.u)
            # res = prob.solve(solver=mpbpy.GUROBI)

            # Check status
            if status != 1:
                raise ValueError('OSQP did not solve the problem')

    elif solver == 'qpoases':

        n_dim = qp.P.shape[0]
        m_dim = qp.A.shape[0]

        # Initialize cpu time
        qpoases_cpu_time = np.array([200.])

        # Get first vector q
        q = qp.q_vecs[:, 0]

        # Number of working set recalculations
        nWSR = np.array([1000])

        # Initialize qpOASES
        qpoases_m = qpoases.PyQProblem(n_dim, m_dim)
        options = qpoases.PyOptions()
        options.printLevel = qpoases.PyPrintLevel.NONE
        qpoases_m.setOptions(options)
        qpoases_m.init(qp.P.todense(), q, qp.A.todense(),
                       None, None, qp.l, qp.u,
                       nWSR, qpoases_cpu_time)




        for i in range(n_prob):
            q = qp.q_vecs[:, i]

            # Reset cpu time
            qpoases_cpu_time = np.array([200.])

            # Reset number of of working set recalculations
            nWSR = np.array([1000])

            # Solve new hot started problem
            qpoases_m.hotstart(q, None, None,
                               qp.l, qp.u, nWSR, qpoases_cpu_time)

            # Save time
            time[i] = qpoases_cpu_time[0]

            # Save number of iterations
            niter[i] = nWSR[0]

    else:
        raise ValueError('Solver not understood')


    # Return statistics
    return Statistics(time), Statistics(niter)

'''
Solve problems
'''
# Generate gamma parameters and cost vectors
n_gamma = 21
gammas = np.logspace(-2, 2, n_gamma)


# Assets
n_vec = np.array([100, 200, 300, 400])

# Factors
k_vec = (n_vec / 10).astype(int)


# Define statistics for osqp and qpoases
osqp_timing = []
osqp_iter = []
qpoases_timing = []
qpoases_iter = []


for i in range(len(n_vec)):
    # Generate QP matrices
    qp_matrices = gen_qp_matrices(k_vec[i], n_vec[i], gammas)

    # Solve loop with emosqp
    timing, niter = solve_loop(qp_matrices, 'emosqp')
    osqp_timing.append(timing)
    osqp_iter.append(niter)

    # Solving loop with qpOASES
    timing, niter = solve_loop(qp_matrices, 'qpoases')
    qpoases_timing.append(timing)
    qpoases_iter.append(niter)


# Plot timings
osqp_avg = np.array([x.avg for x in osqp_timing])
qpoases_avg = np.array([x.avg for x in qpoases_timing])

plt.figure()
ax = plt.gca()
plt.semilogy(n_vec, osqp_avg, color=colors['b'], label='OSQP')
plt.semilogy(n_vec, qpoases_avg, color=colors['o'], label='qpOASES')
plt.legend()
plt.grid()
ax.set_xlabel(r'Number of assets $n$')
ax.set_ylabel(r'Time [s]')
plt.tight_layout()
plt.show(block=False)
