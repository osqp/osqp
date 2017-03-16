from __future__ import print_function
from __future__ import division
import numpy as np
import scipy.sparse as sp
from builtins import range

# Reload function for reloading emosqp module
try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3



# Import osqp
import osqp

# Plotting
import matplotlib.pylab as plt

# Iterations
import tqdm


class QPmatrices(object):
    """
    QP problem matrices

    q_vecs is the matrix containing different linear costs
    """
    def __init__(self, P, q_vecs,
                 A, l, u):
        self.P = P
        self.q_vecs = q_vecs
        self.A = A
        self.l = l
        self.u = u


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
    return QPmatrices(P, q_vecs, A, l, u)


def solve_loop(qp_matrices, solver='emosqp'):
    """
    Solve portfolio optimization loop for all gammas
    """
    # Shorter name for qp_matrices
    qp = qp_matrices

    # Get number of problems to solve
    n_prob = qp.q_vecs.shape[1]

    # Initialize time vector
    time = np.zeros(n_prob)

    # Initialize number of iterations vector
    niter = np.zeros(n_prob)

    if solver == 'emosqp':
        # Pass the data to OSQP
        m = osqp.OSQP()
        m.setup(qp.P, qp.q_vecs[:, 0], qp.A, qp.l, qp.u, rho=0.1)

        # Generate the code
        m.codegen("code")
        try:
            reload(emosqp)
        except NameError:
            import emosqp

        for i in range(n_prob):
            q = qp.q_vecs[:, i]

            # Update linear cost
            emosqp.update_lin_cost(q)

            # Solve
            x, y, status, niter[i], time[i] = emosqp.solve()

            print('Numer of iterations %i' % niter[i])

            # Check status
            if status != 1:
                raise ValueError('OSQP did not solve the problem')

        # Return statistics
        return Statistics(time), Statistics(niter)

'''
Solve problems
'''
# Generate gamma parameters and cost vectors
n_gamma = 21
gammas = np.logspace(-2, 2, n_gamma)


# Assets
n_vec = np.array([50])

# Factors
k_vec = (n_vec / 10).astype(int)

for i in range(len(n_vec)):
    # Generate QP matrices
    qp_matrices = gen_qp_matrices(k_vec[i], n_vec[i], gammas)

    # Solve loop
    stats_time, stats_niter = solve_loop(qp_matrices, 'emosqp')



# for i in range(n_gamma):
#     # Update linear cost
#     q_new = np.append(-mu / gammas[i], np.zeros(k))
#     emosqp.update_lin_cost(q_new)
#
#     # Solve
#     sol = emosqp.solve()
#     x = sol[0]
#     y = sol[1]
#     status_val = sol[2]
#     numofiter = sol[3]
#
#     # Compute details
#     if status_val == 1:
#         status_string = 'Solved'
#     else:
#         status_string = 'NOT Solved'
#     objval = .5 * np.dot(x, P.dot(x)) + np.dot(q_new, x)
#
#     # Print details
#     print("Risk aversion parameter:  %f" % gammas[i])
#     print("Status:                   %s" % status_string)
#     print("Number of iterations:     %d" % numofiter)
#     print("Objective value:          %f" % objval)
#     print(" ")
