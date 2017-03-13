from __future__ import print_function
import numpy as np
import scipy.sparse as sp
import osqp
from builtins import range

np.random.seed(1)

# Problem parameters
k = 10
n = 2000
dens_lvl = 0.5
version = 'sparse'  # 'dense' or 'sparse'

# Generate data
F = sp.random(n, k, density=dens_lvl, format='csc')
D = sp.diags(np.random.rand(n) * np.sqrt(k), format='csc')
mu = np.random.randn(n)
gamma = 1

# Construct the problem
if version == 'dense':
    #       minimize	x' (F * F' + D) x - (1/gamma) * mu' x
    #       subject to  1' x = 1
    #                   0 <= x <= 1
    P = 2 * (F.dot(F.T) + D)
    q = -mu / gamma
    A = sp.vstack([np.ones((1, n)), sp.eye(n)]).tocsc()
    l = np.append([1.], np.zeros(n))
    u = np.append([1.], np.ones(n))

elif version == 'sparse':
    #       minimize	x' D x + y' I y - (1/gamma) * mu' x
    #       subject to  1' x = 1
    #                   F' x = y
    #                   0 <= x <= 1
    P = sp.block_diag((2*D, 2*sp.eye(k)), format='csc')
    q = np.append(-mu / gamma, np.zeros(k))
    A = sp.vstack([
            sp.hstack([sp.csc_matrix(np.ones((1, n))), sp.csc_matrix((1, k))]),
            sp.hstack([F.T, -sp.eye(k)]),
            sp.hstack([sp.eye(n), sp.csc_matrix((n, k))])
        ]).tocsc()
    l = np.hstack([1., np.zeros(k), np.zeros(n)])
    u = np.hstack([1., np.zeros(k), np.ones(n)])


# Write mu vector in the file
np.savetxt('portfolio_data.txt', mu)

# Pass the data to OSQP
m = osqp.OSQP()
m.setup(P, q, A, l, u, rho=0.1)

# Generate the code
m.codegen("code", 'Unix Makefiles', early_terminate=1, embedded_flag=1)

import emosqp

# Generate gamma parameters
n_gamma = 21
gammas = np.logspace(-2, 2, n_gamma)
for i in range(n_gamma):
    # Update linear cost
    q_new = np.append(-mu / gammas[i], np.zeros(k))
    emosqp.update_lin_cost(q_new)

    # Solve
    sol = emosqp.solve()
    x = sol[0]
    y = sol[1]
    status_val = sol[2]
    numofiter = sol[3]

    # Compute details
    if status_val == 1:
        status_string = 'Solved'
    else:
        status_string = 'NOT Solved'
    objval = .5 * np.dot(x, P.dot(x)) + np.dot(q_new, x)

    # Print details
    print("Risk aversion parameter:  %f" % gammas[i])
    print("Status:                   %s" % status_string)
    print("Number of iterations:     %d" % numofiter)
    print("Objective value:          %f" % objval)
    print(" ")
