import numpy as np
import scipy.sparse as sp
import osqp

np.random.seed(1)

# Problem parameters
k = 2
n = 20
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
w = m.codegen("code", 'Unix Makefiles', early_terminate=1, embedded_flag=1)

# Solve the problem
# res = m.solve()
