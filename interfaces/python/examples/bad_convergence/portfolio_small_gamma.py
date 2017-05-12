"""
Portfolio example with small gamma
"""
import numpy as np
import scipy.sparse as spa

# import osqppurepy as osqp
import osqp


# Reset RNG
np.random.seed(1)

# Dimension
n = 10
k = 1

# Density level
dens_lvl = 0.5

# Gamma
gamma = 1e-02

# Generate data
F = spa.random(n, k, density=dens_lvl, format='csc')
D = spa.diags(np.random.rand(n) * np.sqrt(k), format='csc')
mu = np.random.randn(n)

# Construct the problem
#       minimize	x' D x + y' I y - (1/gamma) * mu' x
#       subject to  1' x = 1
#                   F' x = y
#                   0 <= x <= 1
P = spa.block_diag((2*D, 2*spa.eye(k)), format='csc')
q = np.append(-mu / gamma, np.zeros(k))
A = spa.vstack([
        spa.hstack([spa.csc_matrix(np.ones((1, n))),
                   spa.csc_matrix((1, k))]),
        spa.hstack([F.T, -spa.eye(k)]),
        spa.hstack([spa.eye(n), spa.csc_matrix((n, k))])]).tocsc()
l = np.append(np.hstack([1., np.zeros(k)]), np.zeros(n))
u = np.append(np.hstack([1., np.zeros(k)]),  np.ones(n))


# Try to normalize
# P = P/np.linalg.norm(q)
# q = q/np.linalg.norm(q)

# Setup OSQP
# import osqppurepy as osqp
m = osqp.OSQP()
m.setup(P, q, A, l, u,
        auto_rho=True,
        scaling=True,
        scaling_iter=100,
        rho=0.1,
        sigma=0.001,
        polish=False,
        verbose=True)

res = m.solve()
