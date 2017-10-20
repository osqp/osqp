"""
Big lasso problem
"""

import numpy as np
import scipy.sparse as spa
import osqp

# Problem dimensions and sparity
n = 100
m = 10*n
dens_lvl = 0.5

np.random.seed(1)
# Generate data
Ad = spa.random(m, n, density=dens_lvl)
x_true = np.multiply((np.random.rand(n) > 0.5).astype(float),
                     np.random.randn(n)) / np.sqrt(n)
bd = Ad.dot(x_true) + .5*np.random.randn(m)
gamma = 1.

#       minimize	y.T * y + gamma * np.ones(n).T * t
#       subject to  y = Ax
#                   -t <= x <= t
P = spa.block_diag((spa.csc_matrix((n, n)), 2*spa.eye(m),
                    spa.csc_matrix((n, n))), format='csc')
q = np.append(np.zeros(m + n), gamma*np.ones(n))
In = spa.eye(n)
Onm = spa.csc_matrix((n, m))
A = spa.vstack([spa.hstack([Ad, -spa.eye(m),
                            spa.csc_matrix((m, n))]),
                spa.hstack([In, Onm, -In]),
                spa.hstack([In, Onm, In])]).tocsc()
l = np.hstack([bd, -np.inf * np.ones(n), np.zeros(n)])
u = np.hstack([bd, np.zeros(n), np.inf * np.ones(n)])

osqp_opts = {'check_termination': 0,
             'polish': False,
             'scaling': True,
             'max_iter': 50000}

model = osqp.OSQP()
model.setup(P=P, q=q, A=A, l=l, u=u, **osqp_opts)
res_osqp = model.solve()
