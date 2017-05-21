import numpy as np
import scipy as sp
import scipy.sparse as spa
import osqp
import osqppurepy

P = spa.csc_matrix(np.array([[11., 0.], [0., 0.]]))
q = np.array([3., 4.])

A = spa.csc_matrix(np.array([[-1.0, 0.], [0., -1.], [-1., 3.],
                             [2., 5.], [3., 4]]))
l = -np.inf * np.ones(A.shape[0])
u = np.array([0., 0., -15., 100., 80.])

n = P.shape[0]
m = A.shape[0]


A = None
l = None
u = None

# OSQP options
osqp_opts = {'rho': 1.0,
             'auto_rho': False,
             'scaling_iter': 15,
             'scaling': True,
             'early_terminate_interval': 1,
             'max_iter': 500}

# Solve problem with OSQP
m = osqp.OSQP()
m.setup(P, q, A, l, u, **osqp_opts)
res = m.solve()

# Solve problem with OSQPPUREPY
m = osqppurepy.OSQP()
m.setup(P, q, A, l, u, **osqp_opts)
res = m.solve()
