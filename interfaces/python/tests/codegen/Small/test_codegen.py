import numpy as np
import scipy.sparse as sp
import osqp

np.random.seed(3)

n = 5
m = 10
P = sp.rand(n, n, .2)
P = (P.T).dot(P)
q = np.random.randn(n)
A = sp.rand(m, n, .5)
l = np.random.randn(m) - 5
u = np.random.randn(m) + 5

m = osqp.OSQP()
m.setup(P, q, A, l, u, rho=0.1)

# Test workspace return
m.codegen("code", 'Unix Makefiles', early_terminate=1, embedded_flag=1)

# res = m.solve()
