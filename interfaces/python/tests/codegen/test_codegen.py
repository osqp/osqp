import osqp
import numpy as np
import scipy.sparse as spa


n = 5
m = 10
P = spa.random(n, n, density=0.5)
P = P.dot(P.T).tocsc()
q = np.random.randn(n)
A = spa.random(m, n, density=0.5)
u = np.random.rand(m)
l = -np.random.rand(m)


m = osqp.OSQP()
m.setup(P, q, A, l, u, time_limit=2)

m.codegen('code')
