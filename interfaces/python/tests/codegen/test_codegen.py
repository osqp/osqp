import numpy as np
import scipy.sparse as sp
import osqp

target_dir = "build"

n = 10
m = 5
P = sp.rand(n, n, .2)
P = (P.T).dot(P)
q = np.random.randn(n)
A = sp.rand(m, n, .5)
l = np.random.randn(m) - 5
u = np.random.randn(m) + 5

m = osqp.OSQP()
m.setup(P, q, A, l, u)

# Test workspace return
work = m.codegen("code", 'Unix Makefiles', embedded_flag=1)
