import numpy as np
import scipy.sparse as spa
import osqp

np.random.seed(3)

n = 10
m = 20
P = spa.rand(n, n, density=.2, format='csc')
P = (P.T).dot(P)
q = np.random.randn(n)
A = spa.rand(m, n, density=.2, format='csc')
l = np.random.randn(m) - 5
u = np.random.randn(m) + 5

m = osqp.OSQP()
m.setup(P, q, A, l, u, rho=0.001)

# Test workspace return
m.codegen("code", 'Unix Makefiles', embedded=1, python_ext_name='test')

# res = m.solve()
#
# import mathprogbasepy as mpbpy
# prob = mpbpy.QuadprogProblem(P, q, A, l, u)
# res_gurobi = prob.solve(solver=mpbpy.GUROBI)
