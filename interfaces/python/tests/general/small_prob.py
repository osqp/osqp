import osqp
import osqppurepy as osqppurepy
import scipy.sparse as sparse
import scipy as sp
import numpy as np
import mathprogbasepy as mpbpy
sp.random.seed(2)

n = 2
m = 3
A = sparse.random(m, n, density=0.6, format='csc')
# A = sparse.eye(m)

# l = -sp.rand(m) * 2.
# u = sp.rand(m) * 2.

l = -sp.rand(m)
u = sp.rand(m)


# norm_l = np.linalg.norm(l)
# norm_u = np.linalg.norm(u)
Escal = 1

A /= Escal
l /= Escal
u /= Escal




P = sparse.random(n, n, density=0.6, format='csc')
P = P.dot(P.T).tocsc()
q = sp.randn(n)


# Divide cost function
norm_q = np.linalg.norm(q)
# norm_q = 1


# q = q/norm_q

# print("old P ")
# print(P.todense())
# P = P/norm_q
# print("new P ")
# print(P.todense())

osqp_opts = {'rho': 0.5,
             'sigma': 0.001,
             'auto_rho': False,
             'polish': False,
             'eps_abs': 1e-03,
             'eps_rel': 1e-03,
             'early_terminate_interval': 1,
             'max_iter': 174,
             'scaling': False}





# qp = mpbpy.QuadprogProblem(P, q, A, l, u)
# res_gurobi = qp.solve(solver=mpbpy.GUROBI, verbose=True)
# res_purepy = qp.solve(solver=OSQP_PUREPY, **osqp_opts)
# res_osqp = qp.solve(solver=mpbpy.OSQP, **osqp_opts)

model = osqp.OSQP()
model.setup(P=P, q=q, A=A, l=l, u=u, **osqp_opts)
res = model.solve()


model = osqppurepy.OSQP()
model.setup(P=P, q=q, A=A, l=l, u=u, **osqp_opts)
res = model.solve()



# x = res.x
# y = res.y
# pri = .5*x.dot(P.dot(x)) + q.dot(x)
# dual = -.5*x.dot(P.dot(x)) - u.dot(np.maximum(y, 0)) - l.dot(np.minimum(y, 0))
