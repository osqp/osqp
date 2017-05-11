import osqp
import osqppurepy as osqppurepy
import scipy.sparse as sparse
import scipy as sp
import numpy as np
import mathprogbasepy as mpbpy
sp.random.seed(3)

n = 20
m = 30
A = sparse.random(m, n, density=0.9, format='csc')
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




P = sparse.random(n, n, density=0.9)
# P = sparse.random(n, n, density=0.9).tocsc()
P = P.dot(P.T).tocsc()
q = sp.randn(n)

# Try to fix sparse matrix format
# P = P.tocoo().tocsc()

# Divide cost function
norm_q = np.linalg.norm(q)
# norm_q = 1


# q = q/norm_q

# print("old P ")
# print(P.todense())
# P = P/norm_q
# print("new P ")
# print(P.todense())

osqp_opts = {'rho': 0.1,
             'sigma': 0.001,
             'auto_rho': True,
             'polish': True,
             'eps_abs': 1e-03,
             'eps_rel': 1e-03,
             'early_terminate_interval': 1,
             'max_iter': 200,
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
res_purepy = model.solve()


print("Norm difference x OSQP and OSQPPUREPY %.4e" %
      np.linalg.norm(res.x - res_purepy.x))
print("Norm difference y OSQP and OSQPPUREPY %.4e" %
      np.linalg.norm(res.y - res_purepy.y))

# x = res.x
# y = res.y
# pri = .5*x.dot(P.dot(x)) + q.dot(x)
# dual = -.5*x.dot(P.dot(x)) - u.dot(np.maximum(y, 0)) - l.dot(np.minimum(y, 0))
