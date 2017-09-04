import osqp
import osqppurepy
import scipy.sparse as sparse
import scipy as sp
import numpy as np
import mathprogbasepy as mpbpy
sp.random.seed(2)

n = 50
m = 200
A = sparse.random(m, n, density=0.9,
                  data_rvs=np.random.randn,
                  format='csc')
l = -np.random.rand(m) * 2.
u = np.random.rand(m) * 2.
# l[4:5] = u[4:5]

P = sparse.random(n, n, density=0.9,
                  data_rvs=np.random.randn,
                  format='csc')
P = P.dot(P.T)
q = sp.randn(n)

# Test
# rho = 5.0
# P = P
# q = q

# Test
rho = 0.1
# rho=10.0
# q /= 100
# P *= 100
q *= 1000


osqp_opts = {'rho': rho,
             #  'auto_rho': True,
             'sigma': 1e-06,
             #  'eps_rel': 1e-08,
             #  'eps_abs': 1e-08,
             'scaled_termination': False,
             'early_terminate_interval': 1,
             'polish': False,
             'scaling': True,
             'scaling_norm': -1,
             'max_iter': 2500,
             'verbose': True
             }

qp = mpbpy.QuadprogProblem(P, q, A, l, u)
res_gurobi = qp.solve(solver=mpbpy.GUROBI)
# res_purepy = qp.solve(solver=mpbpy.OSQP_PUREPY, **osqp_opts)
# res_osqp = qp.solve(solver=mpbpy.OSQP, **osqp_opts)

model = osqppurepy.OSQP()
model.setup(P=P, q=q, A=A, l=l, u=u, **osqp_opts)
res_osqppurepy = model.solve()


# Check difference with gurobi
print("Difference Purepy vs Gurobi")
print("  - primal = %.4f" %
      (np.linalg.norm(res_gurobi.x - res_osqppurepy.x) /
       np.linalg.norm(res_gurobi.x)))
print("  - dual = %.4f" %
      (np.linalg.norm(res_gurobi.y - res_osqppurepy.y) /
       np.linalg.norm(res_gurobi.y)))


# Solve with SCS
# import cvxpy
# x = cvxpy.Variable(n)
# objective = cvxpy.Minimize(cvxpy.quad_form(x, P) + q * x)
# constraints = [l <= A * x, A * x <= u]
# problem = cvxpy.Problem(objective, constraints)
# problem.solve(solver=cvxpy.SCS, verbose=True)

model = osqp.OSQP()
model.setup(P=P, q=q, A=A, l=l, u=u, **osqp_opts)
res_osqp = model.solve()


# # Store optimal values
# x_opt = res_osqp.x
# y_opt = res_osqp.y
#
# # Warm start with zeros
# model.warm_start(x=np.zeros(n), y=np.zeros(m))
# res_osqp_zero_warmstart = model.solve()
#
# # Warm start with optimal values
# model.warm_start(x=x_opt, y=y_opt)
# res_osqp_opt_warmstart = model.solve()
