import osqp
import osqppurepy
import scipy.sparse as sparse
import scipy as sp
import numpy as np
import mathprogbasepy as mpbpy
sp.random.seed(2)

n = 100
m = 1000
A = sparse.random(m, n, density=0.5,
                  data_rvs=np.random.randn,
                  format='csc')
l = -1. - np.random.rand(m)
u = 1 + np.random.rand(m)


# A = sparse.eye(n).tocsc()
# l = -1 * np.ones(n)
# u = 1 * np.ones(n)

# l += 10
# u += 10

# l *= 1000
# u *= 1000
# A *= 1000

# Make problem infeasible
# A_temp = A[5, :]
# A[6, :] = A_temp
# l[6] = l[5] + 2.
# u[6] = l[6] + 3.

P = sparse.random(n, n, density=0.5,
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
# q *= 2000


osqp_opts = {'rho': rho,
             #  'auto_rho': True,
             'sigma': 1e-06,
            #  'eps_rel': 1e-05,
            #  'eps_abs': 1e-05,
             'scaled_termination': False,
             'early_terminate_interval': 25,
             'polish': True,
             'scaling': True,
             'scaling_norm': -1,
             'max_iter': 2500,
             'verbose': True,
             'linsys_solver': 0
             }

qp = mpbpy.QuadprogProblem(P, q, A, l, u)
res_gurobi = qp.solve(solver=mpbpy.GUROBI, verbose=False)
# res_purepy = qp.solve(solver=mpbpy.OSQP_PUREPY, **osqp_opts)
# res_osqp = qp.solve(solver=mpbpy.OSQP, **osqp_opts)
#
# model = osqppurepy.OSQP()
# model.setup(P=P, q=q, A=A, l=l, u=u, **osqp_opts)
# res_osqppurepy = model.solve()

# Solve with SuiteSparse LDL
model = osqp.OSQP()
model.setup(P=P, q=q, A=A, l=l, u=u, **osqp_opts)
res_osqp = model.solve()

# Solve with Pardiso
model2 = osqp.OSQP()
osqp_opts['linsys_solver'] = 1
model2.setup(P=P, q=q, A=A, l=l, u=u, **osqp_opts)
res_osqp2 = model2.solve()

print("Difference SuiteSparse LDL vs Pardiso")
print("SuiteSparse LDL runtime = %.4f" % res_osqp.info.run_time)
print("Pardiso runtime         = %.4f" % res_osqp2.info.run_time)

# Check difference with gurobi
if res_gurobi.status == 'optimal':
    print("Difference OSQP vs Gurobi")
    print("  - primal = %.4f" %
          (np.linalg.norm(res_gurobi.x - res_osqp.x) /
           np.linalg.norm(res_gurobi.x)))
    print("  - dual = %.4f" %
          (np.linalg.norm(res_gurobi.y - res_osqp.y) /
           np.linalg.norm(res_gurobi.y)))


# Solve with SCS
# import cvxpy
# x = cvxpy.Variable(n)
# objective = cvxpy.Minimize(cvxpy.quad_form(x, P) + q * x)
# constraints = [l <= A * x, A * x <= u]
# problem = cvxpy.Problem(objective, constraints)
# problem.solve(solver=cvxpy.SCS, verbose=True)


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
