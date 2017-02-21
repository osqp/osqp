import numpy as np
import scipy.sparse as spa
import scipy as sp
import utils.codegen_utils as cu
import cvxpy
import osqp


n = 50
m = 150
# Generate random Matrices
Pt = spa.random(n, n)
P = Pt.T.dot(Pt).tocsc() + spa.eye(n)
q = sp.randn(n)
A = spa.random(m, n).tolil()  # Lil for efficiency
u = 3 + sp.randn(m)
l = -3 + sp.randn(m)

# Make random problem infeasible
A[int(n/2), :] = A[int(n/2)+1, :]
l[int(n/2)] = u[int(n/2)+1] + 10 * sp.rand()
u[int(n/2)] = l[int(n/2)] + 0.5

# Convert A to csc
A = A.tocsc()



# Solve with CVXPY and get solution
x = cvxpy.Variable(n)
objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q * x )
constraints = [A * x <= u, l <= A * x]
prob = cvxpy.Problem(objective, constraints)
prob.solve()
status_test = prob.status

# import ipdb; ipdb.set_trace()

# Generate problem solutions
sols_data = {'status_test': status_test}


# Generate problem data
cu.generate_problem_data(P, q, A, l, u, 'infeasibility', sols_data)



# Write new vectors
# f = open("basic_qp/basic_qp.h", "a")
#
# cu.write_vec_float(f, x.value, "sol_x")
#
# f.close()
