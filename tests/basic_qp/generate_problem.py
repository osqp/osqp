import numpy as np
import scipy.sparse as spa
import utils.codegen_utils as cu
import cvxpy

P = spa.csc_matrix(np.array([[4., 1.], [1., 2.]]))
q = np.ones(2)

A = spa.csc_matrix(np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
l = np.array([1.0, 0.0, 0.0 ])
u = np.array([1.0, 0.7, 0.7])

n = P.shape[0]
m = A.shape[0]


# Solve with CVXPY and get solution
x = cvxpy.Variable(n)
objective = cvxpy.Minimize(cvxpy.quad_form(x, P) + q * x )
constraints = [A.dot(x) <= u, l <= A.dot(x)]
prob = cvxpy.Problem(objective, constraints)
prob.solve()


# Generate problem data
cu.generate_code(P, q, A, l, u, "basic_qp")


# Write new vectors
# f = open("basic_qp/basic_qp.h", "a")
#
# cu.write_vec_float(f, x.value, "sol_x")
#
# f.close()
