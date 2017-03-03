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
objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q * x )
constraints = [A * x <= u, l <= A * x]
prob = cvxpy.Problem(objective, constraints)
prob.solve(abstol=1e-10, reltol=1e-10)
x_test = np.asarray(x.value).flatten()
y_test = (constraints[0].dual_value - constraints[1].dual_value).A1
obj_value_test = objective.value
status_test = prob.status

# New data
q_new = np.array([2.5, 3.2])
l_new = np.array([0.8, -3.4, -11.])
u_new = np.array([1.6, 1.0, 15.])

# Generate problem solutions
sols_data = {'x_test': x_test,
             'y_test': y_test,
             'obj_value_test': obj_value_test,
             'status_test': status_test,
             'q_new': q_new,
             'l_new': l_new,
             'u_new': u_new}


# Generate problem data
cu.generate_problem_data(P, q, A, l, u, 'basic_qp', sols_data)
