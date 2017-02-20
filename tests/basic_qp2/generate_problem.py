import numpy as np
import scipy.sparse as spa
import utils.codegen_utils as cu
import cvxpy
import osqp

P = spa.csc_matrix(np.array([[11., 0.], [0., 0.]]))
q = np.array([3., 4.])


A = spa.csc_matrix(np.array([[-1.0, 0.], [0., -1.], [-1., 3.],
                             [2., 5.], [3., 4]]))
l = -np.inf * np.ones(A.shape[0])
u = np.array([0., 0., -15., 100., 80.])

n = P.shape[0]
m = A.shape[0]


# Solve with CVXPY and get solution
x = cvxpy.Variable(n)
objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q * x )
constraints = [A * x <= u]
prob = cvxpy.Problem(objective, constraints)
prob.solve()
x_test = np.asarray(x.value).flatten()
y_test = (constraints[0].dual_value).A1
obj_value_test = objective.value
status_test = prob.status



# New data
q_new = np.array([1., 1.])
u_new = np.array([-2., 0., -20., 100., 80.])


# Solve with CVXPY and get solution for new data
x = cvxpy.Variable(n)
objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q_new * x )
constraints = [A * x <= u_new]
prob = cvxpy.Problem(objective, constraints)
prob.solve()
x_test_new = np.asarray(x.value).flatten()
y_test_new = (constraints[0].dual_value).A1
obj_value_test_new = objective.value
status_test_new = prob.status


# Generate problem solutions
sols_data = {'x_test': x_test,
             'y_test': y_test,
             'obj_value_test': obj_value_test,
             'status_test': status_test,
             'q_new': q_new,
             'u_new': u_new,
             'x_test_new': x_test_new,
             'y_test_new': y_test_new,
             'obj_value_test_new': obj_value_test_new,
             'status_test_new': status_test_new}


# Generate problem data
cu.generate_problem_data(P, q, A, l, u, 'basic_qp2', sols_data)


# import ipdb; ipdb.set_trace()




# Write new vectors
# f = open("basic_qp/basic_qp.h", "a")
#
# cu.write_vec_float(f, x.value, "sol_x")
#
# f.close()
