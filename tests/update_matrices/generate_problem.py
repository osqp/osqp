import numpy as np
import scipy.sparse as spa
import utils.codegen_utils as cu
import cvxpy



# Define tests
n = 5
m = 8
test_form_KKT_n = n
test_form_KKT_m = m
p = 0.7

test_form_KKT_A = spa.random(test_form_KKT_m, test_form_KKT_n, density=p).tocsc()
test_form_KKT_P = spa.random(n, n, density=p)
test_form_KKT_P = test_form_KKT_P.dot(test_form_KKT_P.T).tocsc() + spa.eye(n).tocsc()
test_form_KKT_Pu = spa.triu(test_form_KKT_P).tocsc()
test_form_KKT_rho = 1.6
test_form_KKT_sigma = 0.1
test_form_KKT_KKT = spa.vstack([
                        spa.hstack([test_form_KKT_P + test_form_KKT_sigma *
                        spa.eye(test_form_KKT_n), test_form_KKT_A.T]),
                     spa.hstack([test_form_KKT_A,
                        -1./test_form_KKT_rho * spa.eye(test_form_KKT_m)])]).tocsc()
test_form_KKT_KKTu = spa.triu(test_form_KKT_KKT).tocsc()


# Create new P, A and KKT
test_form_KKT_A_new = test_form_KKT_A.copy()
test_form_KKT_A_new.data += np.random.randn(test_form_KKT_A_new.nnz)
test_form_KKT_Pu_new = test_form_KKT_Pu.copy()
test_form_KKT_Pu_new.data += 0.1 * np.random.randn(test_form_KKT_Pu_new.nnz)
test_form_KKT_P_new = test_form_KKT_Pu_new + test_form_KKT_Pu_new.T - spa.diags(test_form_KKT_Pu_new.diagonal())

test_form_KKT_KKT_new = spa.vstack([
                        spa.hstack([test_form_KKT_P_new + test_form_KKT_sigma *
                        spa.eye(test_form_KKT_n), test_form_KKT_A_new.T]),
                     spa.hstack([test_form_KKT_A_new,
                        -1./test_form_KKT_rho * spa.eye(test_form_KKT_m)])]).tocsc()
test_form_KKT_KKTu_new = spa.triu(test_form_KKT_KKT_new).tocsc()




# Test solve problem with initial P and A
test_solve_P = test_form_KKT_P.copy()
test_solve_q = np.random.randn(n)
test_solve_A = test_form_KKT_A.copy()
test_solve_l = -30 + np.random.randn(m)
test_solve_u = 30 + np.random.randn(m)


# Solve problem with cvxpy
x = cvxpy.Variable(n)
objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, test_solve_P.todense()) + test_solve_q * x )
constraints = [test_solve_A.todense() * x <= test_solve_u, test_solve_l <= test_solve_A.todense() * x]
prob = cvxpy.Problem(objective, constraints)
prob.solve(abstol=1e-7, reltol=1e-7)
test_solve_x = np.asarray(x.value).flatten()
test_solve_y = (constraints[0].dual_value - constraints[1].dual_value).A1
test_solve_obj_value = objective.value
test_solve_status = prob.status

# Solve with new P
test_solve_P_new = test_form_KKT_P_new.copy()
x = cvxpy.Variable(n)
objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, test_solve_P_new) + test_solve_q * x )
constraints = [test_solve_A * x <= test_solve_u, test_solve_l <= test_solve_A * x]
prob = cvxpy.Problem(objective, constraints)
prob.solve(abstol=1e-7, reltol=1e-7)
test_solve_P_new_x = np.asarray(x.value).flatten()
test_solve_P_new_y = (constraints[0].dual_value - constraints[1].dual_value).A1
test_solve_P_new_obj_value = objective.value
test_solve_P_new_status = prob.status



# Solve with new A
test_solve_A_new = test_form_KKT_A_new.copy()
x = cvxpy.Variable(n)
objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, test_solve_P) + test_solve_q * x )
constraints = [test_solve_A_new * x <= test_solve_u, test_solve_l <= test_solve_A_new * x]
prob = cvxpy.Problem(objective, constraints)
prob.solve(abstol=1e-7, reltol=1e-7)
test_solve_A_new_x = np.asarray(x.value).flatten()
test_solve_A_new_y = (constraints[0].dual_value - constraints[1].dual_value).A1
test_solve_A_new_obj_value = objective.value
test_solve_A_new_status = prob.status


# Solve with new P and A
x = cvxpy.Variable(n)
objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, test_solve_P_new) + test_solve_q * x )
constraints = [test_solve_A_new * x <= test_solve_u, test_solve_l <= test_solve_A_new * x]
prob = cvxpy.Problem(objective, constraints)
prob.solve(abstol=1e-7, reltol=1e-7)
test_solve_P_A_new_x = np.asarray(x.value).flatten()
test_solve_P_A_new_y = (constraints[0].dual_value - constraints[1].dual_value).A1
test_solve_P_A_new_obj_value = objective.value
test_solve_P_A_new_status = prob.status




# Generate test data and solutions
data = {'test_form_KKT_n':test_form_KKT_n,
        'test_form_KKT_m':test_form_KKT_m,
        'test_form_KKT_A': test_form_KKT_A,
        'test_form_KKT_P': test_form_KKT_P,
        'test_form_KKT_Pu': test_form_KKT_Pu,
        'test_form_KKT_rho': test_form_KKT_rho,
        'test_form_KKT_sigma': test_form_KKT_sigma,
        'test_form_KKT_KKT': test_form_KKT_KKT,
        'test_form_KKT_KKTu': test_form_KKT_KKTu,
        'test_form_KKT_A_new': test_form_KKT_A_new,
        'test_form_KKT_P_new': test_form_KKT_P_new,
        'test_form_KKT_Pu_new': test_form_KKT_Pu_new,
        'test_form_KKT_KKT_new': test_form_KKT_KKT_new,
        'test_form_KKT_KKTu_new': test_form_KKT_KKTu_new,
        'test_solve_P': test_solve_P,
        'test_solve_q': test_solve_q,
        'test_solve_A': test_solve_A,
        'test_solve_l': test_solve_l,
        'test_solve_u': test_solve_u,
        'n': n,
        'm': m,
        'test_solve_x': test_solve_x,
        'test_solve_y': test_solve_y,
        'test_solve_obj_value': test_solve_obj_value,
        'test_solve_status': test_solve_status,
        'test_solve_P_new': test_solve_P_new,
        'test_solve_A_new': test_solve_A_new,
        'test_solve_P_new_x': test_solve_P_new_x,
        'test_solve_P_new_y': test_solve_P_new_y,
        'test_solve_P_new_obj_value': test_solve_P_new_obj_value,
        'test_solve_P_new_status': test_solve_P_new_status,
        'test_solve_A_new_x': test_solve_A_new_x,
        'test_solve_A_new_y': test_solve_A_new_y,
        'test_solve_A_new_obj_value': test_solve_A_new_obj_value,
        'test_solve_A_new_status': test_solve_A_new_status,
        'test_solve_P_A_new_x': test_solve_P_A_new_x,
        'test_solve_P_A_new_y': test_solve_P_A_new_y,
        'test_solve_P_A_new_obj_value': test_solve_P_A_new_obj_value,
        'test_solve_P_A_new_status': test_solve_P_A_new_status
        }

# import ipdb; ipdb.set_trace()

# Generate test data
cu.generate_data('update_matrices', data)
