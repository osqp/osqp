import numpy as np
import scipy.sparse as spa
import utils.codegen_utils as cu
import cvxpy
import osqp



# Define tests
test_solve_KKT_n = 5
test_solve_KKT_m = 6
p = 0.3

test_solve_KKT_A = spa.random(test_solve_KKT_m, test_solve_KKT_n, density=p).tocsc()
test_solve_KKT_P = spa.random(test_solve_KKT_n, test_solve_KKT_n, density=p).tocsc()
test_solve_KKT_P = test_solve_KKT_P + test_solve_KKT_P.T
test_solve_KKT_Pu = spa.triu(test_solve_KKT_P).tocsc()
test_solve_KKT_rho = 1.6
test_solve_KKT_sigma = 0.1
test_solve_KKT_KKT = spa.vstack([
                        spa.hstack([test_solve_KKT_P + test_solve_KKT_sigma *
                        spa.eye(test_solve_KKT_n), test_solve_KKT_A.T]),
                     spa.hstack([test_solve_KKT_A,
                        -1./test_solve_KKT_rho * spa.eye(test_solve_KKT_m)])]).tocsc()
test_solve_KKT_rhs = np.random.randn(test_solve_KKT_m + test_solve_KKT_n)
test_solve_KKT_x = spa.linalg.splu(test_solve_KKT_KKT.tocsc()).solve(test_solve_KKT_rhs)

# import ipdb; ipdb.set_trace()

# Generate test data and solutions
data = {'test_solve_KKT_n':test_solve_KKT_n,
        'test_solve_KKT_m':test_solve_KKT_m,
        'test_solve_KKT_A': test_solve_KKT_A,
        'test_solve_KKT_P': test_solve_KKT_P,
        'test_solve_KKT_Pu': test_solve_KKT_Pu,
        'test_solve_KKT_rho': test_solve_KKT_rho,
        'test_solve_KKT_sigma': test_solve_KKT_sigma,
        'test_solve_KKT_KKT': test_solve_KKT_KKT,
        'test_solve_KKT_rhs': test_solve_KKT_rhs,
        'test_solve_KKT_x': test_solve_KKT_x
        }

# Generate test data
cu.generate_data('solve_linsys', data)
