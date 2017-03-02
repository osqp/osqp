import numpy as np
import scipy.sparse as spa
import utils.codegen_utils as cu
import cvxpy
import osqp



# Define tests
test_form_KKT_n = 5
test_form_KKT_m = 8
p = 0.3

test_form_KKT_A = spa.random(test_form_KKT_m, test_form_KKT_n, density=p).tocsc()
test_form_KKT_P = spa.random(test_form_KKT_n, test_form_KKT_n, density=p).tocsc()
test_form_KKT_P = test_form_KKT_P + test_form_KKT_P.T
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
test_form_KKT_Pu_new.data += np.random.randn(test_form_KKT_Pu_new.nnz)
test_form_KKT_P_new = test_form_KKT_Pu_new + test_form_KKT_Pu_new.T - spa.diags(test_form_KKT_Pu_new.diagonal())

test_form_KKT_KKT_new = spa.vstack([
                        spa.hstack([test_form_KKT_P_new + test_form_KKT_sigma *
                        spa.eye(test_form_KKT_n), test_form_KKT_A_new.T]),
                     spa.hstack([test_form_KKT_A_new,
                        -1./test_form_KKT_rho * spa.eye(test_form_KKT_m)])]).tocsc()
test_form_KKT_KKTu_new = spa.triu(test_form_KKT_KKT_new).tocsc()



import ipdb; ipdb.set_trace()

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
        'test_form_KKT_KKTu_new': test_form_KKT_KKTu_new
        }

# Generate test data
cu.generate_data('update_matrices', data)
