import numpy as np
import scipy.sparse as spa
import utils.codegen_utils as cu
import cvxpy
import osqp



# Define tests


# Test sparse matrix construction vs dense
test_sp_matrix_Adns = np.around(.6*np.random.rand(5, 6)) + np.random.randn(5,6)
test_sp_matrix_A =  spa.csc_matrix(test_sp_matrix_Adns)


# Test vector operations
test_vec_ops_n = 10
test_vec_ops_v1 = np.random.randn(test_vec_ops_n)
test_vec_ops_v2 = np.random.randn(test_vec_ops_n)
test_vec_ops_sc = np.random.randn()
test_vec_ops_norm2_diff = np.linalg.norm(test_vec_ops_v1 - test_vec_ops_v2)
test_vec_ops_add_scaled = test_vec_ops_v1 + test_vec_ops_sc * test_vec_ops_v2
test_vec_ops_norm2 = np.linalg.norm(test_vec_ops_v1)
test_vec_ops_ew_reciprocal = np.reciprocal(test_vec_ops_v1)
test_vec_ops_vec_prod = test_vec_ops_v1.dot(test_vec_ops_v2)


# Test matrix operations
test_mat_ops_n = 5
test_mat_ops_A = spa.csc_matrix(np.random.randn(test_mat_ops_n, test_mat_ops_n))
test_mat_ops_d = np.random.randn(test_mat_ops_n)
test_mat_ops_prem_diag = spa.diags(test_mat_ops_d).dot(test_mat_ops_A)
test_mat_ops_postm_diag = test_mat_ops_A.dot(spa.diags(test_mat_ops_d))
test_mat_ops_ew_square = test_mat_ops_A.copy()
test_mat_ops_ew_square.data = np.square(test_mat_ops_ew_square.data)
test_mat_ops_ew_abs = test_mat_ops_A.copy()
test_mat_ops_ew_abs.data = np.abs(test_mat_ops_ew_abs.data)

# import ipdb; ipdb.set_trace()

# Generate test data and solutions
data = {'test_sp_matrix_A': test_sp_matrix_A,
        'test_sp_matrix_Adns': test_sp_matrix_Adns,
        'test_vec_ops_n': test_vec_ops_n,
        'test_vec_ops_v1': test_vec_ops_v1,
        'test_vec_ops_v2': test_vec_ops_v2,
        'test_vec_ops_sc': test_vec_ops_sc,
        'test_vec_ops_norm2_diff': test_vec_ops_norm2_diff,
        'test_vec_ops_add_scaled': test_vec_ops_add_scaled,
        'test_vec_ops_norm2': test_vec_ops_norm2,
        'test_vec_ops_ew_reciprocal': test_vec_ops_ew_reciprocal,
        'test_vec_ops_vec_prod': test_vec_ops_vec_prod,
        'test_mat_ops_n': test_mat_ops_n,
        'test_mat_ops_A': test_mat_ops_A,
        'test_mat_ops_d': test_mat_ops_d,
        'test_mat_ops_prem_diag': test_mat_ops_prem_diag,
        'test_mat_ops_postm_diag': test_mat_ops_postm_diag,
        'test_mat_ops_ew_square': test_mat_ops_ew_square,
        'test_mat_ops_ew_abs': test_mat_ops_ew_abs
        }

# Generate test data
cu.generate_data('lin_alg', data)
