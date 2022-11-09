import numpy as np
from scipy import sparse
import scipy.sparse.linalg as sla
import utils.codegen_utils as cu
from numpy.random import Generator, PCG64

# Set random seed for reproducibility
rg = Generator(PCG64(1))

# Test sparse matrix construction vs dense
test_sp_matrix_Adns = np.around(.6*rg.random((5, 6))) + rg.standard_normal((5,6))
test_sp_matrix_A = sparse.csc_matrix(test_sp_matrix_Adns)


# Test vector operations
test_vec_ops_n = 10
test_vec_ops_vn = test_vec_ops_n * np.ones(test_vec_ops_n)
test_vec_ops_vn_neg = -1 * test_vec_ops_vn
test_vec_ops_ones = np.ones(test_vec_ops_n)
test_vec_ops_zero = np.zeros(test_vec_ops_n)
test_vec_ops_zero_int = test_vec_ops_zero.astype(int)
test_vec_ops_v1 = rg.standard_normal(test_vec_ops_n)
test_vec_ops_v2 = rg.standard_normal(test_vec_ops_n)
test_vec_ops_v3 = rg.standard_normal(test_vec_ops_n)
test_vec_ops_pos_v1 = abs(test_vec_ops_v1)
test_vec_ops_neg_v1 = -test_vec_ops_v1
test_vec_ops_neg_v2 = -test_vec_ops_v2
test_vec_ops_neg_v3 = -test_vec_ops_v3
test_vec_ops_shift_v1 = test_vec_ops_v1 + 4
test_vec_ops_shift_v2 = -test_vec_ops_v2 - 4
test_vec_ops_sc1 = rg.standard_normal()
test_vec_ops_sc2 = rg.standard_normal()
test_vec_ops_sc3 = rg.standard_normal()
test_vec_ops_same = np.zeros(test_vec_ops_n)
test_vec_ops_same.fill(test_vec_ops_sc1)
test_vec_ops_norm_1 = np.linalg.norm(test_vec_ops_v1, 1)
test_vec_ops_pos_norm_1 = np.linalg.norm(test_vec_ops_pos_v1, 1)
test_vec_ops_neg_norm_1 = np.linalg.norm(test_vec_ops_neg_v1, 1)
test_vec_ops_norm_2 = np.linalg.norm(test_vec_ops_v1, 2)
test_vec_ops_norm_inf = np.linalg.norm(test_vec_ops_v1, np.inf)
test_vec_ops_norm_inf_scaled = np.linalg.norm(test_vec_ops_v1 * test_vec_ops_v2, np.inf)
test_vec_ops_norm_inf_diff = np.linalg.norm(test_vec_ops_v1 - test_vec_ops_v2,
                                            np.inf)
test_vec_ops_add = test_vec_ops_v1 + test_vec_ops_v2
test_vec_ops_sub = test_vec_ops_v1 - test_vec_ops_v2
test_vec_ops_add_scaled = test_vec_ops_sc1 * test_vec_ops_v1 + test_vec_ops_sc2 * test_vec_ops_v2
test_vec_ops_add_scaled_inc = test_vec_ops_v1 + test_vec_ops_sc2 * test_vec_ops_v2
test_vec_ops_add_scaled3 = test_vec_ops_sc1 * test_vec_ops_v1 + test_vec_ops_sc2 * test_vec_ops_v2 + test_vec_ops_sc3 * test_vec_ops_v3
test_vec_ops_add_scaled3_inc = test_vec_ops_v1 + test_vec_ops_sc2 * test_vec_ops_v2 + test_vec_ops_sc3 * test_vec_ops_v3
test_vec_ops_ew_sqrt = np.sqrt(test_vec_ops_shift_v1)
test_vec_ops_ew_reciprocal = np.reciprocal(test_vec_ops_v1)
test_vec_ops_ew_prod = test_vec_ops_v1 * test_vec_ops_v2
test_vec_ops_sca_prod = test_vec_ops_sc1 * test_vec_ops_v1
test_vec_ops_vec_dot = test_vec_ops_v1@test_vec_ops_v2
test_vec_ops_vec_dot_v1 = test_vec_ops_v1@test_vec_ops_v1
test_vec_ops_vec_dot_pos = test_vec_ops_v1[(test_vec_ops_v2 > 0)]@test_vec_ops_v2[(test_vec_ops_v2 > 0)]
test_vec_ops_vec_dot_neg = test_vec_ops_v1[(test_vec_ops_v2 < 0)]@test_vec_ops_v2[(test_vec_ops_v2 < 0)]
test_vec_ops_vec_dot_pos_flip = test_vec_ops_v2[(test_vec_ops_v1 > 0)]@test_vec_ops_v1[(test_vec_ops_v1 > 0)]
test_vec_ops_vec_dot_neg_flip = test_vec_ops_v2[(test_vec_ops_v1 < 0)]@test_vec_ops_v1[(test_vec_ops_v1 < 0)]
test_vec_ops_vec_dot_pos_v1 = test_vec_ops_v1[(test_vec_ops_v1 > 0)]@test_vec_ops_v1[(test_vec_ops_v1 > 0)]
test_vec_ops_vec_dot_neg_v1 = test_vec_ops_v1[(test_vec_ops_v1 < 0)]@test_vec_ops_v1[(test_vec_ops_v1 < 0)]
test_vec_ops_ew_bound_vec = np.minimum(test_vec_ops_v2, np.maximum(test_vec_ops_v1, test_vec_ops_v3))
test_vec_ops_ew_max_vec = np.maximum(test_vec_ops_v1, test_vec_ops_v2)
test_vec_ops_ew_min_vec = np.minimum(test_vec_ops_v1, test_vec_ops_v2)
test_vec_subvec_ind0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
test_vec_subvec_ind5 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
test_vec_subvec_ind10 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
test_vec_subvec_0 = test_vec_ops_v1[(test_vec_subvec_ind0 == 1)]
test_vec_subvec_5 = test_vec_ops_v1[(test_vec_subvec_ind5 == 1)]
test_vec_subvec_assign_5 = np.copy(test_vec_ops_v1)
test_vec_subvec_assign_5[2:(2+5)] = test_vec_subvec_5
test_vec_ops_sca_lt = np.copy(test_vec_ops_v1)
test_vec_ops_sca_lt[test_vec_ops_sca_lt < test_vec_ops_sc1] = test_vec_ops_sc2
test_vec_ops_sca_gt = np.copy(test_vec_ops_v1)
test_vec_ops_sca_gt[test_vec_ops_sca_gt > test_vec_ops_sc1] = test_vec_ops_sc2
test_vec_ops_sca_cond = np.sign(test_vec_ops_v1).astype(int)
test_vec_ops_sca_cond[2] = 0
test_vec_ops_sca_cond[6] = 0
test_vec_ops_sca_cond_res = np.zeros(test_vec_ops_n)
test_vec_ops_sca_cond_res[(test_vec_ops_sca_cond < 0)] = test_vec_ops_sc1
test_vec_ops_sca_cond_res[(test_vec_ops_sca_cond == 0)] = test_vec_ops_sc2
test_vec_ops_sca_cond_res[(test_vec_ops_sca_cond > 0)] = test_vec_ops_sc3

#test_vec_ops_ew_min_vec = np.minimum(test_vec_ops_v1, test_vec_ops_v2)


# Test matrix operations
test_mat_ops_n = 2
test_mat_ops_A = sparse.random(test_mat_ops_n, test_mat_ops_n, density=0.8, format='csc', random_state=rg)
test_mat_ops_d = rg.standard_normal(test_mat_ops_n)
D = sparse.diags(test_mat_ops_d, format='csc')
test_mat_ops_prem_diag = (D@test_mat_ops_A).tocoo().tocsc()   # Force matrix reordering
test_mat_ops_postm_diag = (test_mat_ops_A@D).tocoo().tocsc()  # Force matrix reordering
test_mat_ops_scaled = (2*test_mat_ops_A).tocoo().tocsc()  # Force matrix reordering
test_mat_ops_inf_norm_cols = np.amax(np.abs(
    np.asarray(test_mat_ops_A.todense())), axis=0)
test_mat_ops_inf_norm_rows = np.amax(np.abs(
    np.asarray(test_mat_ops_A.todense())), axis=1)

test_mat_ops_diag_m = test_vec_ops_n
test_mat_ops_diag_n = 6
test_mat_ops_diag_A = sparse.random(test_mat_ops_diag_n, test_mat_ops_diag_n, density=0.8, format='csc', random_state=rg)
test_mat_ops_diag_Ar = sparse.random(test_mat_ops_diag_m, test_mat_ops_diag_n, density=0.8, format='csc', random_state=rg)
test_mat_ops_diag_P = test_mat_ops_diag_A + test_mat_ops_diag_A.T
test_mat_ops_diag_Pu = sparse.triu(test_mat_ops_diag_P, format='csc')
test_mat_ops_diag_dA = np.diag(sparse.csc_matrix.toarray(test_mat_ops_diag_A))
test_mat_ops_diag_dP = np.diag(sparse.csc_matrix.toarray(test_mat_ops_diag_Pu))
test_mat_ops_diag_AtA = np.diag((test_mat_ops_diag_Ar.T@test_mat_ops_diag_Ar).todense())
tmpR = sparse.diags(test_vec_ops_vn, format='csc')
test_mat_ops_diag_AtRnA = np.diag((test_mat_ops_diag_Ar.T@tmpR@test_mat_ops_diag_Ar).todense())
tmpR = sparse.diags(test_vec_ops_v1, format='csc')
test_mat_ops_diag_AtRA = np.diag((test_mat_ops_diag_Ar.T@tmpR@test_mat_ops_diag_Ar).todense())

# Special cases for matrices/vectors with no size or entries
test_mat_no_entries = sparse.csc_matrix([[0., 0.], [0., 0.]])
test_mat_no_rows = sparse.csc_matrix((0,2))
test_mat_no_cols = sparse.csc_matrix((2,0))
test_vec_empty = np.array([])
test_vec_mat_empty = np.array([1., 2.])
test_vec_zeros = np.array([0., 0.])

# Test matrix vector operations
m = 5
n = 4
p = 0.4

test_mat_vec_n = n
test_mat_vec_m = m
test_mat_vec_A = sparse.random(m, n, density=1.0, format='csc', random_state=rg)
test_mat_vec_P = sparse.random(n, n, density=0.8, format='csc', random_state=rg)
test_mat_vec_P = test_mat_vec_P + test_mat_vec_P.T
test_mat_vec_Pu = sparse.triu(test_mat_vec_P, format='csc')
test_mat_vec_x = rg.standard_normal(n)
test_mat_vec_y = rg.standard_normal(m)
test_mat_vec_Ax = test_mat_vec_A@test_mat_vec_x
test_mat_vec_Ax_cum = test_mat_vec_A@test_mat_vec_x + test_mat_vec_y
test_mat_vec_ATy = test_mat_vec_A.T@test_mat_vec_y
test_mat_vec_ATy_cum = test_mat_vec_A.T@test_mat_vec_y + test_mat_vec_x
test_mat_vec_Px = test_mat_vec_P@test_mat_vec_x
test_mat_vec_Px_cum = test_mat_vec_P@test_mat_vec_x + test_mat_vec_x

# Test extracting submatrices
test_submat_A4_num = 4
test_submat_A5_num = 5
test_submat_A3_num = 3
test_submat_A0_num = 0
test_submat_A4_ind = np.array([1, 1, 0, 1, 1])
test_submat_A5_ind = np.array([1, 1, 1, 1, 1])
test_submat_A3_ind = np.array([1, 0, 1, 0, 1])
test_submat_A0_ind = np.array([0, 0, 0, 0, 0])
test_submat_A4 = test_mat_vec_A[(test_submat_A4_ind == 1)]
test_submat_A5 = test_mat_vec_A[(test_submat_A5_ind == 1)]
test_submat_A3 = test_mat_vec_A[(test_submat_A3_ind == 1)]
test_submat_A0 = test_mat_vec_A[(test_submat_A0_ind == 1)]

# Test extract upper triangular
test_mat_extr_triu_n = 5
test_mat_extr_triu_P = sparse.random(test_mat_extr_triu_n, test_mat_extr_triu_n, density=0.8, format='csc', random_state=rg)
test_mat_extr_triu_P = test_mat_extr_triu_P + test_mat_extr_triu_P.T
test_mat_extr_triu_Pu = sparse.triu(test_mat_extr_triu_P, format='csc')
test_mat_extr_triu_P_inf_norm_cols = np.amax(np.abs(
    np.asarray(test_mat_extr_triu_P.todense())), axis=0)


# Test compute quad form
test_qpform_n = 4
test_qpform_P = sparse.random(test_qpform_n, test_qpform_n, density=0.8, format='csc', random_state=rg)
test_qpform_P = test_qpform_P + test_qpform_P.T
test_qpform_Pu = sparse.triu(test_qpform_P, format='csc')
test_qpform_x = rg.standard_normal(test_qpform_n)
test_qpform_value = .5 * test_qpform_x.T @ (test_qpform_P@test_qpform_x)


# Generate test data and solutions
data = {'test_sp_matrix_A': test_sp_matrix_A,
        'test_sp_matrix_Adns': test_sp_matrix_Adns,
        'test_vec_ops_n': test_vec_ops_n,
        'test_vec_ops_vn': test_vec_ops_vn,
        'test_vec_ops_vn_neg': test_vec_ops_vn_neg,
        'test_vec_ops_ones': test_vec_ops_ones,
        'test_vec_ops_zero': test_vec_ops_zero,
        'test_vec_ops_zero_int': test_vec_ops_zero_int,
        'test_vec_ops_v1': test_vec_ops_v1,
        'test_vec_ops_v2': test_vec_ops_v2,
        'test_vec_ops_v3': test_vec_ops_v3,
        'test_vec_ops_pos_v1': test_vec_ops_pos_v1,
        'test_vec_ops_neg_v1': test_vec_ops_neg_v1,
        'test_vec_ops_neg_v2': test_vec_ops_neg_v2,
        'test_vec_ops_neg_v3': test_vec_ops_neg_v3,
        'test_vec_ops_shift_v1': test_vec_ops_shift_v1,
        'test_vec_ops_shift_v2': test_vec_ops_shift_v2,
        'test_vec_ops_sc1': test_vec_ops_sc1,
        'test_vec_ops_sc2': test_vec_ops_sc2,
        'test_vec_ops_sc3': test_vec_ops_sc3,
        'test_vec_ops_same': test_vec_ops_same,
        'test_vec_ops_norm_1': test_vec_ops_norm_1,
        'test_vec_ops_pos_norm_1': test_vec_ops_pos_norm_1,
        'test_vec_ops_neg_norm_1': test_vec_ops_neg_norm_1,
        'test_vec_ops_norm_2': test_vec_ops_norm_2,
        'test_vec_ops_norm_inf': test_vec_ops_norm_inf,
        'test_vec_ops_norm_inf_scaled': test_vec_ops_norm_inf_scaled,
        'test_vec_ops_norm_inf_diff': test_vec_ops_norm_inf_diff,
        'test_vec_ops_sub': test_vec_ops_sub,
        'test_vec_ops_add': test_vec_ops_add,
        'test_vec_ops_add_scaled': test_vec_ops_add_scaled,
        'test_vec_ops_add_scaled_inc': test_vec_ops_add_scaled_inc,
        'test_vec_ops_add_scaled3': test_vec_ops_add_scaled3,
        'test_vec_ops_add_scaled3_inc': test_vec_ops_add_scaled3_inc,
        'test_vec_ops_ew_sqrt': test_vec_ops_ew_sqrt,
        'test_vec_ops_ew_reciprocal': test_vec_ops_ew_reciprocal,
        'test_vec_ops_ew_prod': test_vec_ops_ew_prod,
        'test_vec_ops_sca_prod': test_vec_ops_sca_prod,
        'test_vec_ops_vec_dot': test_vec_ops_vec_dot,
        'test_vec_ops_vec_dot_v1': test_vec_ops_vec_dot_v1,
        'test_vec_ops_vec_dot_pos': test_vec_ops_vec_dot_pos,
        'test_vec_ops_vec_dot_neg': test_vec_ops_vec_dot_neg,
        'test_vec_ops_vec_dot_pos_flip': test_vec_ops_vec_dot_pos_flip,
        'test_vec_ops_vec_dot_neg_flip': test_vec_ops_vec_dot_neg_flip,
        'test_vec_ops_vec_dot_pos_v1': test_vec_ops_vec_dot_pos_v1,
        'test_vec_ops_vec_dot_neg_v1': test_vec_ops_vec_dot_neg_v1,
        'test_vec_ops_ew_bound_vec': test_vec_ops_ew_bound_vec,
        'test_vec_ops_ew_max_vec': test_vec_ops_ew_max_vec,
        'test_vec_ops_ew_min_vec': test_vec_ops_ew_min_vec,
        'test_vec_subvec_ind0': test_vec_subvec_ind0,
        'test_vec_subvec_ind5': test_vec_subvec_ind5,
        'test_vec_subvec_ind10': test_vec_subvec_ind10,
        'test_vec_subvec_0': test_vec_subvec_0,
        'test_vec_subvec_5': test_vec_subvec_5,
        'test_vec_subvec_assign_5': test_vec_subvec_assign_5,
        'test_vec_ops_sca_lt': test_vec_ops_sca_lt,
        'test_vec_ops_sca_gt': test_vec_ops_sca_gt,
        'test_vec_ops_sca_cond': test_vec_ops_sca_cond,
        'test_vec_ops_sca_cond_res': test_vec_ops_sca_cond_res,
        #'test_vec_ops_ew_min_vec': test_vec_ops_ew_min_vec,
        'test_mat_ops_n': test_mat_ops_n,
        'test_mat_ops_A': test_mat_ops_A,
        'test_mat_ops_d': test_mat_ops_d,
        'test_mat_ops_prem_diag': test_mat_ops_prem_diag,
        'test_mat_ops_postm_diag': test_mat_ops_postm_diag,
        'test_mat_ops_scaled' : test_mat_ops_scaled,
        'test_mat_ops_inf_norm_cols': test_mat_ops_inf_norm_cols,
        'test_mat_ops_inf_norm_rows': test_mat_ops_inf_norm_rows,
        'test_mat_ops_diag_m': test_mat_ops_diag_m,
        'test_mat_ops_diag_n': test_mat_ops_diag_n,
        'test_mat_ops_diag_A': test_mat_ops_diag_A,
        'test_mat_ops_diag_Ar': test_mat_ops_diag_Ar,
        'test_mat_ops_diag_P': test_mat_ops_diag_P,
        'test_mat_ops_diag_Pu': test_mat_ops_diag_Pu,
        'test_mat_ops_diag_dA': test_mat_ops_diag_dA,
        'test_mat_ops_diag_dP': test_mat_ops_diag_dP,
        'test_mat_ops_diag_AtA': test_mat_ops_diag_AtA,
        'test_mat_ops_diag_AtRnA': test_mat_ops_diag_AtRnA,
        'test_mat_ops_diag_AtRA': test_mat_ops_diag_AtRA,
        'test_mat_vec_n': test_mat_vec_n,
        'test_mat_vec_m': test_mat_vec_m,
        'test_mat_vec_A': test_mat_vec_A,
        'test_mat_vec_Pu': test_mat_vec_Pu,
        'test_mat_vec_x': test_mat_vec_x,
        'test_mat_vec_y': test_mat_vec_y,
        'test_mat_vec_Ax': test_mat_vec_Ax,
        'test_mat_vec_Ax_cum': test_mat_vec_Ax_cum,
        'test_mat_vec_ATy': test_mat_vec_ATy,
        'test_mat_vec_ATy_cum': test_mat_vec_ATy_cum,
        'test_mat_vec_Px': test_mat_vec_Px,
        'test_mat_vec_Px_cum': test_mat_vec_Px_cum,
        'test_submat_A4_num': test_submat_A4_num,
        'test_submat_A5_num': test_submat_A5_num,
        'test_submat_A3_num': test_submat_A3_num,
        'test_submat_A0_num': test_submat_A0_num,
        'test_submat_A4_ind': test_submat_A4_ind,
        'test_submat_A5_ind': test_submat_A5_ind,
        'test_submat_A3_ind': test_submat_A3_ind,
        'test_submat_A0_ind': test_submat_A0_ind,
        'test_submat_A4': test_submat_A4,
        'test_submat_A5': test_submat_A5,
        'test_submat_A3': test_submat_A3,
        'test_submat_A0': test_submat_A0,
        'test_mat_extr_triu_n': test_mat_extr_triu_n,
        'test_mat_extr_triu_P': test_mat_extr_triu_P,
        'test_mat_extr_triu_Pu': test_mat_extr_triu_Pu,
        'test_mat_extr_triu_P_inf_norm_cols':
        test_mat_extr_triu_P_inf_norm_cols,
        'test_qpform_n': test_qpform_n,
        'test_qpform_Pu': test_qpform_Pu,
        'test_qpform_x': test_qpform_x,
        'test_qpform_value': test_qpform_value,
        'test_mat_no_entries' : test_mat_no_entries,
        'test_mat_no_rows' : test_mat_no_rows,
        'test_mat_no_cols' : test_mat_no_cols,
        'test_vec_empty' : test_vec_empty,
        'test_vec_mat_empty' : test_vec_mat_empty,
        'test_vec_zeros' : test_vec_zeros,
        }

# Generate test data
cu.generate_data('lin_alg', data)
