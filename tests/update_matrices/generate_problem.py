import numpy as np
from scipy import sparse
import utils.codegen_utils as cu
from numpy.random import Generator, PCG64

# Set random seed for reproducibility
rg = Generator(PCG64(2))

# Define tests
n = 5
m = 8
test_form_KKT_n = n
test_form_KKT_m = m
p = 0.7

test_form_KKT_A = sparse.random(test_form_KKT_m, test_form_KKT_n, density=p, format='csc', random_state=rg)
test_form_KKT_P = sparse.random(n, n, density=p, random_state=rg)
test_form_KKT_P = (test_form_KKT_P @ test_form_KKT_P.T).tocsc() + sparse.eye(n, format='csc')
test_form_KKT_Pu = sparse.triu(test_form_KKT_P, format='csc')
test_form_KKT_rho = 1.6
test_form_KKT_sigma = 0.1
test_form_KKT_KKT = sparse.bmat([[test_form_KKT_P + test_form_KKT_sigma *
                                  sparse.eye(test_form_KKT_n), test_form_KKT_A.T],
                                 [test_form_KKT_A, -1./test_form_KKT_rho *
                                  sparse.eye(test_form_KKT_m)]], format='csc')
test_form_KKT_KKTu = sparse.triu(test_form_KKT_KKT, format='csc')


# Create new P, A and KKT
test_form_KKT_A_new = test_form_KKT_A.copy()
test_form_KKT_A_new.data += rg.standard_normal(test_form_KKT_A_new.nnz)
test_form_KKT_Pu_new = test_form_KKT_Pu.copy()
test_form_KKT_Pu_new.data += 0.1 * rg.standard_normal(test_form_KKT_Pu_new.nnz)
test_form_KKT_P_new = test_form_KKT_Pu_new + test_form_KKT_Pu_new.T - sparse.diags(test_form_KKT_Pu_new.diagonal())

test_form_KKT_KKT_new = sparse.bmat([[test_form_KKT_P_new + test_form_KKT_sigma *
                                      sparse.eye(test_form_KKT_n), test_form_KKT_A_new.T],
                                     [test_form_KKT_A_new, -1./test_form_KKT_rho *
                                      sparse.eye(test_form_KKT_m)]], format='csc')
test_form_KKT_KKTu_new = sparse.triu(test_form_KKT_KKT_new, format='csc')


# Test solve problem with initial P and A
test_solve_P = test_form_KKT_P.copy()
test_solve_Pu = test_form_KKT_Pu.copy()
test_solve_q = rg.standard_normal(n)
test_solve_A = test_form_KKT_A.copy()
test_solve_l = -30 + rg.standard_normal(m)
test_solve_u = 30 + rg.standard_normal(m)


# Define new P
test_solve_P_new = test_form_KKT_P_new.copy()
test_solve_Pu_new = test_form_KKT_Pu_new.copy()


# Define new A
test_solve_A_new = test_form_KKT_A_new.copy()


# Generate test data and solutions
data = {'test_form_KKT_n': test_form_KKT_n,
        'test_form_KKT_m': test_form_KKT_m,
        'test_form_KKT_A': test_form_KKT_A,
        'test_form_KKT_Pu': test_form_KKT_Pu,
        'test_form_KKT_rho': test_form_KKT_rho,
        'test_form_KKT_sigma': test_form_KKT_sigma,
        'test_form_KKT_KKT': test_form_KKT_KKT,
        'test_form_KKT_KKTu': test_form_KKT_KKTu,
        'test_form_KKT_A_new': test_form_KKT_A_new,
        'test_form_KKT_Pu_new': test_form_KKT_Pu_new,
        'test_form_KKT_KKT_new': test_form_KKT_KKT_new,
        'test_form_KKT_KKTu_new': test_form_KKT_KKTu_new,
        'test_solve_Pu': test_solve_Pu,
        'test_solve_q': test_solve_q,
        'test_solve_A': test_solve_A,
        'test_solve_l': test_solve_l,
        'test_solve_u': test_solve_u,
        'n': n,
        'm': m,
        'test_solve_x': np.array([-4.61725223e-01, 7.97298788e-01,
                                  5.55470173e-04,  3.37603740e-01,
                                  -1.14060693e+00]),
        'test_solve_y': np.zeros(m),
        'test_solve_obj_value': -1.885431747787806,
        'test_solve_status': 'optimal',
        'test_solve_Pu_new': test_solve_Pu_new,
        'test_solve_P_new_x': np.array([-0.48845963, 0.70997599, -0.09017696,
                                        0.33176037, -1.01867464]),
        'test_solve_P_new_y': np.zeros(m),
        'test_solve_P_new_obj_value': -1.7649689689774013,
        'test_solve_P_new_status': 'optimal',
        'test_solve_A_new': test_solve_A_new,
        'test_solve_A_new_x': np.array([-4.61725223e-01, 7.97298788e-01,
                                        5.55470173e-04, 3.37603740e-01,
                                        -1.14060693e+00]),
        'test_solve_A_new_y': np.zeros(m),
        'test_solve_A_new_obj_value': -1.8854317477878062,
        'test_solve_A_new_status': 'optimal',
        'test_solve_P_A_new_x': np.array([-0.48845963, 0.70997599, -0.09017696,
                                          0.33176037, -1.01867464]),
        'test_solve_P_A_new_y': np.zeros(m),
        'test_solve_P_A_new_obj_value': -1.764968968977401,
        'test_solve_P_A_new_status': 'optimal'
        }


# Generate test data
cu.generate_data('update_matrices', data)
