import numpy as np
from scipy import sparse
import utils.codegen_utils as cu
from numpy.random import Generator, PCG64

# Set random seed for reproducibility
rg = Generator(PCG64(1))

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
test_form_KKT_KKT = sparse.bmat([
                      [test_form_KKT_P + test_form_KKT_sigma * sparse.eye(test_form_KKT_n), test_form_KKT_A.T],
                      [test_form_KKT_A, -1./test_form_KKT_rho * sparse.eye(test_form_KKT_m)]], format='csc')
test_form_KKT_KKTu = sparse.triu(test_form_KKT_KKT, format='csc')


# Create new P, A and KKT
test_form_KKT_A_new = test_form_KKT_A.copy()
test_form_KKT_A_new.data += rg.standard_normal(test_form_KKT_A_new.nnz)
test_form_KKT_Pu_new = test_form_KKT_Pu.copy()
test_form_KKT_Pu_new.data += 0.1 * rg.standard_normal(test_form_KKT_Pu_new.nnz)

test_form_KKT_A_new_idx = np.array(range(test_form_KKT_A_new.nnz))
test_form_KKT_A_new_n = test_form_KKT_A_new.nnz
test_form_KKT_Pu_new_idx = np.array(range(test_form_KKT_Pu_new.nnz))
test_form_KKT_Pu_new_n = test_form_KKT_Pu_new.nnz

test_form_KKT_P_new = test_form_KKT_Pu_new + test_form_KKT_Pu_new.T - sparse.diags(test_form_KKT_Pu_new.diagonal())

test_form_KKT_KKT_new = sparse.bmat([
                          [test_form_KKT_P_new + test_form_KKT_sigma * sparse.eye(test_form_KKT_n), test_form_KKT_A_new.T],
                          [test_form_KKT_A_new, -1./test_form_KKT_rho * sparse.eye(test_form_KKT_m)]], format='csc')
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
        'test_form_KKT_A_new_idx': test_form_KKT_A_new_idx,
        'test_form_KKT_A_new_n': test_form_KKT_A_new_n,
        'test_form_KKT_Pu_new': test_form_KKT_Pu_new,
        'test_form_KKT_Pu_new_idx': test_form_KKT_Pu_new_idx,
        'test_form_KKT_Pu_new_n': test_form_KKT_Pu_new_n,
        'test_form_KKT_KKT_new': test_form_KKT_KKT_new,
        'test_form_KKT_KKTu_new': test_form_KKT_KKTu_new,
        'test_solve_Pu': test_solve_Pu,
        'test_solve_q': test_solve_q,
        'test_solve_A': test_solve_A,
        'test_solve_l': test_solve_l,
        'test_solve_u': test_solve_u,
        'n': n,
        'm': m,
        'test_solve_x': np.array([-0.34282147, 0.0260615, 0.27197987,
                                  -0.01202531, 0.44052732]),
        'test_solve_y': np.zeros(m),
        'test_solve_obj_value': -0.37221,
        'test_solve_status': 'optimal',
        'test_solve_Pu_new': test_solve_Pu_new,
        'test_solve_P_new_x': np.array([-0.36033252, 0.10729745, 0.22965285,
                                         0.06323582, 0.42301398]),
        'test_solve_P_new_y': np.zeros(m),
        'test_solve_P_new_obj_value': -0.36765,
        'test_solve_P_new_status': 'optimal',
        'test_solve_A_new': test_solve_A_new,
        'test_solve_A_new_x': np.array([-0.34282147, 0.0260615, 0.27197987,
                                        -0.01202531, 0.44052732]),
        'test_solve_A_new_y': np.zeros(m),
        'test_solve_A_new_obj_value': -0.37221,
        'test_solve_A_new_status': 'optimal',
        'test_solve_P_A_new_x': np.array([-0.36033252, 0.10729745, 0.22965285,
                                           0.06323582, 0.42301398]),
        'test_solve_P_A_new_y': np.zeros(m),
        'test_solve_P_A_new_obj_value': -0.36765,
        'test_solve_P_A_new_status': 'optimal'
        }


# Generate test data
cu.generate_data('update_matrices', data)
