import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import utils.codegen_utils as cu
from numpy.random import Generator, PCG64

# Set random seed for reproducibility
rg = Generator(PCG64(2))

# Simple case
test_solve_KKT_n = 3
test_solve_KKT_m = 4

test_solve_KKT_P = sparse.random(test_solve_KKT_n, test_solve_KKT_n,
                                 density=0.4, format='csc', random_state=rg)
test_solve_KKT_P = test_solve_KKT_P.dot(test_solve_KKT_P.T).tocsc()
test_solve_KKT_A = sparse.random(test_solve_KKT_m, test_solve_KKT_n,
                                 density=0.4, format='csc', random_state=rg)
test_solve_KKT_Pu = sparse.triu(test_solve_KKT_P, format='csc')

test_solve_KKT_rho = 4.0
test_solve_KKT_sigma = 1.0
test_solve_KKT_KKT = sparse.vstack([
                        sparse.hstack([test_solve_KKT_P + test_solve_KKT_sigma *
                        sparse.eye(test_solve_KKT_n), test_solve_KKT_A.T]),
                        sparse.hstack([test_solve_KKT_A,
                        -1./test_solve_KKT_rho * sparse.eye(test_solve_KKT_m)])
                        ], format='csc')
test_solve_KKT_rhs = rg.standard_normal(test_solve_KKT_m + test_solve_KKT_n)
test_solve_KKT_x = spla.splu(test_solve_KKT_KKT).solve(test_solve_KKT_rhs)

test_solve_KKT_x[test_solve_KKT_n:] = test_solve_KKT_rhs[test_solve_KKT_n:] + \
                                      test_solve_KKT_x[test_solve_KKT_n:] / test_solve_KKT_rho

# Generate test data and solutions
data = {'test_solve_KKT_n': test_solve_KKT_n,
        'test_solve_KKT_m': test_solve_KKT_m,
        'test_solve_KKT_A': test_solve_KKT_A,
        'test_solve_KKT_Pu': test_solve_KKT_Pu,
        'test_solve_KKT_rho': test_solve_KKT_rho,
        'test_solve_KKT_sigma': test_solve_KKT_sigma,
        'test_solve_KKT_KKT': test_solve_KKT_KKT,
        'test_solve_KKT_rhs': test_solve_KKT_rhs,
        'test_solve_KKT_x': test_solve_KKT_x
        }

# Generate test data
cu.generate_data('solve_linsys', data)
