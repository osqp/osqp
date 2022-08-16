import numpy as np
from scipy import sparse
import utils.codegen_utils as cu

P = sparse.triu([[0., 0.], [0., 0.]], format='csc')
q = np.array([1.1, 0.95])

A = sparse.csc_matrix(np.array([[1., 1.], [1., 0.], [0., 1.], [0., 1.]]))
l = np.array([1., 0., 0., -np.inf])
u = np.array([1., 0.7, 0.7, np.inf])

n = P.shape[0]
m = A.shape[0]

# New data
q_new = np.array([2.5, 3.2])
l_new = np.array([0.8, -3.4, -np.inf, 0.5])
u_new = np.array([1.6, 1.0, np.inf, 0.5])

# Generate problem solutions
sols_data = {'x_test': np.array([0.3, 0.7]),
             'y_test': np.array([-1.1, 0.0, 0.15, 0.0]),
             'obj_value_test': 0.995,
             'status_test': 'optimal',
             'q_new': q_new,
             'l_new': l_new,
             'u_new': u_new}

# Generate problem data
cu.generate_problem_data(P, q, A, l, u, 'basic_lp', sols_data)
