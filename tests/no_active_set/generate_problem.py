import numpy as np
from scipy import sparse
import utils.codegen_utils as cu

P = sparse.triu([[11., 0.], [0., 5.]], format='csc')
q = np.array([0., 0.])

A = sparse.csc_matrix(np.array([[1., 1.], [1., 0.], [0., 1.]]))
l = np.array([-1., -0.5, -0.25])
u = np.array([1., 0.7, 0.7])

n = P.shape[0]
m = A.shape[0]

# Generate problem solutions
sols_data = {'x_test': np.array([0., 0.]),
             'y_test': np.array([0., 0., 0.]),
             'obj_value_test': 0.,
             'status_test': 'optimal'}


# Generate problem data
cu.generate_problem_data(P, q, A, l, u, 'no_active_set', sols_data)
