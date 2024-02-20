import numpy as np
from scipy import sparse
import utils.codegen_utils as cu

P = sparse.triu([[4., 1.], [1., 2.]], format='csc')
q = np.ones(2)

A = sparse.csc_matrix(np.array([[1., 1.], [1., 0.], [0., 1.], [0., 1.]]))
l = np.array([1., 0., 0., -np.inf])
u = np.array([1., 0.7, 0.7, np.inf])

n = P.shape[0]
m = A.shape[0]

dx_1 = np.array([0.01, -0.01])
dy_1 = np.array([0.01, 0.01, 0.01, 0.])

# Generate problem solutions
sols_data = {'x_test': np.array([0.3, 0.7]),
             'y_test': np.array([-2.9, 0.0, 0.2, 0.0]),
             'obj_value_test': 1.88,
             'status_test': 'optimal',
             'dx_zeros': np.zeros(n),
             'dy_zeros': np.zeros(m),
             'dx_1': dx_1,
             'dy_1': dy_1}

# Generate problem data
cu.generate_problem_data(P, q, A, l, u, 'derivative_adjoint', sols_data)
