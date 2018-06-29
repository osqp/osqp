import numpy as np
import scipy.sparse as spa
import utils.codegen_utils as cu

P = spa.csc_matrix(np.array([[2., 5.], [5., 1.]]))
q = np.array([3., 4.])

A = spa.csc_matrix(np.array([[-1.0, 0.], [0., -1.], [-1., 3.],
                             [2., 5.], [3., 4]]))
l = -np.inf * np.ones(A.shape[0])
u = np.array([0., 0., -15., 100., 80.])

n = P.shape[0]
m = A.shape[0]

# New data
q_new = np.array([1., 1.])
u_new = np.array([-2., 0., -20., 100., 80.])

# Generate problem data
cu.generate_problem_data(P, q, A, l, u, 'non_cvx')
