import numpy as np
from scipy import sparse
import utils.codegen_utils as cu

P = sparse.triu([[2., 5.], [5., 1.]], format='csc')
q = np.array([3., 4.])

A = sparse.csc_matrix([[-1., 0.], [0., -1.], [-1., 3.], [2., 5.], [3., 4]])
l = -np.inf * np.ones(A.shape[0])
u = np.array([0., 0., -15., 100., 80.])

sols_data = {'sigma_new': 5}

# Generate problem data
cu.generate_problem_data(P, q, A, l, u, 'non_cvx', sols_data)
