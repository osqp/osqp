from scipy import sparse
import utils.codegen_utils as cu
from numpy.random import Generator, PCG64

# Set random seed for reproducibility
rg = Generator(PCG64(2))

n = 50
m = 150

# Generate random Matrices
Pt = sparse.random(n, n, random_state=rg)
P = Pt.T@Pt + sparse.eye(n)
P = sparse.triu(P, format='csc')
q = rg.standard_normal(n)
A = sparse.random(m, n, random_state=rg).tolil()  # Lil for efficiency
u = 3 + rg.standard_normal(m)
l = -3 + rg.standard_normal(m)

# Make random problem primal infeasible
A[int(n/2), :] = A[int(n/2)+1, :]
l[int(n/2)] = u[int(n/2)+1] + 10 * rg.random()
u[int(n/2)] = l[int(n/2)] + 0.5

# Convert A to csc
A = A.tocsc()

# Generate problem solutions
sols_data = {'status_test': 'primal_infeasible'}


# Generate problem data
cu.generate_problem_data(P, q, A, l, u, 'primal_infeasibility', sols_data)
