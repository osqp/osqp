import scipy.io as spio
import scipy.sparse as spspa
import numpy as np

import mathprogbasepy as mpbpy

def load_maros_meszaros_problem(f):
    # Load file
    m = spio.loadmat(f)

    # Convert matrices
    P = m['Q'].astype(float).tocsc()
    n = P.shape[0]
    q = m['c'].T.flatten().astype(float)
    A = m['A'].astype(float)
    A = spspa.vstack([A, spspa.eye(n)]).tocsc()
    u = np.append(m['ru'].T.flatten().astype(float),
                  m['ub'].T.flatten().astype(float))
    l = np.append(m['rl'].T.flatten().astype(float),
                  m['lb'].T.flatten().astype(float))
    # Define problem
    p = mpbpy.QuadprogProblem(P, q, A, l, u)

    return p
