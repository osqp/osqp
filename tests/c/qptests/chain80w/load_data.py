import test_utils.codeutils as cu

# Load data in Python
import numpy as np
import scipy.sparse as spspa


def gen_chain80w_test():

    direct = "qptests/chain80w/data/"
    # Load data (convert to sparse)
    A = spspa.csc_matrix(np.loadtxt(direct+'A.oqp'))
    lA = np.loadtxt(direct+'lb.oqp')
    uA = np.loadtxt(direct+'ub.oqp')
    P = spspa.csc_matrix(np.loadtxt(direct+'H.oqp'))
    q = np.loadtxt(direct+'g.oqp')

    # Get only first elements of lbA and q
    lA = lA[0, :]
    uA = uA[0, :]
    q = q[0, :]

    # A = spspa.rand(4, 3, 0.3).tocsc()
    # lA = -sp.rand(A.shape[0])
    # uA = sp.rand(A.shape[0])
    # P = spspa.eye(A.shape[1]).tocsc()
    # q = sp.rand(P.shape[0])

    # Name of the problem
    problem_name = "chain80w"

    cu.generate_code(P, q, A, lA, uA, problem_name)
