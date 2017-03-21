# Load data in Python
import numpy as np
import scipy.sparse as spspa
import test_utils.codeutils as cu
import ipdb


def gen_diesel_test():

    direct = "qptests/diesel/data/"
    # Load data (convert to sparse)
    A = spspa.csc_matrix(np.loadtxt(direct + 'A.oqp'))
    A = spspa.vstack([A, spspa.eye(A.shape[1])]).tocsc()
    l = np.loadtxt(direct + 'lbA.oqp')
    u = np.loadtxt(direct + 'ubA.oqp')
    l = np.hstack([l, np.loadtxt(direct + 'lb.oqp')])
    u = np.hstack([u, np.loadtxt(direct + 'ub.oqp')])
    P = spspa.csc_matrix(np.loadtxt(direct + 'H.oqp'))
    q = np.loadtxt(direct + 'g.oqp')

    ipdb.set_trace()
    # Get only first elements of l, u and q
    l = l[0, :]
    u = u[0, :]
    q = q[0, :]

    # A = spspa.rand(4, 3, 0.3).tocsc()
    # lA = -sp.rand(A.shape[0])
    # uA = sp.rand(A.shape[0])
    # P = spspa.eye(A.shape[1]).tocsc()
    # q = sp.rand(P.shape[0])

    # Write bounds as +/0 infinity
    # lx = -1*np.inf*np.ones(P.shape[0])
    # ux = np.inf*np.ones(P.shape[0])

    # Name of the problem
    problem_name = "diesel"

    cu.generate_code(P, q, A, l, u, problem_name)
