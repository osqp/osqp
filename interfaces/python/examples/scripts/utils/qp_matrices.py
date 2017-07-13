"""
Structure to define QP problems
"""
import os
import pandas as pd
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp

class QPmatrices(object):
    """
    QP problem matrices

    If this structure describes multiple problems, the elements can have higher dimensions. For example, if there are multiple linear costs, q becomes a 2d array.
    """
    def __init__(self, P, q, A, l, u, lx=None, ux=None):
        self.P = P

        if q.ndim > 1:
            self.q_vec = q
        else:
            self.q = q

        self.A = A
        self.l = l
        self.u = u
        self.lx = lx
        self.ux = ux
        self.nnzA = A.nnz
        self.nnzP = P.nnz

    def is_optimal(self, x, y, eps_abs=1e-03, eps_rel=1e-03):
        '''
        Check optimality condition of the QP given the
        primal-dual solution (x, y) and the tolerance eps
        '''

        # Get problem matrices
        P = self.P
        q = self.q
        A = self.A
        l = self.l
        u = self.u

        # Check primal feasibility
        Ax = A.dot(x)
        eps_pri = eps_abs + eps_rel * la.norm(Ax, np.inf)
        pri_res = np.minimum(Ax - l, 0) + np.maximum(Ax - u, 0)

        if la.norm(pri_res, np.inf) > eps_pri:
            print("Error in primal residual: %.4e > %.4e" %
                  (la.norm(pri_res, np.inf), eps_pri), end='')
            return False

        # Check dual feasibility
        Px = P.dot(x)
        Aty = A.T.dot(y)
        eps_dua = eps_abs + eps_rel * np.max([la.norm(Px, np.inf),
                                              la.norm(q, np.inf),
                                              la.norm(Aty, np.inf)])
        dua_res = Px + q + Aty

        if la.norm(dua_res, np.inf) > eps_dua:
            print("Error in dual residual: %.4e > %.4e" %
                  (la.norm(dua_res, np.inf), eps_dua), end='')
            return False

        # Check complementary slackness
        y_plus = np.maximum(y, 0)
        y_minus = np.minimum(y, 0)

        eps_comp = eps_abs + eps_rel * la.norm(Ax, np.inf)

        comp_res_u = np.minimum(np.abs(y_plus), np.abs(Ax - u))
        comp_res_l = np.minimum(np.abs(y_minus), np.abs(Ax - l))

        if la.norm(comp_res_l, np.inf) > eps_comp:
            print("Error in complementary slackness residual l: %.4e > %.4e" %
                  (la.norm(comp_res_l, np.inf), eps_dua), end='')
            return False

        if la.norm(comp_res_u, np.inf) > eps_comp:
            print("Error in complementary slackness residual u: %.4e > %.4e" %
                  (la.norm(comp_res_u, np.inf), eps_dua), end='')
            return False

        # If we arrived until here, the solution is optimal
        return True


def store_dimensions(example_name, dims_dict):
    dims_table = pd.DataFrame(dims_dict)
    # dims_table = dims_table[cols]

    data_dir = 'scripts/%s/data' % example_name

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    dims_table.to_csv('%s/dimensions.csv' % data_dir,
                            index=False)

     # Converting results to latex table and storing them to a file
    formatter = lambda x: '%1.2f' % x
    latex_table = dims_table.to_latex(header=False, index=False,
                                            float_format=formatter)
    f = open('%s/dimensions.tex' % data_dir, 'w')
    f.write(latex_table)
    f.close()
