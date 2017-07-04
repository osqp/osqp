"""
Structure to define QP problems
"""
import os
import pandas as pd


class QPmatrices(object):
    """
    QP problem matrices

    If this structure describes multiple problems, the elements can have higher dimensions. For example, if there are multiple linear costs, q becomes a 2d array.
    """
    def __init__(self, P, q, A, l, u, lx=None, ux=None):
        self.P = P
        self.q = q
        self.A = A
        self.l = l
        self.u = u
        self.lx = lx
        self.ux = ux
        self.nnzA = A.nnz
        self.nnzP = P.nnz



def store_dimensions(example_name, dims_dict, cols):
    dims_table = pd.DataFrame(dims_dict)
    dims_table = dims_table[cols]

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
