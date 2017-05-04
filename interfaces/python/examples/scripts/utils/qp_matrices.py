"""
Structure to define QP problems
"""

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
