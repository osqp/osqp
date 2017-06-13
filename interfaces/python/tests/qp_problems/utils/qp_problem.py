import numpy as np


class QPProblem(object):

    """QP problem"""

    def __init__(self, P=None, q=None, 
            A=None, l=None, u=None):

        #
        # Get problem dimensions
        #
        if P is None:
            if q is not None:
                self.n = len(q)
            elif A is not None:
                self.n = A.shape[1]
            else:
                raise ValueError("The problem does not have any variables")
        else:
            self.n = P.shape[0]
        if A is None:
            self.m = 0
        else:
            self.m = A.shape[0]

        self.P = P
        self.q = q
        self.A = A
        self.l = l if l is not None else -np.inf*np.ones(P.shape[0])
        self.u = u if u is not None else np.inf*np.ones(P.shape[0])
