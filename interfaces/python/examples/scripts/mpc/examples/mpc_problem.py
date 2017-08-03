import numpy as np


class MPCProblem(object):

    def __init__(self,
                 name,                           # Name
                 A, B,                           # Dynamics
                 R=None, Q=None, QN=None,        # Cost
                 umin=None, umax=None,           # Input bounds
                 xmin=None, xmax=None,           # State bounds
                 T=None, tmin=None, tmax=None,   # Final state constraints
                 x0=None,                        # Initial state
                 xr=None                         # Reference state
                 ):

        # Get problem name
        self.name = name

        # Get input dimensions
        n = A.shape[0]
        m = B.shape[1]

        # Get dynamics
        if A.shape[1] != n:
            raise ValueError('A matrix is not square')
        self.A = A
        if B.shape[0] != n:
            raise ValueError('B matrix dimensions not correct')
        self.B = B

        # Get cost
        self.R = R
        self.Q = Q
        self.QN = QN

        # Get input bounds
        self.umin = umin
        self.umax = umax

        # Get state bounds
        self.xmin = xmin
        self.xmax = xmax

        # Get final state constraints
        self.T = T
        self.tmin = tmin
        self.tmax = tmax

        # Get initial state
        if x0 is None:
            x0 = np.zeros(n)
        self.x0 = x0

        self.xr = xr
