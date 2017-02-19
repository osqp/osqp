import numpy as np
import scipy.sparse as sp

import mathprogbasepy as mpbpy

A = sp.eye(2)
P = sp.eye(2)

l = np.array([-2, -2])
q = np.array([-3, 4])
u = np.array([10, 10])


qp = mpbpy.QuadprogProblem(P, q, A, l, u)

resOSQP = qp.solve(solver=mpbpy.OSQP, rho=1.1, eps_rel=1e-4, eps_abs=1e-4,
                   polishing=True)
resGUROBI = qp.solve(solver=mpbpy.GUROBI)
