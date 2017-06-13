'''
Solve stupid QP using OSQP
'''
import numpy as np
import scipy.sparse as spa
import osqp

P = spa.csc_matrix(np.array([1000.]))
q = np.array([1.])
A = spa.csc_matrix(np.array([1.]))
l = np.array([-.5])
u = np.array([.5])

#  rho = P.todense().A1 / (A.todense().A1 ** 2)
#  rho = P.todense().A1 / (A.todense().A1 ** 2)
rho = 1.0

m = osqp.OSQP()
m.setup(P, q, A, l, u, rho=rho, auto_rho=False, scaling=False, early_terminate_interval=1)
res = m.solve()




