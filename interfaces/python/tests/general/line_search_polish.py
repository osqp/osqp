# Test QP solver against Maros Mezaros Benchmark suite
import numpy as np
import numpy.linalg as la
import scipy.io as spio
import osqppurepy
import matplotlib.pyplot as plt

# Get data
f = '../../../../extra/difficult_problems/data/polish_fail.mat'
m = spio.loadmat(f)
P = m['P']
A = m['A']
q = m['q'].T.flatten()
l = m['l'].T.flatten()
u = m['u'].T.flatten()

# Solve problem
osqp_opts = {'polish': True,
             'early_terminate_interval': 1}
prob = osqppurepy.OSQP()
prob.setup(P=P, q=q, A=A, l=l, u=u, **osqp_opts)
res = prob.solve()

# Compute primal and dual residuals for the points obtained from line search
N = len(res.linesearch.t)
pri_res_ls = np.zeros(N)
dua_res_ls = np.zeros(N)
X_ls = res.linesearch.X
Z_ls = res.linesearch.Z
Y_ls = res.linesearch.Y

for i in range(N):
    pri_res_ls[i] = la.norm(A.dot(X_ls[i, :]) - Z_ls[i, :], np.inf)
    dua_res_ls[i] = la.norm(P.dot(X_ls[i, :]) + q + A.T.dot(Y_ls[i, :]), np.inf)

# Plot residuals versus stepsize t
fig, ax = plt.subplots()
ax.plot(res.linesearch.t, pri_res_ls, label='Primal residual')
ax.plot(res.linesearch.t, dua_res_ls, label='Dual residual')
plt.legend()
plt.grid()
ax.set_xlabel(r't')
ax.set_ylabel(r'Unscaled residuals (inf-norm)')
plt.show(block=False)
