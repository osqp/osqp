# Test QP solver against Maros Mezaros Benchmark suite
import scipy.io as spio
import osqp
import osqppurepy

# Get data
f = '../../../../extra/difficult_problems/data/polish_fail.mat'
m = spio.loadmat(f)
P = m['P']
A = m['A']
q = m['q'].T.flatten()
l = m['l'].T.flatten()
u = m['u'].T.flatten()

osqp_opts = {'polish': True,
             'early_terminate_interval': 1}

model = osqp.OSQP()
model.setup(P=P, q=q, A=A, l=l, u=u, **osqp_opts)
res_osqp = model.solve()


model = osqppurepy.OSQP()
model.setup(P=P, q=q, A=A, l=l, u=u, **osqp_opts)
res_osqppurepy = model.solve()
