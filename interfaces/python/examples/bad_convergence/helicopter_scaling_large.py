# import osqp
import osqppurepy as osqp
import numpy as np
import scipy.sparse as spa

import pickle

# Load one problem
with open('./data/%s.pickle' % 'helicopter_scaling_large', 'rb') as f:
    problem = pickle.load(f)


# OSQP settings
osqp_settings = {'verbose': True,
                 'scaling': True,
                 'scaling_iter': 100,
                 'early_terminate_interval': 1,
                 'auto_rho': True,
                 'alpha': 1.6,
                 'rho': 0.01,
                 'polish': False}


# Assign problem data
P = problem['P']
q = problem['q']
A = problem['A']
l = problem['l']
u = problem['u']


# Scale data?
# norm_scaling = np.linalg.norm(P.todense())
norm_scaling = 1
P /= norm_scaling
q /= norm_scaling

# Solve with OSQP
model = osqp.OSQP()
model.setup(P, q, A,
            l, u, **osqp_settings)
res_osqp = model.solve()



# Solve with GUROBI
import mathprogbasepy as mpbpy
qp = mpbpy.QuadprogProblem(problem['P'], problem['q'], problem['A'],
                      problem['l'], problem['u'])
res_gurobi = qp.solve(solver=mpbpy.GUROBI, verbose=False)
print("GUROBI time = %.4e" % res_gurobi.cputime)
print("OSQP time = %.4e" % res_osqp.info.run_time)
