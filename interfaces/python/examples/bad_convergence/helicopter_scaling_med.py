# import osqp
import osqppurepy as osqp
import numpy as np
import scipy.sparse as spa

import pickle

# Load one problem
with open('./data/%s.pickle' % 'helicopter_scaling_med', 'rb') as f:
    problem = pickle.load(f)


# OSQP settings
osqp_settings = {'verbose': True,
                 'scaling': True,
                 'scaling_iter': 100,
                 'early_terminate_interval': 1,
                 'auto_rho': True,
                 'rho': 0.0001,
                 'polish': False}

# Solve with OSQP
model = osqp.OSQP()
model.setup(problem['P'], problem['q'], problem['A'],
            problem['l'], problem['u'], **osqp_settings)
res_osqp = model.solve()



# Solve with GUROBI
import mathprogbasepy as mpbpy
qp = mpbpy.QuadprogProblem(problem['P'], problem['q'], problem['A'],
                      problem['l'], problem['u'])
res_gurobi = qp.solve(solver=mpbpy.GUROBI, verbose=False)
print("GUROBI time = %.4e" % res_gurobi.cputime)
print("OSQP time = %.4e" % res_osqp.info.run_time)
