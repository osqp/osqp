# import osqp
import osqppurepy as osqp
import numpy as np
import scipy.sparse as spa

import pickle

# Load one problem
with open('./data/%s.pickle' % 'helicopter_scaling', 'rb') as f:
    problem = pickle.load(f)


# OSQP settings
osqp_settings = {'verbose': True,
                 'scaling': True,
                 'auto_rho': True,
                 'rho': 0.1,
                 'polish': False}

# Solve with OSQP
model = osqp.OSQP()
model.setup(problem['P'], problem['q'], problem['A'],
            problem['l'], problem['u'], **osqp_settings)
model.solve()
