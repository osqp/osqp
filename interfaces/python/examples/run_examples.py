"""
Run all Examples in the OSQP paper

Solver names are:
    - osqp
    - osqp_coldstart
    - osqp_no_caching
    - gurobi
    - mosek
    - ecos
"""
import numpy as np

# Import problems
# from scripts.portfolio.portfolio_example import run_portfolio_example
# from scripts.lasso.lasso_example import run_lasso_example
# from scripts.eq_qp.eq_qp_example import run_eq_qp_example
# from scripts.huber.huber_example import run_huber_example
# from scripts.svm.svm_example import run_svm_example
# from scripts.mpc.mpc_example import run_mpc_example

# Import problems
from scripts.lasso.lasso_example import LassoExample

# OSQP settings
osqp_settings = {'auto_rho': False,
                 'rho': 1.0,
                 'polish': False,
                 'verbose': False}

# Lasso problem
n_lasso = np.array([10, 20])
lambdas_lasso = np.logspace(-1, 1, 11)
solvers_lasso = ['osqp',
                 'osqp_coldstart',
                 'osqp_no_caching',
                 'gurobi',
                 'mosek',
                 'ecos']


'''
Solve problems
'''
# Lasso
lasso = LassoExample(n_lasso, solvers_lasso, lambdas_lasso)
lasso.run(osqp_settings)



# run_eq_qp_example(osqp_settings)
#  run_portfolio_example()
# run_lasso_example(osqp_settings)
#  run_huber_example()
#  run_svm_example()
#  run_mpc_example('helicopter')
#  run_mpc_example('pendulum')
