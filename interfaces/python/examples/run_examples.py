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
from scripts.eq_qp.eq_qp_example import EqqpExample
from scripts.lasso.lasso_example import LassoExample
from scripts.huber.huber_example import HuberExample
from scripts.svm.svm_example import SVMExample
from scripts.portfolio.portfolio_example import PortfolioExample
from scripts.mpc.mpc_example import MPCExample


# OSQP settings
osqp_settings = {'auto_rho': False,
                 'rho': 100.0,
                 'polish': False,
                 'verbose': False}

# Lasso problem
n_lasso = np.array([1, 2])
lambdas_lasso = np.logspace(-1, 1, 11)
solvers_lasso = ['osqp',
                 'osqp_coldstart',
                 'osqp_no_caching',
                 'gurobi',
                 'mosek',
                 #  'qpoases',   # It fails for big problems!
                 'ecos']

# Eq qp problem
n_eqqp = np.array([10, 20])
solvers_eqqp = ['osqp',
                'gurobi',
                'mosek',
                'ecos',
                'qpoases'
                ]

# Huber problem
n_huber = np.array([2, 3])
solvers_huber = ['osqp',
                 'gurobi',
                 'mosek',
                 'ecos',
                 'qpoases'
                 ]


# SVM problem
n_svm = np.array([2, 3, 4])
solvers_svm = ['osqp',
               'gurobi',
               'mosek',
               'ecos',
               'qpoases'
               ]

# Portfolio
n_portfolio = np.array([500, 600])
solvers_portfolio = [
                     'osqp',
                     'osqp_coldstart',
                     'osqp_no_caching',
                     'gurobi',
                     'mosek',
                     'ecos',
                     'qpoases'
                     ]
gammas_portfolio = np.logspace(-1, 1, 51)   # TODO: Change it?

# MPC
n_mpc = np.array([5, 10, 15, 20])
solvers_mpc = [
               'osqp',
               'osqp_coldstart',
               'osqp_no_caching',
               'gurobi',
               'mosek',
               'qpoases',
               'ecos'
               ]
'''
Solve problems
'''
# Lasso
# lasso = LassoExample(n_lasso, solvers_lasso, lambdas_lasso)
# lasso.run(osqp_settings)

# Eq qp
# eqqp = EqqpExample(n_eqqp, solvers_eqqp)
# eqqp.run(osqp_settings)

# Huber
# huber = HuberExample(n_huber, solvers_huber)
# huber.run(osqp_settings)

# SVM
# svm = SVMExample(n_svm, solvers_svm)
# svm.run(osqp_settings)

# Portfolio
# portfolio = PortfolioExample(n_portfolio, solvers_portfolio, gammas_portfolio)
# portfolio.run(osqp_settings)

# MPC
# Examples: ball, helicopter, pendulum
mpc = MPCExample(n_mpc, solvers_mpc, "ball")
mpc.run(osqp_settings)
