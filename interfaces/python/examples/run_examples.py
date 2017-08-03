"""
Run all Examples in the OSQP paper

Solver names are:
    - osqp
    - osqp_coldstart
    - osqp_no_caching
    - gurobi
    - mosek
    - ecos
    - qpoases
"""
import numpy as np

# Import problems
from scripts.eq_qp.eq_qp_example import EqqpExample
from scripts.random_qp.random_qp_example import RandomqpExample
from scripts.lasso.lasso_example import LassoExample
from scripts.huber.huber_example import HuberExample
from scripts.svm.svm_example import SVMExample
from scripts.portfolio.portfolio_example import PortfolioExample
from scripts.mpc.mpc_example import MPCExample


# OSQP settings
osqp_settings = {'auto_rho': False,
                 'rho': 0.1,
                 'polish': False,
                 'scaling_norm': -1,
                 'verbose': False}

# Random qp problem
n_randomqp = np.array([10, 50, 100, 500, 1000])
solvers_randomqp = ['osqp',
                    'gurobi',
                    'mosek',
                    #  'ecos',
                    #  'qpoases'
                    ]

# Eq qp problem
n_eqqp = np.array([10, 50, 100, 500, 1000])
solvers_eqqp = ['osqp',
                'gurobi',
                'mosek',
                #  'ecos',
                #  'qpoases'
                ]
# MPC
# n_mpc = np.array([5, 10, 15, 20])
n_mpc = np.array([5, 10])
solvers_mpc = [
               'osqp',
               'osqp_coldstart',
               'osqp_no_caching',
               'gurobi',
               'mosek',
            #    'qpoases',
            #    'ecos'
               ]

# Portfolio (works for rho = 100. Works also for rho = 10. Other values do not even converge)
n_portfolio = np.array([500, 1000, 5000, 10000, 15000])
solvers_portfolio = [
                     'osqp',
                     'osqp_coldstart',
                     'osqp_no_caching',
                     'gurobi',
                     'mosek',
                     #  'ecos',
                     #  'qpoases'
                     ]
gammas_portfolio = np.logspace(-2, 0, 50)


# Lasso problem
n_lasso = np.array([10, 50, 100, 500, 1000])
lambdas_lasso = np.logspace(-2, 0, 50)
solvers_lasso = [
                 'osqp',
                 'osqp_coldstart',
                 'osqp_no_caching',
                 'gurobi',
                 'mosek',
                 #  'qpoases',   # It fails for big problems!
                  #  'ecos'
                 ]

# Huber problem
n_huber = np.array([10, 50, 100, 500, 1000])
solvers_huber = ['osqp',
                 'gurobi',
                 'mosek',
                 #  'ecos',
                 #  'qpoases'
                 ]


# SVM problem (works around 300 iterations for rho=0.1. It works a bit better for rho=1.0)
n_svm = np.array([10, 50, 100, 500, 1000])
solvers_svm = ['osqp',
               'gurobi',
               'mosek',
               #  'ecos',
               #  'qpoases'
               ]

'''
Solve problems
'''

# Random qp
# randomqp = RandomqpExample(n_randomqp, solvers_randomqp)
# randomqp.run(osqp_settings)


# Eq qp
# eqqp = EqqpExample(n_eqqp, solvers_eqqp)
# eqqp.run(osqp_settings)


# MPC
# Examples: ball, helicopter, pendulum
mpc = MPCExample(n_mpc, solvers_mpc, "helicopter")
mpc.run(osqp_settings)


# Portfolio
# portfolio = PortfolioExample(n_portfolio, solvers_portfolio,
#                              gammas_portfolio)
# portfolio.run(osqp_settings)

# Lasso
# lasso = LassoExample(n_lasso, solvers_lasso, lambdas_lasso)
# lasso.run(osqp_settings)

# Huber
# huber = HuberExample(n_huber, solvers_huber)
# huber.run(osqp_settings)

# SVM
# svm = SVMExample(n_svm, solvers_svm)
# svm.run(osqp_settings)
