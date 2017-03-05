from __future__ import print_function
from mathprogbasepy import OSQP
import numpy as np

# Run QP tests from different examples
from qp_examples.basis_pursuit import basis_pursuit
from qp_examples.huber_fit import huber_fit
from qp_examples.lasso import lasso
from qp_examples.lp import lp
from qp_examples.nonneg_l2 import nonneg_l2
from qp_examples.portfolio import portfolio
from qp_examples.svm import svm

# from multiprocessing import Process, Queue
from multiprocessing import Pool, cpu_count
from functools import partial
# from joblib import Parallel, delayed, cpu_count


import pandas as pd
# pd.set_option('display.width', 1000)  # See all the columns
# import matplotlib.pyplot as plt
# import seaborn as sns
# ipdb.set_trace()

from time import time
# import ipdb

# Define tests ranges
rho_vec_len = 10  # Define rho vector
rho_vec = np.logspace(-4., 3., rho_vec_len)
# rho_vec = np.array([1000])


sigma_vec_len = 10  # Define sigma vector
sigma_vec = np.logspace(-4., 3., sigma_vec_len)

alpha_vec_len = 10  # Define alpha vector
alpha_vec = np.linspace(0.1, 1.9, alpha_vec_len)
# alpha_vec = np.array([1.6])



dim_vecs_len = 1
#  n_max = 50
#  m_max = 50
#  n_vec = np.arange(10, n_max, int(n_max/dim_vecs_len))
#  m_vec = np.arange(10, m_max, int(m_max/dim_vecs_len))
n_vec = np.array([20])
m_vec = np.array([30])


# Number of problems with the same dimensions
nm_num_prob = 100

# Test options
options = {'solver': OSQP,
           'verbose': False,
           'polish': False,
           'scaling_norm': 2,
           'scaling_iter': 3,
           'max_iter': 2500}

# Test types
# test_types = ['basis_pursuit', 'huber_fit', 'lasso', 'nonneg_l2', 'lp',
            #   'portfolio', 'svm']

test_types = ['svm', 'lasso', 'portfolio']

def run_examples(test_type, n_vec, m_vec, rho_vec, sigma_vec,
                 alpha_vec, nm_num_prob, **kwargs):
    if test_type == 'basis_pursuit':
        return basis_pursuit.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                                       alpha_vec, nm_num_prob, **kwargs)
    elif test_type == 'huber_fit':
        return huber_fit.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                                   alpha_vec, nm_num_prob, **kwargs)
    elif test_type == 'lasso':
        return lasso.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                               alpha_vec, nm_num_prob, **kwargs)
    elif test_type == 'nonneg_l2':
        return nonneg_l2.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                                   alpha_vec, nm_num_prob, **kwargs)
    elif test_type == 'lp':
        return lp.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                            alpha_vec, nm_num_prob, **kwargs)
    elif test_type == 'portfolio':
        return portfolio.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                                   alpha_vec, nm_num_prob, **kwargs)
    elif test_type == 'svm':
        return svm.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                             alpha_vec, nm_num_prob, **kwargs)


# PARALLEL IMPLEMENTATION
# ------------------------------------------------------------------------------

partial_tests = partial(run_examples, n_vec=n_vec,
                        m_vec=m_vec, rho_vec=rho_vec, sigma_vec=sigma_vec,
                        alpha_vec=alpha_vec, nm_num_prob=nm_num_prob, **options)

t = time()

# Execute problems in parallel
p = Pool(cpu_count())
res = p.map(partial_tests, test_types)
results = [x[0] for x in res]
results_full = [x[1] for x in res]


# # Execute problems in series
# results = []
# results_full = []
# for i in range(len(test_types)):
#     res, res_full = partial_tests(test_types[i])
#     results.append(res)
#     results_full.append(res_full)

cputime = time() - t
print("total cputime = %.4f sec" % cputime)

# Concatenate dataframes
results = pd.concat(results)
results_full = pd.concat(results_full)


# Export results
results.to_csv('results/results.csv', index=False)
results_full.to_csv('results/results_full.csv', index=False)
