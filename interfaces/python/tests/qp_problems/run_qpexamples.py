from __future__ import print_function
from mathprogbasepy import OSQP
import numpy as np

# Run QP tests from different examples
from tests.qp_problems.qp_examples.basis_pursuit import basis_pursuit
from tests.qp_problems.qp_examples.huber_fit import huber_fit
from tests.qp_problems.qp_examples.lasso import lasso
from tests.qp_problems.qp_examples.lp import lp
from tests.qp_problems.qp_examples.nonneg_l2 import nonneg_l2
from tests.qp_problems.qp_examples.portfolio import portfolio
from tests.qp_problems.qp_examples.svm import svm

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
dim_vecs_len = 10
n_max = 100
m_max = 100
n_vec = np.arange(10, n_max, int(n_max/dim_vecs_len))
m_vec = np.arange(10, m_max, int(m_max/dim_vecs_len))

# Test options
options = {'solver': OSQP,
           'verbose': False,
           'polishing': False,
           'max_iter': 2500}

# Test types
test_types = ['basis_pursuit', 'huber_fit', 'lasso', 'nonneg_l2', 'lp',
              'portfolio', 'svm']


def run_examples(test_type, n_vec, m_vec, rho_vec, sigma_vec,
                 alpha_vec, **kwargs):
    if test_type == 'basis_pursuit':
        return basis_pursuit.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                                       alpha_vec, **kwargs)
    elif test_type == 'huber_fit':
        return huber_fit.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                                   alpha_vec, **kwargs)
    elif test_type == 'lasso':
        return lasso.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                               alpha_vec, **kwargs)
    elif test_type == 'nonneg_l2':
        return nonneg_l2.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                                   alpha_vec, **kwargs)
    elif test_type == 'lp':
        return lp.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                            alpha_vec, **kwargs)
    elif test_type == 'portfolio':
        return portfolio.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                                   alpha_vec, **kwargs)
    elif test_type == 'svm':
        return svm.run_tests(n_vec, m_vec, rho_vec, sigma_vec,
                             alpha_vec, **kwargs)


# PARALLEL IMPLEMENTATION
# ------------------------------------------------------------------------------

partial_tests = partial(run_examples, n_vec=n_vec,
                        m_vec=m_vec, rho_vec=rho_vec, sigma_vec=sigma_vec,
                        alpha_vec=alpha_vec, **options)

t = time()

# Execute problems in parallel
p = Pool(cpu_count())
results = p.map(partial_tests, test_types)


# Execute problems in series
# results = []
# for i in range(len(test_types)):
#     results.append(partial_tests(test_types[i]))

cputime = time() - t
print("total cputime = %.4f sec" % cputime)

# Concatenate dataframes
results = pd.concat(results)

# Export results
results.to_csv('tests/qp_problems/results/full_results.csv', index=False)
