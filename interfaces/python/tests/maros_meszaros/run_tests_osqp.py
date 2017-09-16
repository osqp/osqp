from __future__ import print_function

import numpy as np
import numpy.linalg as la
import scipy.sparse as spa
import sys
import os
import time
from multiprocessing import Pool, cpu_count
from itertools import repeat
from utils.utils import load_maros_meszaros_problem

# import osqppurepy as osqp
import osqp


def constrain_scaling(s, min_val, max_val):
    s = np.minimum(np.maximum(s, min_val), max_val)
    return s


def scale_cost(problem):
    """
    Normalize cost by linear part of the cost.
    """

    # Scaling
    scaling = 1.0

    # Get norms
    norm_q = np.linalg.norm(problem.q)
    norm_P = np.linalg.norm(problem.P.todense())
    trP = np.sum(problem.P.diagonal())
    # cost_scal = constrain_scaling(norm_q, 1e-03, 1e03)
    # if norm_q < 1e-06:  # q is null!
    #     cost_scal = 1.
    # else:
    #     cost_scal = norm_q

    # Scaling
    scaling = np.minimum(np.maximum(trP + norm_q, 1e-06), 1e+6)

    # print("scaling = %.2e" % scaling)

    problem.q /= scaling
    problem.P /= scaling


def scale_constraints(problem):
    """
    Scale constraints of the problem
    """
    m_constraints = len(problem.l)
    E = np.zeros(m_constraints)

    # Scaling by max value
    for i in range(m_constraints):
        abs_l_i = np.abs(problem.l[i])
        if np.isinf(abs_l_i) or abs_l_i > 1e15 or abs_l_i < 1e-06:
            abs_l_i = 1.

        abs_u_i = np.abs(problem.u[i])
        if np.isinf(abs_u_i) or abs_u_i > 1e15 or abs_u_i < 1e-06:
            abs_u_i = 1.

        # Scale using maximum bound
        max_abs_bnds = np.maximum(abs_l_i, abs_u_i)
        E[i] = 1./max_abs_bnds

    # Scaling logarithm
    # for i in range(m_constraints):
    #     abs_l = np.abs(problem.l[i])
    #     if np.isinf(abs_l) or abs_l > 1e10 or abs_l < 1e-06:
    #         abs_l = 1.
    #     else:
    #         abs_l = constrain_scaling(abs_l, 1e-03, 1e03)
    #
    #     abs_u = np.abs(problem.u[i])
    #     if np.isinf(abs_u) or abs_u > 1e10 or abs_l < 1e-06:
    #         abs_u = 1.
    #     else:
    #         abs_u = constrain_scaling(abs_u, 1e-03, 1e03)
    #
    #     # # Scale using maximum bound
    #     # max_abs_bnds = np.minimum(abs_l, abs_u)
    #     # E[i] = 1./max_abs_bnds
    #
    #     # Scale using both bounds
    #     # E[i] = 1. / (abs_l * abs_u)
    #
    #     # Exponentially scale bounds
    #     log_l = np.log(abs_l)
    #     log_u = np.log(abs_u)
    #     E[i] = np.exp((log_l + log_u)/2)

    # Select scaling
    E = spa.diags(E)
    # E = spa.diags(np.ones(m_constraints))

    # New constraints
    problem.l = E.dot(problem.l)
    problem.u = E.dot(problem.u)
    problem.A = E.dot(problem.A).tocsc()


def solve_problem(name, settings):
    """
    Solve single problem called name
    """

    problem = load_maros_meszaros_problem(prob_dir + "/" + name)  # Load prob

    # Scale cost
    # scale_cost(problem)

    # Scale constraints
    # scale_constraints(problem)

    # Solve with OSQP
    s = osqp.OSQP()
    s.setup(problem.P, problem.q, problem.A, problem.l, problem.u,
            **settings)
    res = s.solve()

    # Solve with purepy
    # s = osqpurepy.OSQP()
    # s.setup(problem.P, problem.q, problem.A, problem.l, problem.u,
    #         **settings)
    # res = s.solve()

    if res.info.status_val != \
            s.constant('OSQP_OPTIMAL'):
            solved = False
    else:
        solved = True

    print("%s            \t\t%s" % (name, res.info.status))

    return solved, res.info.iter


def select_small_problems(problems):
    """
    Select only problems with less than
        - 1000 variables
        - 1000 constraints
    """

    new_problems = []

    for problem in problems:
        p = load_maros_meszaros_problem(prob_dir + "/" + problem)

        # Get dimensions
        (m, n) = p.A.shape

        if m <= 1000 and n <= 1000:
            new_problems.append(problem)

    return new_problems, len(new_problems)


'''
Main script
'''

# Directory of problems
prob_dir = './mat'
lst_probs = os.listdir(prob_dir)  # List of problems
# Count number of problems
# n_prob = len([name for name in lst_probs
#              if os.path.isfile(prob_dir + "/" + name)])
problems = [f[:-4] for f in lst_probs]
n_prob = len(problems)

# Select small problems
# problems, n_prob = select_small_problems(problems)

# List of interesting probs
# 'QAFIRO' or name == 'CVXQP1_S':
# 'QAFIRO':
# 'CVXQP1_S':
# 'DUALC1':
# 'PRIMAL4':
# 'CVXQP1_M':
# 'AUG2DCQP':
# 'BOYD1':
# 'AUG2D':
# 'AUG2DC':
# 'CONT-101':
# 'CONT-300':
# 'QPCBOEI2':
# 'AUG3D':
# 'QSHIP04S':

# Solve only few problems
# problems = ['QAFIRO', 'CVXQP1_S', 'QSHIP04S', 'PRIMAL4']
# problems = ['CVXQP1_S']

# Problems index
p = 0

# Number unsolved problems
n_unsolved = 0

# OSQP Settings
settings = {'rho': 0.1,
            'verbose': False,
            'scaled_termination': False,
            # 'diagonal_rho': True,
            # 'auto_rho': False,
            # 'update_rho': False,
            # 'line_search': False,
            'max_iter': 2500,
            'scaling_norm': -1,
            'polish': False,
            'scaling': True,
            'early_terminate_interval': 1}

parallel = True  # Execute script in parallel

# Results
results = []

start = time.perf_counter()


'''
Solve all Maros-Meszaros problems
'''

if parallel:
    # Parallel
    pool = Pool(processes=cpu_count())
    results = pool.starmap(solve_problem, zip(problems, repeat(settings)))
else:
    # Serial
    for name in problems:
        # Solve single problem
        results.append(solve_problem(name, settings))

end = time.perf_counter()
elapsed_time = end - start

zipped_results = list(zip(*results))
solved = list(zipped_results[0])
n_iter = list(zipped_results[1])
unsolved = np.invert(solved)

avg_niter = np.mean([x for x in n_iter if x < 2500])

print('Number of solved problems %i/%i' % (n_prob - sum(unsolved),
                                           n_prob))
print("Average number of iterations (solved probs) %.2f" % (avg_niter))
print("Time elapsed %.2e sec" % elapsed_time)
