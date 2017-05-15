#!/usr/bin/env python

# Test QP solvers against Maros Mezaros Benchmark suite
from __future__ import print_function
from builtins import range
import sys
import os   # List directories

import scipy.io as spio
import scipy.sparse as spspa
import numpy as np
import numpy.linalg as la

import mathprogbasepy as mpbpy

# Generate Plots without using Xserver
import utils.plotting as plotting
import matplotlib.pyplot as plt

from utils.utils import load_maros_meszaros_problem


def is_optimal(x, y, qp, eps_abs=1e-03, eps_rel=1e-03):
    '''
    Check optimality condition of the QP in qp given the
    primal-dual solution (x, y) and the tolerance eps 
    '''

    # Get problem matrices
    P = qp.P
    q = qp.q
    A = qp.A
    l = qp.l
    u = qp.u

    # Check primal feasibility
    Ax = A.dot(x)
    eps_pri = eps_abs + eps_rel * la.norm(Ax, np.inf)
    pri_res = np.minimum(Ax - l, 0) + np.maximum(Ax - u, 0)

    if la.norm(pri_res, np.inf) > eps_pri:
        print("Error in primal residual: %.4e > %.4e" %
              (la.norm(pri_res, np.inf), eps_pri))
        return False

    # Check dual feasibility
    Px = P.dot(x)
    Aty = A.T.dot(y)
    eps_dua = eps_abs + eps_rel * np.max([la.norm(Px, np.inf), 
                                          la.norm(q, np.inf), 
                                          la.norm(Aty, np.inf)])
    dua_res = Px + q + Aty
    
    if la.norm(dua_res, np.inf) > eps_dua:
        print("Error in dual residual: %.4e > %.4e" %
              (la.norm(dua_res, np.inf), eps_dua), end='')
        return False

    # Check complementary slackness
    y_plus = np.maximum(y, 0)
    y_minus = np.minimum(y, 0)
    
    eps_comp = eps_abs + eps_rel * la.norm(Ax, np.inf)

    comp_res_u = np.minimum(np.abs(y_plus), np.abs(Ax - u))
    comp_res_l = np.minimum(np.abs(y_minus), np.abs(Ax - l))

    if la.norm(comp_res_l, np.inf) > eps_comp:
        print("Error in complementary slackness residual l: %.4e > %.4e" %
              (la.norm(comp_res_l, np.inf), eps_dua)), end='')
        return False

    if la.norm(comp_res_u, np.inf) > eps_comp:
        print("Error in complementary slackness residual u: %.4e > %.4e" %
              (la.norm(comp_res_u, np.inf), eps_dua)), end='')
        return False

    # If we arrived until here, the solution is optimal
    return True


'''
Run main script
'''
# Define all solvers
solvers = [mpbpy.GUROBI, mpbpy.MOSEK, mpbpy.OSQP]
# solvers = [mpbpy.MOSEK]
# solvers = [mpbpy.GUROBI]
nsolvers = len(solvers)  # Number of solvers

# Directory of problems
prob_dir = 'mat'
lst_probs = os.listdir(prob_dir)

# Remove wrong problems
#  lst_probs.remove('CVXQP1_L.mat')  # Not PSD (GUROBI)
#  lst_probs.remove('CVXQP2_L.mat')  # Not PSD (GUROBI)
#  lst_probs.remove('CVXQP3_L.mat')  # Not PSD (MOSEK)
#  lst_probs.remove('VALUES.mat')    # Not PSD (MOSEK)

# Might be wrong
# CVXQP3_M

# DEBUG: insert only problem bad for GUROBI
#  lst_probs = ['VALUES.mat']

# Count number of problems
nprob = len([name for name in lst_probs
             if os.path.isfile(os.path.join(prob_dir, name))])
p = 0


# Preallocate times
t = np.zeros((nprob, nsolvers))
r = np.zeros((nprob, nsolvers))
rM = 1e5  # Maximum number of time to solve the problems

# Preallocate min times
mint = np.zeros(nprob)

# Solve all Maroz Meszaros problems for all solvers
for f in lst_probs:
    #  if p in range(nprob):  # Solve only first problems
    m = load_maros_meszaros_problem(os.path.join(prob_dir, f))  # Load problem
    print("%3i) %s" % (p, f[:-4]))

    # Solve with all solvers
    for s in range(nsolvers):
        print("\t\t- %s: " % solvers[s], end='')

        if solvers[s] == mpbpy.OSQP:
            res = m.solve(solver=solvers[s],
                          verbose=False,
                          scaling_iter=50,
                          polish=False,
                          max_iter=10000)  # No verbosity
        else:
            res = m.solve(solver=solvers[s], verbose=False)  # No verbosity
        if res.status in mpbpy.quadprog.problem.SOLUTION_PRESENT:  # If solution found
            # Check if solution is actually optimal
            if is_optimal(res.x, res.y, m):
                t[p, s] = res.cputime
                print("OK! Status = %s" % res.status)
            else:
                print("Failed! Status = %s but optimality conditions not satisfied" % res.status)
                t[p, s] = np.inf
        else:
            print("Failed! Status = %s" % res.status)
            t[p, s] = np.inf

    # Get minimum time
    mint[p] = np.amin(t[p, :])

    # compute r values
    for s in range(nsolvers):
        if t[p, s] != np.inf and not np.isinf(mint[p]):
            r[p, s] = t[p, s]/mint[p]
        else:
            r[p, s] = rM
    p += 1  # Increment p problem index

# Compute rho curve for all solvers
ntau = 1000
tauvec = np.logspace(0, 3, ntau)
rho = np.zeros((nsolvers, ntau))

for s in range(nsolvers):  # Compute curve for all solvers
    for tau in range(ntau):
        count_p = 0
        for p in range(nprob):
            if r[p, s] <= tauvec[tau]:
                count_p += 1  # Sum all solved problems
        rho[s, tau] = 1./nprob*count_p  # Compute rho


# Plot results
ax = plotting.create_figure(0.9)
for s in range(nsolvers):
    plt.plot(tauvec, rho[s, :], label=solvers[s])
plt.legend(loc='best')
plt.ylabel(r'$\rho_{s}$')
plt.xlabel(r'$\tau$')
ax.set_xlim(1., 1e3)
plt.grid()
plt.xscale('log')


#  plt.show(block=False)
plt.savefig('results_maros_meszaros.pdf')
