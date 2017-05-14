#!/usr/bin/env python

# Test QP solvers against Maros Mezaros Benchmark suite
from __future__ import print_function
from builtins import range
import sys
import os   # List directories

import scipy.io as spio
import scipy.sparse as spspa
import numpy as np


import mathprogbasepy as mpbpy

# Generate Plots without using Xserver
import utils.plotting as plotting
import matplotlib.pyplot as plt

from utils.utils import load_maros_meszaros_problem



# Define all solvers
solvers = [mpbpy.GUROBI, mpbpy.MOSEK, mpbpy.OSQP]
# solvers = [mpbpy.MOSEK]
# solvers = [mpbpy.GUROBI]
nsolvers = len(solvers)  # Number of solvers

# Directory of problems
prob_dir = 'mat'
lst_probs = os.listdir(prob_dir)

# Remove wrong problems
lst_probs.remove('CVXQP1_L.mat')  # Not PSD (GUROBI)
lst_probs.remove('CVXQP2_L.mat')  # Not PSD (GUROBI)
lst_probs.remove('CVXQP3_L.mat')  # Not PSD (MOSEK)
lst_probs.remove('VALUES.mat')    # Not PSD (MOSEK)

# Might be wrong
# CVXQP3_M

# Count number of problems
nprob = len([name for name in lst_probs
             if os.path.isfile(os.path.join(prob_dir, name))])
p = 0

# DEBUG: To Remove
# count so that you do not solve all
# nprob = 50

# Preallocate times
t = np.zeros((nprob, nsolvers))
r = np.zeros((nprob, nsolvers))
rM = 100

# Preallocate min times
mint = np.zeros(nprob)

# Solve all Maroz Meszaros problems for all solvers
for f in lst_probs:
    #  if p in range(nprob):  # Solve only first problems
    m = load_maros_meszaros_problem(os.path.join(prob_dir, f))  # Load problem
    print("%3i) %s" % (p, f[:-4]))

    #  if isPSD(m.Q):
    # Solve with all solvers
    for s in range(nsolvers):
        print("\t\t\t- %s: " % solvers[s], end='')

        if solvers[s] == mpbpy.OSQP:
            res = m.solve(solver=solvers[s],
                          verbose=False,
                          scaling_iter=50,
                          max_iter=10000)  # No verbosity
        else:
            res = m.solve(solver=solvers[s], verbose=False)  # No verbosity
        if res.status in mpbpy.quadprog.problem.SOLUTION_PRESENT:  # If solution found
            t[p, s] = res.cputime
            print("OK! Status = %s" % res.status)
        else:
            print("Failed! Status = %s" % res.status)
            t[p, s] = np.inf

    # Get minimum time
    mint[p] = np.amin(t[p, :])

    # compute r values
    for s in range(nsolvers):
        if t[p, s] != np.inf:
            r[p, s] = t[p, s]/mint[p]
        else:
            r[p, s] = rM
    p += 1  # Increment p problem index
#  else:
    #  nprob -= 1  # One problem is not "usable" (Non convex)

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
plt.xscale('log')


#  plt.show(block=False)
plt.savefig('results_maros_meszaros.pdf')
