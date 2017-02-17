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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns  # Nicer plots
import ipdb

sns.set_style("whitegrid")
# Use Latex Labels in Plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Fix Seaborn Default set_style
sns.set_context("paper", font_scale=1.5)


def load_maros_meszaros_problem(f):
    # Load file
    m = spio.loadmat(f)

    # Convert matrices
    P = m['Q'].astype(float)
    n = P.shape[0]
    q = m['c'].T.flatten().astype(float)
    A = m['A'].astype(float)
    A = spspa.vstack([A, spspa.eye(n)])
    u = np.append(m['ru'].T.flatten().astype(float),
                  m['ub'].T.flatten().astype(float))
    l = np.append(m['rl'].T.flatten().astype(float),
                  m['lb'].T.flatten().astype(float))
    # Define problem
    p = mpbpy.QuadprogProblem(P, q, A, l, u)

    return p


def main():
    # Define all solvers
    solvers = [mpbpy.GUROBI, mpbpy.CPLEX, mpbpy.OSQP]
    nsolvers = len(solvers)  # Number of solvers

    # Directory of problems
    prob_dir = './tests/maros_meszaros'
    lst_probs = os.listdir(prob_dir)

    # Count number of problems
    nprob = len([name for name in lst_probs
                 if os.path.isfile(prob_dir + "/" + name)])
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
        m = load_maros_meszaros_problem(prob_dir + "/" + f)  # Load problem
        print("Problem %i: %s \n" % (p, f[:-4]))

        #  if isPSD(m.Q):
        # Solve with all solvers
        for s in range(nsolvers):
            print("  - Solving with %s: " % solvers[s])
            res = m.solve(solver=solvers[s], verbose=False)  # No verbosity
            if res.status == mpbpy.OPTIMAL:  # If optimal solution found
                t[p, s] = res.cputime
            else:
                print("Failed!\n")
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
    ntau = 100
    tauvec = np.linspace(0, 10, ntau)
    rho = np.zeros((nsolvers, ntau))

    for s in range(nsolvers):  # Compute curve for all solvers
        for tau in range(ntau):
            count_p = 0
            for p in range(nprob):
                if r[p, s] <= tauvec[tau]:
                    count_p += 1  # Sum all solved problems
            rho[s, tau] = 1./nprob*count_p  # Compute rho

    # Plot results
    plt.figure()
    for s in range(nsolvers):
        plt.plot(tauvec, rho[s, :], label=solvers[s])
    plt.legend(loc='best')
    plt.ylabel(r'$\rho_{s}$')
    plt.xlabel(r'$\tau$')
    ipdb.set_trace()
    #  plt.show(block=False)
    plt.savefig('results_maros_meszaros.pdf')


# Parsing optional command line arguments
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1:])
