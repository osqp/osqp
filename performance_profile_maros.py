#!/usr/bin/env python

# Test QP solvers against Maros Mezaros Benchmark suite
import sys
import os   # List directories
import scipy.io as spio
import scipy.sparse as spspa
# import scipy as sp
import numpy as np
import quadprog.problem as qp
from quadprog.solvers.solvers import GUROBI, CPLEX  # OSQP
import matplotlib.pyplot as plt
import seaborn as sns  # Nicer plots
sns.set_style("whitegrid")
# Use Latex Labels in Plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Fix Seaborn Default set_style
sns.set_context("paper", font_scale=1.5)
import ipdb


def load_maros_meszaros_problem(f):
    # Load file
    m = spio.loadmat(f)

    # Convert matrices
    Q = m['Q'].astype(float)
    c = m['c'].T.flatten().astype(float)
    Aeq = m['A'].astype(float)
    beq = m['ru'].T.flatten().astype(float)
    lb = m['lb'].T.flatten().astype(float)
    ub = m['ub'].T.flatten().astype(float)
    nx = Q.shape[0]
    Aineq = spspa.csc_matrix(np.zeros((1, nx)))
    bineq = np.array([0.0])

    # Define problem
    p = qp.quadprogProblem(Q, c, Aeq, beq, Aineq, bineq, lb, ub)

    return p


def main():
    # Define all solvers
    solvers = [GUROBI, CPLEX]
    nsolvers = len(solvers)  # Number of solvers

    # Directory of problems
    prob_dir = './tests/maros_meszaros'
    lst_probs = os.listdir(prob_dir)

    # Count number of problems
    nprob = len([name for name in lst_probs
             if os.path.isfile(name)])
    p = 0

    # DEBUG: To Remove
    # count so that you do not solve all
    nprob = 50

    # Preallocate times
    t = np.zeros((nprob, nsolvers))
    r = np.zeros((nprob, nsolvers))
    rM = 100

    # Preallocate min times
    mint = np.zeros(nprob)

    # Solve all Maroz Meszaros problems for all solvers
    for f in lst_probs:
        if p in range(nprob):  # Solve only first problems
            m = load_maros_meszaros_problem(prob_dir + "/" + f)  # Load problem
            print "Problem %i\n" % p

            # Solve with all solvers
            for s in xrange(nsolvers):
                res = m.solve(solver=solvers[s], verbose=0)  # Suppress verbosity
                if res.status == qp.OPTIMAL:  # If optimal solution found
                    t[p, s] = res.cputime
                else:
                    t[p, s] = np.inf

            # Get minimum time
            mint[p] = np.amin(t[p, :])

            # compute r values
            for s in xrange(nsolvers):
                if t[p, s] != np.inf:
                    r[p, s] = t[p, s]/mint[p]
                else:
                    r[p, s] = rM
        p += 1  # Increment p problem index

    # Compute rho curve for all solvers
    ntau = 100
    tauvec = np.linspace(0, 20, ntau)
    rho = np.zeros((nsolvers, ntau))

    for s in xrange(nsolvers):  # Compute curve for all solvers
        for tau in xrange(ntau):
            count_p = 0
            for p in xrange(nprob):
                if r[p, s] <= tauvec[tau]:
                    count_p += 1  # Sum all solved problems
            rho[s, tau] = 1./nprob*count_p  # Compute rho

    # Plot results
    plt.figure()
    for s in xrange(nsolvers):
        plt.plot(tauvec, rho[s, :], label=solvers[s])
    plt.legend(loc='best')
    plt.ylabel(r'$\rho_{s}$')
    plt.xlabel(r'$\tau$')
    plt.show(block=False)
    plt.savefig('results_maros_meszaros.pdf')


    ipdb.set_trace()




# Parsing optional command line arguments
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1:])
