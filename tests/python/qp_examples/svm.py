#!/usr/bin/env python

import scipy.sparse as spspa
import scipy as sp
import numpy as np
import quadprog.problem as qp
from quadprog.solvers.solvers import GUROBI, CPLEX, OSQP
import quadprog.solvers.osqp.osqp as osqp


# Generate and solve an SVM problem
class svm(object):
    """
    Support vector machine problem is defined as
            minimize	|| x ||^2 + gamma * 1.T * max(0, diag(b) A x + 1)

    Arguments
    ---------
    m, n        - Dimensions of matrix A        <int>
    osqp_opts   - Parameters of OSQP solver
    dens_lvl    - Density level of matrix A     <float>
    """

    def __init__(self, m, n, dens_lvl=1.0, osqp_opts={}):
        # Generate data
        if m % 2 == 1:
            m = m + 1
        N = m / 2
        gamma = 1.0
        b = np.append(np.ones(N), -np.ones(N))
        A_upp = spspa.random(N, n, density=dens_lvl)
        A_low = spspa.random(N, n, density=dens_lvl)
        A = spspa.vstack([
                A_upp / np.sqrt(n) + (A_upp != 0.).astype(float) / n,
                A_low / np.sqrt(n) - (A_low != 0.).astype(float) / n]).tocsc()

        # Construct the problem
        #       minimize	 x.T * x + gamma 1.T * t
        #       subject to  t >= diag(b) A x + 1
        #                   t >= 0

        P = spspa.block_diag((2*spspa.eye(n), spspa.csc_matrix((m, m))),
                             format='csc')
        q = np.append(np.zeros(n), gamma*np.ones(m))
        A = spspa.hstack([spspa.diags(b).dot(A), -spspa.eye(m)]).tocsc()
        uA = -np.ones(m)
        lA = -np.inf * np.ones(m)

        # Add bounds for t
        Abounds = spspa.hstack([spspa.csc_matrix((m, n)), spspa.eye(m)])
        A = spspa.vstack([A, Abounds])
        uA = np.append(uA, +np.inf * np.ones(m))
        lA = np.append(lA, np.zeros(m))

        # Create a quadprogProblem and store it in a private variable
        self._prob = qp.quadprogProblem(P, q, A, lA, uA)
        # Create an OSQP object and store it in a private variable
        self._osqp = osqp.OSQP(**osqp_opts)
        self._osqp.problem(P, q, A, lA, uA)

        # Q = spspa.block_diag((2*spspa.eye(n), spspa.csc_matrix((m, m))),
        #                      format='csc')
        # c = np.append(np.zeros(n), gamma*np.ones(m))
        # Aeq = spspa.csc_matrix((0, n + m))
        # beq = np.zeros(0)
        # Aineq = spspa.hstack([spspa.diags(b).dot(A), -spspa.eye(m)]).tocsc()
        # bineq = -np.ones(m)
        # lb = np.append(-np.inf*np.ones(n), np.zeros(m))
        #
        # # Create a quadprogProblem and store it in a private variable
        # self._prob = qp.quadprogProblem(Q, c, Aeq, beq, Aineq, bineq, lb)
        # # Create an OSQP object and store it in a private variable
        # self._osqp = osqp.OSQP(**osqp_opts)
        # self._osqp.problem(Q, c, Aeq, beq, Aineq, bineq, lb)

    def solve(self, solver=OSQP):
        """
        Solve the problem with a specificed solver.
        """
        if solver == OSQP:
            results = self._osqp.solve()
        elif solver == CPLEX:
            results = self._prob.solve(solver=CPLEX, verbose=0)
        elif solver == GUROBI:
            results = self._prob.solve(solver=GUROBI, OutputFlag=0)
        else:
            assert False, "Unhandled solver"
        return results


# ==================================================================
# ==    Solve a small example
# ==================================================================

# Set the random number generator
sp.random.seed(1)

# Set problem
n = 30
m = 2*n

# Set options of the OSQP solver
options = {'eps_abs':       1e-5,
           'eps_rel':       1e-5,
           'alpha':         1.6,
           'scale_problem': True,
           'scale_steps':   20,
           'polish':        False}

# Create an svm object
svm_obj = svm(m, n, dens_lvl=0.3, osqp_opts=options)

# Solve with different solvers
resultsCPLEX = svm_obj.solve(solver=CPLEX)
resultsGUROBI = svm_obj.solve(solver=GUROBI)
resultsOSQP = svm_obj.solve(solver=OSQP)


# Print objective values
print "CPLEX  Objective Value: %.3f" % resultsCPLEX.objval
print "GUROBI Objective Value: %.3f" % resultsGUROBI.objval
print "OSQP   Objective Value: %.3f" % resultsOSQP.objval
print "\n"

# Print timings
print "CPLEX  CPU time: %.3f" % resultsCPLEX.cputime
print "GUROBI CPU time: %.3f" % resultsGUROBI.cputime
print "OSQP   CPU time: %.3f" % resultsOSQP.cputime
