#!/usr/bin/env python

import scipy.sparse as spspa
import scipy as sp
import numpy as np
import quadprog.problem as qp
from quadprog.solvers.solvers import GUROBI, CPLEX, OSQP
import quadprog.solvers.osqp.osqp as osqp


# Generate and solve a Huber fitting problem
class huber_fit(object):
    """
    Huber fitting problem is defined as
            minimize	sum( huber(ai'x - bi) ),

    where huber() is the Huber penalty function defined as
                        | 1/2 x^2       |x| <= 1
            huber(x) = <
                        | |x| - 1/2     |x| > 1

    Arguments
    ---------
    m, n        - Dimensions of matrix A        <int>
    osqp_opts   - Parameters of OSQP solver
    dens_lvl    - Density level of matrix A     <float>
    """

    def __init__(self, m, n, dens_lvl=1.0, osqp_opts={}):
        # Generate data
        A = spspa.random(m, n, density=dens_lvl, format='csc')
        x_true = np.random.randn(n) / np.sqrt(n)
        ind95 = (np.random.rand(m) < 0.95).astype(float)
        b = A.dot(x_true) + np.multiply(0.5*np.random.randn(m), ind95) \
                          + np.multiply(10.*np.random.rand(m), 1. - ind95)

        # Construct the problem
        #       minimize	1/2 u.T * u + np.ones(m).T * v
        #       subject to  -u - v <= Ax - b <= u + v
        #                   0 <= u <= 1
        #                   v >= 0
        Im = spspa.eye(m)
        P = spspa.block_diag((spspa.csc_matrix((n, n)), Im,
                              spspa.csc_matrix((m, m))), format='csc')
        q = np.append(np.zeros(m + n), np.ones(m))
        A = spspa.vstack([
                spspa.hstack([A, Im, Im]),
                spspa.hstack([A, -Im, -Im]),
                spspa.hstack([spspa.csc_matrix((m, n)), Im,
                              spspa.csc_matrix((m, m))]),
                spspa.hstack([spspa.csc_matrix((m, n + m)), Im])]).tocsc()
        lA = np.hstack([b, -np.inf*np.ones(m), np.zeros(2*m)])
        uA = np.hstack([np.inf*np.ones(m), b, np.ones(m), np.inf*np.ones(m)])

        # Create a quadprogProblem and store it in a private variable
        self._prob = qp.quadprogProblem(P, q, A, lA, uA)
        # Create an OSQP object and store it in a private variable
        self._osqp = osqp.OSQP(**osqp_opts)
        self._osqp.problem(P, q, A, lA, uA)

    def solve(self, solver=OSQP):
        """
        Solve the problem with a specificed solver.
        """
        if solver == OSQP:
            results = self._osqp.solve()
        elif solver == CPLEX:
            results = self._prob.solve(solver=CPLEX, verbose=1)
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
n = 20
m = 10*n

# Set options of the OSQP solver
options = {'eps_abs':       1e-4,
           'eps_rel':       1e-4,
           'alpha':         1.6,
           'scale_problem': True,
           'scale_steps':   4,
           'polish':        False}

# Create a lasso object
huber_fit__obj = huber_fit(m, n, dens_lvl=0.50, osqp_opts=options)

# Solve with different solvers
resultsCPLEX = huber_fit__obj.solve(solver=CPLEX)
resultsGUROBI = huber_fit__obj.solve(solver=GUROBI)
resultsOSQP = huber_fit__obj.solve(solver=OSQP)

# Print objective values
print "CPLEX  Objective Value: %.3f" % resultsCPLEX.objval
print "GUROBI Objective Value: %.3f" % resultsGUROBI.objval
print "OSQP   Objective Value: %.3f" % resultsOSQP.objval
print "\n"

# Print timings
print "CPLEX  CPU time: %.3f" % resultsCPLEX.cputime
print "GUROBI CPU time: %.3f" % resultsGUROBI.cputime
print "OSQP   CPU time: %.3f" % resultsOSQP.cputime
