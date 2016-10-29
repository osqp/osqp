#!/usr/bin/env python

import scipy.sparse as spspa
import scipy as sp
import numpy as np
import quadprog.problem as qp
from quadprog.solvers.solvers import GUROBI, CPLEX, OSQP
import quadprog.solvers.osqp.osqp as osqp


# Generate and solve a Portfolio optimization problem
class nonneg_l2(object):
    """
    Nonnegative least-squares problem is defined as
            minimize	|| Ax - b ||^2
            subjec to   x >= 0

    Arguments
    ---------
    m, n        - Dimensions of matrix A        <int>
    osqp_opts   - Parameters of OSQP solver
    dens_lvl    - Density level of matrix A     <float>
    version     - QP reformulation              ['dense', 'sparse']
    """

    def __init__(self, m, n, dens_lvl=1.0, version='dense',
                 osqp_opts={}):
        # Generate data
        A = spspa.random(m, n, density=dens_lvl, format='csc')
        x_true = np.ones(n) / n + np.random.randn(n) / np.sqrt(n)
        b = A.dot(x_true) + 0.5*np.random.randn(m)

        # Construct the problem
        if version == 'dense':
            #       minimize	1/2 x.T (A.T * A) x - (A.T * b).T * x
            #       subject to  x >= 0
            Q = A.T.dot(A)
            c = -A.T.dot(b)
            Aeq = spspa.csc_matrix((0, n))
            beq = np.zeros(0)
            Aineq = spspa.csc_matrix((0, n))
            bineq = np.zeros(0)
            lb = np.zeros(n)
        elif version == 'sparse':
            #       minimize	1/2 y.T*y
            #       subject to  y = Ax - b
            #                   x >= 0
            Im = spspa.eye(m)
            Q = spspa.block_diag((spspa.csc_matrix((n, n)), Im), format='csc')
            c = np.zeros(n + m)
            Aeq = spspa.hstack([A, -Im]).tocsr()
            beq = b
            Aineq = spspa.csc_matrix((0, n + m))
            bineq = np.zeros(0)
            lb = np.append(np.zeros(n), -np.inf*np.ones(m))
        else:
            assert False, "Unhandled version"

        # Create a quadprogProblem and store it in a private variable
        self._prob = qp.quadprogProblem(Q, c, Aeq, beq, Aineq, bineq, lb)
        # Create an OSQP object and store it in a private variable
        self._osqp = osqp.OSQP(**osqp_opts)
        self._osqp.problem(Q, c, Aeq, beq, Aineq, bineq, lb)

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
m = 20
n = 5*m
numofinst = 5

# Set options of the OSQP solver
options = {'eps_abs':       1e-4,
           'eps_rel':       1e-4,
           'alpha':         1.6,
           'scale_problem': True,
           'scale_steps':   4,
           'polish':        False,
           'warm_start':    True}

# Create a lasso object
nonneg_l2__obj = nonneg_l2(m, n, dens_lvl=0.3, version='dense',
                          osqp_opts=options)

# Solve with different solvers
resultsCPLEX = nonneg_l2__obj.solve(solver=CPLEX)
resultsGUROBI = nonneg_l2__obj.solve(solver=GUROBI)
resultsOSQP = nonneg_l2__obj.solve(solver=OSQP)

# Print timings
print "CPLEX  CPU time: %.3f" % resultsCPLEX.cputime
print "GUROBI CPU time: %.3f" % resultsGUROBI.cputime
print "OSQP   CPU time: %.3f" % resultsOSQP.cputime
