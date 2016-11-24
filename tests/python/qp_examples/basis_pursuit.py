#!/usr/bin/env python

import scipy.sparse as spspa
import scipy as sp
import numpy as np
import quadprog.problem as qp
from quadprog.solvers.solvers import GUROBI, CPLEX, OSQP
import quadprog.solvers.osqp.osqp as osqp


# Generate and solve a basis pursuit problem
class basis_pursuit(object):
    """
    the basis purusit problem is defined as
            minimize	|| x ||_1
            subjec to   Ax = b

    Arguments
    ---------
    m, n        - Dimensions of matrix A        <int>
    osqp_opts   - Parameters of OSQP solver
    dens_lvl    - Density level of matrix A     <float>
    """

    def __init__(self, m, n, dens_lvl=1.0, osqp_opts={}):
        # Generate data
        Ad = spspa.random(m, n, density=dens_lvl)
        x_true = np.multiply((np.random.rand(n) > 0.5).astype(float),
                             np.random.randn(n)) / np.sqrt(n)
        bd = Ad.dot(x_true)

        # Construct the problem
        #       minimize	np.ones(n).T * t
        #       subject to  Ax = b
        #                   -t <= x <= t
        In = spspa.eye(n)
        P = spspa.csc_matrix((2*n, 2*n))
        q = np.append(np.zeros(n), np.ones(n))
        A = spspa.vstack([spspa.hstack([Ad, spspa.csc_matrix((m, n))]),
                          spspa.hstack([In, -In]),
                          spspa.hstack([In, In])])
        lA = np.hstack([bd, -np.inf * np.ones(n), np.zeros(n)])
        uA = np.hstack([bd, np.zeros(n), np.inf * np.ones(n)])

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
sp.random.seed(2)

# Set problem
n = 20
m = 200

# Set options of the OSQP solver
options = {'eps_abs':       1e-4,
           'eps_rel':       1e-4,
           'alpha':         1.6,
           'scale_problem': True,
           'scale_steps':   4,
           'polish':        False}

# Create an svm object
basis_pursuit_obj = basis_pursuit(m, n, dens_lvl=0.3, osqp_opts=options)

# Solve with different solvers
resultsCPLEX = basis_pursuit_obj.solve(solver=CPLEX)
resultsGUROBI = basis_pursuit_obj.solve(solver=GUROBI)
resultsOSQP = basis_pursuit_obj.solve(solver=OSQP)


# Print objective values
print "CPLEX  Objective Value: %.3f" % resultsCPLEX.objval
print "GUROBI Objective Value: %.3f" % resultsGUROBI.objval
print "OSQP   Objective Value: %.3f" % resultsOSQP.objval
print "\n"

# Print timings
print "CPLEX  CPU time: %.3f" % resultsCPLEX.cputime
print "GUROBI CPU time: %.3f" % resultsGUROBI.cputime
print "OSQP   CPU time: %.3f" % resultsOSQP.cputime
