#!/usr/bin/env python

import scipy.sparse as spspa
import scipy as sp
import numpy as np
import quadprog.problem as qp
from quadprog.solvers.solvers import GUROBI, CPLEX, OSQP
import quadprog.solvers.osqp.osqp as osqp


# Generate and solve a (sequence of) Lasso problem(s)
class lasso(object):
    """
    Lasso problem is defined as
            minimize	|| Ax - b ||^2 + gamma * || x ||_1

    Arguments
    ---------
    m, n        - Dimensions of matrix A        <int>
    osqp_opts   - Parameters of OSQP solver
    dens_lvl    - Density level of matrix A     <float>
    version     - QP reformulation              ['dense', 'sparse']
    """

    def __init__(self, m, n, dens_lvl=1.0, inst=1, version='dense',
                 osqp_opts={}):
        # Generate data
        A = spspa.random(m, n, density=dens_lvl, format='csc')
        x_true = np.multiply((np.random.rand(n) > 0.5).astype(float),
                             np.random.randn(n)) / np.sqrt(n)
        b = A.dot(x_true) + .5*np.random.randn(m)
        gamma_max = 0.2 * np.linalg.norm(A.T.dot(b), np.inf)
        self._gammas = np.exp(np.linspace(np.log(gamma_max),
                              np.log(gamma_max * 1e-2), inst))
        self._iter_cnt = 0

        # Construct the problem
        if version == 'dense':
            #       minimize	|| Ax - b ||^2 + gamma * np.ones(b).T * t
            #       subject to  -t <= x <= t
            Q = spspa.block_diag((2*A.T.dot(A), spspa.csc_matrix((n, n))),
                                 format='csc')
            c = np.append(A.T.dot(b), self._gammas[0]*np.ones(n))
            Aeq = spspa.csc_matrix((0, 2*n))
            beq = np.zeros(0)
            In = spspa.eye(n)
            Aineq = spspa.vstack([spspa.hstack([In, -In]),
                                  spspa.hstack([-In, -In])]).tocsc()
            bineq = np.zeros(2*n)
        elif version == 'sparse':
            #       minimize	y.T * y + gamma * np.ones(b).T * t
            #       subject to  y = Ax
            #                   -t <= x <= t
            Q = spspa.block_diag((spspa.csc_matrix((n, n)), 2*spspa.eye(m),
                                  spspa.csc_matrix((n, n))), format='csc')
            c = np.append(np.zeros(m + n), self._gammas[0]*np.ones(n))
            Aeq = spspa.hstack([A, -spspa.eye(m), spspa.csc_matrix((m, n))])
            beq = b
            In = spspa.eye(n)
            Onm = spspa.csc_matrix((n, m))
            Aineq = spspa.vstack([spspa.hstack([In, Onm, -In]),
                                  spspa.hstack([-In, Onm, -In])]).tocsc()
            bineq = np.zeros(2*n)
        else:
            assert False, "Unhandled version"

        # Create a quadprogProblem and store it in a private variable
        self._prob = qp.quadprogProblem(Q, c, Aeq, beq, Aineq, bineq)
        # Create an OSQP object and store in a private variable
        self._osqp = osqp.OSQP(**osqp_opts)
        self._osqp.problem(Q, c, Aeq, beq, Aineq, bineq)

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

    def update_gamma(self):
        """
        Update parameter gamma in the problem and in
        Return:
        0   if the update is successful
        1   if iter_cnt  > len(gammas)-1
        """
        self._iter_cnt = self._iter_cnt + 1
        try:
            gamma = self._gammas[self._iter_cnt]
        except IndexError:
            return 1
        # Update current c in the problem
        c = self._prob.c
        n = self._osqp.problem.nineq / 2
        c[-n:] = gamma*np.ones(n)
        self._prob.c = c
        # Update gmma in the OSQP object
        self._osqp.set_problem_data(c=c)
        # Update successfull
        return 0


# ==================================================================
# ==    Solve a small example
# ==================================================================

# Set the random number generator
sp.random.seed(1)

# Set problem
m = 20
n = 10*m
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
lasso_obj = lasso(m, n, inst=numofinst, version='sparse', osqp_opts=options)
for i in range(numofinst):
    # Solve with different solvers
    resultsCPLEX = lasso_obj.solve(solver=CPLEX)
    resultsGUROBI = lasso_obj.solve(solver=GUROBI)
    resultsOSQP = lasso_obj.solve(solver=OSQP)
    # Print timings
    print "CPLEX  CPU time: %.3f" % resultsCPLEX.cputime
    print "GUROBI CPU time: %.3f" % resultsGUROBI.cputime
    print "OSQP   CPU time: %.3f\n" % resultsOSQP.cputime
    if numofinst > 1:
        lasso_obj.update_gamma()
