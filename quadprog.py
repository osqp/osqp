#!/usr/bin/env python

import sys
import numpy as np
from numpy import linalg as npla
import scipy as sp
import cvxpy as cvx
import ipdb		# ipdb.set_trace()
import osqp

reload(osqp)


def main():
    # Select an example QP
    expl = 'random'  # {'small', 'random'}
    if expl == 'random':
        nx = 50
        neq = 10
        nineq = 20
        # Generate random Matrices
        Qt = sp.randn(nx, nx)
        Q = np.dot(Qt.T, Qt)
        c = sp.randn(nx)
        Aeq = sp.randn(neq, nx)
        beq = sp.randn(neq)
        Aineq = sp.randn(nineq, nx)
        bineq = 100 * sp.rand(nineq)
        lb = 0. * np.ones(nx)
        ub = 5. * np.ones(nx)
    elif expl == 'small':
        nx = 2
        neq = 1
        nineq = 0
        # Generate a small example
        Q = np.array([[4., 1.], [1., 2.]])
        c = np.ones(nx)
        Aeq = np.ones((1, nx))
        beq = np.array([1.0])
        Aineq = np.zeros((0, nx))
        bineq = np.zeros(0)
        lb = np.zeros(nx)
        ub = 0.7 * np.ones(nx)
    else:
        assert False, "Unknown example"

    # todo: Generate QPs coming from MPC, finance, SVM, etc.

    # Solve QP via ADMM
    solADMM = osqp.OSQP(Q, c, Aeq, beq, Aineq, bineq, lb, ub,
                        max_iter=5000, print_iter=100)

    # Solve QP via cvxpy+ECOS
    x = cvx.Variable(nx)
    dict_constr = {}  # dict_constr has keys of all given constraints
    constraints = []  # list of constraints
    if beq.size:  # if beq is nonempty
        constraints = constraints + [Aeq * x == beq]
        # value assigned to each key is an index of the constraint
        dict_constr['eq'] = len(constraints) - 1
    if bineq.size:
        constraints = constraints + [Aineq * x <= bineq]
        dict_constr['ineq'] = len(constraints) - 1
    if lb.size:
        constraints = constraints + [x >= lb]
        dict_constr['lb'] = len(constraints) - 1
    if ub.size:
        constraints = constraints + [x <= ub]
        dict_constr['ub'] = len(constraints) - 1
    objective = cvx.Minimize(.5 * cvx.quad_form(x, Q) + c.T * x)
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=cvx.CVXOPT, verbose=False)

    # Get dual variables
    dual_eq = np.asarray(
        constraints[dict_constr['eq']].dual_value).flatten() \
        if 'eq' in dict_constr.keys() else np.zeros(0)
    dual_ineq = np.asarray(
        constraints[dict_constr['ineq']].dual_value).flatten() \
        if 'ineq' in dict_constr.keys() else np.zeros(0)
    dual_lb = np.asarray(
        constraints[dict_constr['lb']].dual_value).flatten() \
        if 'lb' in dict_constr.keys() else np.zeros(0)
    dual_ub = np.asarray(
        constraints[dict_constr['ub']].dual_value).flatten() \
        if 'ub' in dict_constr.keys() else np.zeros(0)

    # Compare solutions
    print "ADMM Objective Value = %.3f" % solADMM.objval
    print "ECOS Objective Value = %.3f" % objective.value

    # Compare dual variables of ADMM and ECOS solutions
    if 'eq' in dict_constr.keys():
        print "dual_eq DIFF   = %.4f" % \
            npla.norm(dual_eq - solADMM.sol_dual_eq)
    if 'ineq' in dict_constr.keys():
        print "dual_ineq DIFF = %.4f" % \
            npla.norm(dual_ineq - solADMM.sol_dual_ineq)
    if 'lb' in dict_constr.keys():
        print "dual_lb DIFF   = %.4f" % \
            npla.norm(dual_lb - solADMM.sol_dual_lb)
    if 'ub' in dict_constr.keys():
        print "dual_ub DIFF   = %.4f" % \
            npla.norm(dual_ub - solADMM.sol_dual_ub)
    
    ipdb.set_trace()

# Parsing optional command line arguments
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1:])
