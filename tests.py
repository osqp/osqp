#!/usr/bin/env python

# Test QP solver against Maros Mezaros Benchmark suite
import sys
import scipy.io as spio
import scipy.sparse as spspa
import numpy as np
import ipdb
import quadprog.problem as qp
from quadprog.solvers.solvers import *

import osqp
reload(osqp)


def main():
    # for file in os.listdir('tests/maros_meszaros'):
    # Do all the tests
    p = spio.loadmat('tests/maros_meszaros/CVXQP1_S.mat')
    # p = spio.loadmat('tests/maros_meszaros/AUG2D.mat')
    Q = p['Q'].astype(float)  # Convert to dense matrix (To remove)
    c = p['c'].T.flatten().astype(float)
    Aeq = p['A'].astype(float)  # Convert to dense matrix (To remove)
    beq = p['ru'].T.flatten().astype(float)
    lb = p['lb'].T.flatten().astype(float)
    ub = p['ub'].T.flatten().astype(float)
    nx = Q.shape[0]
    Aineq = spspa.csc_matrix(np.zeros((1, nx)))
    bineq = np.array([0.0])

    # Define problem
    p = qp.quadprogProblem(Q, c, Aeq, beq, Aineq, bineq, lb, ub)

    # Solve with CPLEX
    resultsCPLEX = p.solve(solver=CPLEX)

    # Solve with GUROBI
    resultsGUROBI = p.solve(solver=CPLEX)

    print "\n"
    print "Norm of objective value difference %.4f" % \
        np.linalg.norm(resultsCPLEX.objval - resultsGUROBI.objval)
    print "Norm of solution difference %.4f" % \
        np.linalg.norm(resultsCPLEX.x - resultsGUROBI.x)
    print "Norm of dual eq difference %.4f" % \
        np.linalg.norm(resultsCPLEX.sol_dual_eq - resultsGUROBI.sol_dual_eq)
    print "Norm of dual ineq difference %.4f" % \
        np.linalg.norm(resultsCPLEX.sol_dual_ineq - resultsGUROBI.sol_dual_ineq)
    print "Norm of dual ub difference %.4f" % \
        np.linalg.norm(resultsCPLEX.sol_dual_ub - resultsGUROBI.sol_dual_ub)
    print "Norm of dual lb difference %.4f" % \
        np.linalg.norm(resultsCPLEX.sol_dual_lb - resultsGUROBI.sol_dual_lb)


# Parsing optional command line arguments
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1:])
