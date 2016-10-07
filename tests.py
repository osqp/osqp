#!/usr/bin/env python

# Test QP solver against Maros Mezaros Benchmark suite
import sys
import scipy.io as spio
import scipy.sparse as spspa
import scipy as sp
import numpy as np
import ipdb
import quadprog.problem as qp
from quadprog.solvers.solvers import *

reload(qp)

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
    example = 'random'  # {'small', 'random', 'maros_meszaros'}

    if example == 'maros_meszaros':
        # Maros Meszaros Examples
        # f = 'tests/maros_meszaros/CVXQP2_S.mat'
        # f = 'tests/maros_meszaros/PRIMAL3.mat'
        f = 'tests/maros_meszaros/QBANDM.mat'
        p = load_maros_meszaros_problem(f)

    elif example == 'small':
        # Our Examples
        # Small Example
        Q = spspa.csc_matrix(np.array([[4., 1.], [1., 2.]]))
        c = np.ones(2)
        Aeq = spspa.csc_matrix(np.ones((1, 2)))
        beq = np.array([1.0])
        Aineq = spspa.csc_matrix(np.zeros((0, 2)))
        bineq = np.zeros(0)
        lb = 0.6*np.zeros(2)
        ub = 0.7 * np.ones(2)
        p = qp.quadprogProblem(Q, c, Aeq, beq, Aineq, bineq)
    elif example == 'random':
        # Random Example
        nx = 50
        neq = 10
        nineq = 20
        # Generate random Matrices
        Qt = sp.randn(nx, nx)
        Q = spspa.csc_matrix(np.dot(Qt.T, Qt))
        c = sp.randn(nx)
        Aeq = spspa.csc_matrix(sp.randn(neq, nx))
        beq = sp.randn(neq)
        Aineq = spspa.csc_matrix(sp.randn(nineq, nx))
        bineq = 100 * sp.rand(nineq)
        lb = 0. * np.ones(nx)
        ub = 5. * np.ones(nx)
        p = qp.quadprogProblem(Q, c, Aeq, beq, Aineq, bineq, lb, ub)
    else:
        assert False, "Unknown example"

    # Solve with CPLEX
    resultsCPLEX = p.solve(solver=CPLEX)

    # Solve with GUROBI
    resultsGUROBI = p.solve(solver=GUROBI)

    # Solve with OSQP. You can pass options to OSQP solver
    resultsOSQP = p.solve(solver=OSQP, max_iter=5000)

    print "\n"
    print("Comparison CPLEX - GUROBI")
    print("-------------------------")
    print "Norm of objective value difference %.8f" % \
        np.linalg.norm(resultsCPLEX.objval - resultsGUROBI.objval)
    print "Norm of solution difference %.8f" % \
        np.linalg.norm(resultsCPLEX.x - resultsGUROBI.x)
    print "Norm of dual eq difference %.8f" % \
        np.linalg.norm(resultsCPLEX.sol_dual_eq - resultsGUROBI.sol_dual_eq)
    print "Norm of dual ineq difference %.8f" % \
        np.linalg.norm(resultsCPLEX.sol_dual_ineq - resultsGUROBI.sol_dual_ineq)
    print "Norm of dual ub difference %.8f" % \
        np.linalg.norm(resultsCPLEX.sol_dual_ub - resultsGUROBI.sol_dual_ub)
    print "Norm of dual lb difference %.8f" % \
        np.linalg.norm(resultsCPLEX.sol_dual_lb - resultsGUROBI.sol_dual_lb)

    print "\n"
    print("Comparison OSQP - GUROBI")
    print("-------------------------")
    print "Norm of objective value difference %.8f" % \
        np.linalg.norm(resultsOSQP.objval - resultsGUROBI.objval)
    print "Norm of solution difference %.8f" % \
        np.linalg.norm(resultsOSQP.x - resultsGUROBI.x)
    print "Norm of dual eq difference %.8f" % \
        np.linalg.norm(resultsOSQP.sol_dual_eq - resultsGUROBI.sol_dual_eq)
    print "Norm of dual ineq difference %.8f" % \
        np.linalg.norm(resultsOSQP.sol_dual_ineq - resultsGUROBI.sol_dual_ineq)
    print "Norm of dual ub difference %.8f" % \
        np.linalg.norm(resultsOSQP.sol_dual_ub - resultsGUROBI.sol_dual_ub)
    print "Norm of dual lb difference %.8f" % \
        np.linalg.norm(resultsOSQP.sol_dual_lb - resultsGUROBI.sol_dual_lb)

    # ipdb.set_trace()

# Parsing optional command line arguments
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1:])
