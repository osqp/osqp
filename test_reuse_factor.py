#!/usr/bin/env python

# Test QP solver when factorization is reused
import sys
import scipy.sparse as spspa
import scipy as sp
import numpy as np
import quadprog.problem as qp
import quadprog.solvers.osqp.osqp as osqp

GUROBI = "GUROBI"


def main():
    example = 'random'  # {'small1', 'small2', 'random', 'maros_meszaros'}

    if example == 'random':
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

    for i in range(2):
        if example == 'random' and i >= 1:
            # Reuse factorizations
            c = sp.randn(nx)
            beq = sp.randn(neq)
            bineq = 100 * sp.rand(nineq)
            p = qp.quadprogProblem(Q, c, Aeq, beq, Aineq, bineq, lb, ub)

        # Solve with GUROBI
        resultsGUROBI = p.solve(solver=GUROBI)

        # Solve with OSQP
        if i == 0:
            # options = {'max_iter': 5000}
            probOSQP = osqp.OSQP()
            probOSQP.problem(Q, c, Aeq, beq, Aineq, bineq, lb, ub)
        else:
            probOSQP.set_problem_data(c=c, beq=beq, bineq=bineq)
        resultsOSQP = probOSQP.solve()

        print "\n"
        print("Comparison OSQP - GUROBI")
        print("-------------------------")
        print "Difference in objective value %.8f" % \
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
