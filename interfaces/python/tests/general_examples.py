#!/usr/bin/env python

# Test QP solver against Maros Mezaros Benchmark suite
from __future__ import print_function
import sys
import scipy.io as spio
import scipy.sparse as spspa
import scipy as sp
import numpy as np
import ipdb

import mathprogbasepy as mpbpy


def load_maros_meszaros_problem(f):
    # Load file
    m = spio.loadmat(f)

    # Convert matrices
    P = m['Q'].astype(float)
    n = P.shape[0]
    q = m['c'].T.flatten().astype(float)
    A = m['A'].astype(float)
    A = spspa.vstack([A, spspa.eye(n)])
    u = np.append(m['ru'].T.flatten().astype(float),
                  m['ub'].T.flatten().astype(float))
    l = np.append(m['rl'].T.flatten().astype(float),
                  m['lb'].T.flatten().astype(float))
    # Define problem
    p = mpbpy.QuadprogProblem(P, q, A, l, u)

    return p


def main():
    sp.random.seed(6)
    # Possible ops:  {'small1', 'small2', 'random',
    #                 'infeasible', 'random_infeasible',
    #                 'maros_meszaros', 'lp', 'unbounded_lp',
    #                 'unbounded_qp'}
    example = 'random'

    if example == 'maros_meszaros':
        # Maros Meszaros Examples
        # f = 'tests/maros_meszaros/CVXQP2_S.mat'
        # f = 'tests/maros_meszaros/CVXQP1_S.mat'
        # f = 'tests/maros_meszaros/AUG2D.mat'
        f = 'tests/maros_meszaros/CONT-200.mat'
        # f = 'tests/maros_meszaros/PRIMAL3.mat'
        # f = 'tests/maros_meszaros/QBANDM.mat'
        p = load_maros_meszaros_problem(f)

    elif example == 'small1':
        # Our Examples
        # Small Example 1
        P = spspa.csc_matrix(np.array([[4., 1.], [1., 2.]]))
        q = np.ones(2)
        A = spspa.vstack([spspa.csc_matrix(np.ones((1, 2))),
                         spspa.eye(P.shape[0])]).tocsc()
        l = np.array([1.0, 0.0, 0.0])
        u = np.array([1.0, 0.7, 0.7])
        p = mpbpy.QuadprogProblem(P, q, A, l, u)
    elif example == 'small2':
        # Small Example 2
        P = spspa.csc_matrix(np.array([[11., 0.], [0., 0.]]))
        q = np.array([3, 4])
        A = spspa.csc_matrix(np.array([[-1, 0], [0, -1], [-1, -3],
                                      [2, 5], [3, 4]]))
        u = np.array([0., 0., -15, 100, 80])
        l = -np.inf * np.ones(len(u))
        p = mpbpy.QuadprogProblem(P, q, A, l, u)
    elif example == 'infeasible':
        # Infeasible example
        # P = spspa.eye(2)
        P = spspa.csc_matrix((2, 2))
        q = np.ones(2)
        A = spspa.csc_matrix(np.array([[1, 0], [0, 1], [1, 1]]))
        l = np.array([0., 0., -1.])
        u = np.array([np.inf, np.inf, -1.])
        p = mpbpy.QuadprogProblem(P, q, A, l, u)
    elif example == 'random_infeasible':
        # Random Example
        n = 50
        m = 500
        # Generate random Matrices
        Pt = sp.randn(n, n)
        P = spspa.csc_matrix(np.dot(Pt.T, Pt))
        q = sp.randn(n)
        A = spspa.csc_matrix(sp.randn(m, n))
        u = 3 + sp.randn(m)
        # l = u
        l = -3 + sp.randn(m)

        # Make random problem infeasible
        A[int(n/2), :] = A[int(n/2)+1, :]
        l[int(n/2)] = u[int(n/2)+1] + 100 * sp.rand()
        u[int(n/2)] = l[int(n/2)] + 0.5
        # l[int(n/3)] = u[int(n/3)] + 100 * sp.rand()
        # l[int(n/4)] = u[int(n/4)] + 50. * sp.rand()

        p = mpbpy.QuadprogProblem(P, q, A, l, u)
    elif example == 'unbounded_lp':
        # Unbounded example
        P = spspa.csc_matrix((2, 2))
        q = np.array([2, -1])
        A = spspa.eye(2)
        l = np.array([0., 0.])
        u = np.array([np.inf, np.inf])
        p = mpbpy.QuadprogProblem(P, q, A, l, u)
    elif example == 'unbounded_qp':
        # Unbounded example
        P = spspa.csc_matrix(np.diag(np.array([4., 0.])))
        q = np.array([0, 2])
        A = spspa.csc_matrix([[1., 1.], [-1., 1.]])
        l = np.array([-np.inf, -np.inf])
        u = np.array([2., 3.])
        p = mpbpy.QuadprogProblem(P, q, A, l, u)
    elif example == 'random':
        # Random Example
        n = 30
        m = 50
        # Generate random Matrices
        Pt = sp.randn(n, n)
        P = spspa.csc_matrix(np.dot(Pt.T, Pt))
        q = sp.randn(n)
        A = spspa.csc_matrix(sp.randn(m, n))
        u = 3 + sp.randn(m)
        # l = u
        l = -3 + sp.randn(m)

        p = mpbpy.QuadprogProblem(P, q, A, l, u)
    elif example == 'lp':
        # Random Example
        n = 100
        m = 50
        # Generate random Matrices
        P = spspa.csc_matrix(np.zeros((n, n)))
        q = sp.randn(n)
        A = spspa.vstack([spspa.csc_matrix(sp.randn(m, n)), spspa.eye(n)])
        l = np.append(- 3 + sp.randn(m), - 3 + sp.randn(n))
        u = np.append(3 + sp.randn(m), 3 + sp.randn(n))
        p = mpbpy.QuadprogProblem(P, q, A, l, u)
    else:
        assert False, "Unknown example"

    # Solve with CPLEX
    print("\nSolve with CPLEX")
    print("-----------------")
    resultsCPLEX = p.solve(solver=mpbpy.CPLEX, verbose=True)

    # Solve with GUROBI
    print("\nSolve with GUROBI")
    print("-----------------")
    resultsGUROBI = p.solve(solver=mpbpy.GUROBI, OutputFlag=1)

    # Solve with OSQP. You can pass options to OSQP solver
    print("\nSolve with OSQP")
    print("-----------------")
    resultsOSQP = p.solve(solver=mpbpy.OSQP, max_iter=2500,
                          eps_rel=1e-05,
                          eps_abs=1e-5,
                          alpha=1.6,
                          rho=0.1,
                          sigma=0.1,
                          polishing=True)

    if resultsGUROBI.status != 'solver_error':
        print("\n")
        print("Comparison CPLEX - GUROBI")
        print("-------------------------")
        print("Difference in objective value %.8f" % \
            np.linalg.norm(resultsCPLEX.objval - resultsGUROBI.objval))
        print("Norm of solution difference %.8f" % \
            np.linalg.norm(resultsCPLEX.x - resultsGUROBI.x))
        print("Norm of dual difference %.8f" % \
            np.linalg.norm(resultsCPLEX.y - resultsGUROBI.y))

        print("\n")
        print("Comparison OSQP - GUROBI")
        print("-------------------------")
        print("Difference in objective value %.8f" % \
            np.linalg.norm(resultsOSQP.objval - resultsGUROBI.objval))
        print("Norm of solution difference %.8f" % \
            np.linalg.norm(resultsOSQP.x - resultsGUROBI.x))
        print("Norm of dual difference %.8f" % \
            np.linalg.norm(resultsOSQP.y - resultsGUROBI.y))


    ipdb.set_trace()

# Parsing optional command line arguments
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1:])
