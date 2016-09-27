# Base QP Solver
import numpy as np
from numpy import linalg as npla
from scipy import linalg as spla
import cvxpy as cvx

import scipy as sp
import matplotlib.pyplot as plt
import ipdb

# Solver Constants
OPTIMAL = "optimal"
UNSOLVED = "optimal_inaccurate"


class quadProgSolution:

    def __init__(self, status, objval, sol):
        self.status = status
        self.objval = objval
        self.sol = sol


def project(xbar, l, u):
    nx = np.size(l)

    # Round x part to [l, u] interval
    xbar[:nx] = np.minimum(np.maximum(xbar[:nx], l), u)

    # Round slack variables to positive ortant
    xbar[nx:] = np.maximum(xbar[nx:], 0)

    return xbar


def SQPSSolve(c, Q, Aeq, beq, Aineq, bineq, lb, ub):

    max_iter = 100
    rho = 1.6

    # Ger dimensions
    (neq, nx) = np.shape(Aeq)
    nineq = np.size(Aineq, 0)
    nvar = nx + nineq  # Number of variables in standard form: x and s variables

    # Form complete (c) matrices for inequality constraints
    Ac = np.vstack([np.hstack([Aeq, np.zeros((neq, nineq))]), np.hstack([Aineq, np.eye(nineq)])])
    bc = np.append(beq, bineq)
    Qc = spla.block_diag(Q, np.zeros((nineq, nineq)))
    cc = np.append(c, np.zeros(nineq))

    # Factorize Matrices (Later)
    M = np.vstack([np.hstack([Qc + rho*np.eye(nvar), Ac.T]), np.hstack([Ac, -1./rho*np.eye(nineq + neq)])])

    print("Splitting QP Solver")
    print("-------------------\n")

    print("iter |\tcost \n")

    # Run ADMM
    z = np.zeros(nvar)
    u = np.zeros(neq + nineq + nvar)  # Number of dual variables in ADMM

    for i in range(max_iter):
        qtemp = -cc + rho*(z + np.dot(Ac.T, bc) - np.dot(Ac.T, u[:neq + nineq]) - u[neq + nineq:])

        qbar = np.append(qtemp, np.zeros(nineq+neq))

        # x update
        x = sp.linalg.solve(M, qbar)
        x = x[:nvar]  # Select only first nvar elements
        # z update
        z = project(x + u[neq+nineq:], lb, ub)
        # u update
        u = u + np.append(np.dot(Ac, x), x) - np.append(np.zeros(neq + nineq), z) - np.append(bc, np.zeros(nvar))

        # Compute cost function
        xtemp = x[:nx]
        f = np.dot(np.dot(xtemp.T, Q), xtemp) + np.dot(c.T, xtemp)

        if (i == 1) | (np.mod(i, 10) == 0):
            print "%.3i | \t%.2f" % (i, f)

    print "Optimization Done\n"

    sol = x[:nx]
    objval = .5*np.dot(np.dot(sol, Q), sol) + np.dot(c, sol)
    return quadProgSolution(OPTIMAL, objval, sol)


def isPSD(A, tol=1e-8):
    E, V = sp.linalg.eigh(A)
    return np.all(E > -tol)


# Define problem and solve it
def main():
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
    bineq = 100*np.abs(sp.randn(nineq))

    l = -3.
    u = 5.

    # Solve QP ADMM
    result = SQPSSolve(c, Q, Aeq, beq, Aineq, bineq, l, u)
    # print result.sol
    print "ADMM Objective Value = %.3f" % result.objval

    # Solve QP with cvxpy
    x = cvx.Variable(nx)
    constraints = [Aeq*x == beq] + [Aineq*x <= bineq] + [x >= l] + [x <= u]
    objective = cvx.Minimize(.5*cvx.quad_form(x, Q) + c.T*x)
    problem = cvx.Problem(objective, constraints)
    results = problem.solve(solver=cvx.GUROBI, verbose=True)
    # print x.value
    print objective.value


if __name__ == '__main__':
    main()
