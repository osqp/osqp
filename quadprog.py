# Base QP Solver
import numpy as np
from numpy import linalg as npla
from scipy import linalg as spla

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


def OSqpSolve(c, Q, Aeq, beq, Aineq, bineq, lb, ub):

    max_iter = 100
    rho = 1.6

    # Ger dimensions
    (neq, nx) = np.shape(Aeq)
    nineq = np.size(Aineq, 0)
    nvar = nx + nineq  # Number of variables in standard form: x and s variables

    # Form complete (c) matrices for inequality constraints
    Ac = np.asarray(np.bmat([[Aeq, np.zeros((neq, nineq))], [Aineq, np.eye(nineq)]]))
    bc = np.append(beq, bineq)
    Qc = spla.block_diag(Q, np.zeros((nineq, nineq)))
    cc = np.append(c, np.zeros(nineq))
    # Factorize Matrices (Later)
    M = np.asarray(np.bmat([[Qc + rho*np.eye(nvar), Ac.T], [Ac, -1./rho*np.eye(nineq + neq)]]))

    # Run ADMM
    z = np.zeros(nvar)
    u = np.zeros(neq + nineq + nvar)  # Number of dual variables in ADMM

    for i in range(max_iter):
        qtemp = -cc + rho*(z + np.dot(Ac.T, bc) - np.dot(Ac.T, u[:neq + nineq]) - u[neq + nineq:])

        qbar = np.append(qtemp, np.zeros(nineq+neq))

        # x update
        x = sp.linalg.solve(M, qbar)
        x = x[:nvar]
        # z update
        z = project(x + u[neq+nineq:], lb, ub)
        # u update
        u = u + np.append(np.dot(Ac, x), x) - np.append(np.zeros(neq + nineq), z) - np.append(bc, np.zeros(nvar))

    sol = x[:nx]
    objval = .5*np.dot(np.dot(sol, Q), sol) + np.dot(c, sol)
    return quadProgSolution(OPTIMAL, objval, sol)


# Define problem and solve it
def main():
    nx = 50
    neq = 10
    nineq = 20

    # Generate random Matrices
    Q = npla.matrix_power(sp.randn(nx, nx), 2)
    c = sp.randn(nx)
    Aeq = sp.randn(neq, nx)
    beq = sp.randn(neq)
    Aineq = sp.randn(nineq, nx)
    bineq = 100*np.abs(sp.randn(nineq))

    l = -3.
    u = 5.

    # Solve QP ADMM
    result = OSqpSolve(c, Q, Aeq, beq, Aineq, bineq, l, u)
    print result.sol
    print result.objval


if __name__ == '__main__':
    main()
