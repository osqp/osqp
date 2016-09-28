import numpy as np
from scipy import linalg as spla
import scipy as sp
import ipdb		# ipdb.set_trace()

# Solver Constants
OPTIMAL = "optimal"
UNSOLVED = "optimal_inaccurate"
INFEASIBLE = "infeasible"
UNBOUNDED = "unbounded"


class quadProgSolution:
    """
    stores QP solution
    """

    def __init__(self, status, objval, sol_prim, sol_dual_eq,
                 sol_dual_ineq, sol_dual_lb, sol_dual_ub):
        self.status = status
        self.objval = objval
        self.sol_prim = sol_prim
        self.sol_dual_eq = sol_dual_eq
        self.sol_dual_ineq = sol_dual_ineq
        self.sol_dual_lb = sol_dual_lb
        self.sol_dual_ub = sol_dual_ub


def project(xbar, lb, ub):
    nx = np.size(lb)

    # Round x part to [l, u] interval
    xbar[:nx] = np.minimum(np.maximum(xbar[:nx], lb), ub)

    # Round slack variables to positive ortant
    xbar[nx:] = np.maximum(xbar[nx:], 0)

    return xbar


# Base QP Solver
def OSQP(Q, c, Aeq, beq, Aineq, bineq, lb, ub, x0=0,
         max_iter=500, rho=1.6, printiter=25, scaling=False):
    """
    Operator splitting solver for a QP problem given
    in the following form
            minimize	1/2 x' Q x + c'x
            subject to	Aeq x == beq
                        Aineq x <= bineq
                        lb <= x <= ub
    """

    # Get dimensions
    nx = c.shape[0]
    neq = beq.shape[0]
    nineq = bineq.shape[0]
    nvar = nx + nineq  # Num of variables in standard form: x and s variables

    # Form compact (c) matrices for standard QP from
    Qc = spla.block_diag(Q, np.zeros((nineq, nineq)))
    cc = np.append(c, np.zeros(nineq))
    Ac = np.vstack([np.hstack([Aeq, np.zeros((neq, nineq))]),
                    np.hstack([Aineq, np.eye(nineq)])])
    bc = np.append(beq, bineq)

    # Scaling (s): Normalize rows of Ac
    if scaling:
        scaler = np.sqrt(np.square(Ac).sum(1))  # norm of each row of Ac
        As = Ac / scaler[:, None]
        bs = bc / scaler
    else:
        As = Ac
        bs = bc

    # Factorize KKT matrix using LU decomposition (for now)
    KKT = np.vstack([np.hstack([Qc + rho * np.eye(nvar), As.T]),
                     np.hstack([As, -1. / rho * np.eye(nineq + neq)])])
    LU, piv = spla.lu_factor(KKT)

    print "Splitting QP Solver"
    print "-------------------\n"
    print "iter |\t   cost\n"

    # Run ADMM
    z = np.zeros(nvar)
    u = np.zeros(neq + nineq + nvar)  # Number of dual variables in ADMM

    for i in range(max_iter):

        # Update RHS of KKT system
        qtemp = -cc + rho * (z + np.dot(As.T, bs) -
                             np.dot(As.T, u[:neq + nineq]) - u[neq + nineq:])
        qbar = np.append(qtemp, np.zeros(nineq + neq))

        # x update
        # Select only first nvar elements
        x = spla.lu_solve((LU, piv), qbar)[:nvar]
        # z update
        z = project(x + u[neq + nineq:], lb, ub)
        # u update
        u = u + np.append(np.dot(As, x), x) - \
            np.append(np.zeros(neq + nineq), z) - \
            np.append(bs, np.zeros(nvar))

        # TODO: Stopping criterion

        # Print cost function every printiter iterations
        if (i + 1 == 1) | (np.mod(i + 1, printiter) == 0):
            xtemp = z[:nx]
            f = .5 * np.dot(np.dot(xtemp.T, Q), xtemp) + np.dot(c.T, xtemp)
            print "%4s | \t%7.2f" % (i + 1, f)
    print "Optimization Done\n"

    # Rescale (r) dual variables
    if scaling:
        dual_vars = rho * u[:neq + nineq] / scaler
    else:
        dual_vars = rho * u[:neq + nineq]

    # TODO: What is a status of the obtained solution?

    # TODO: Solution polishing

    # Retrieve solution
    sol_prim = z[:nx]
    objval = .5 * np.dot(np.dot(sol_prim, Q), sol_prim) + np.dot(c, sol_prim)

    # Retrieve dual variables
    sol_dual_eq = dual_vars[:neq]
    sol_dual_ineq = dual_vars[neq:]
    stat_cond = np.dot(Q, sol_prim) + \
        c + np.dot(Aeq.T, sol_dual_eq) + \
        np.dot(Aineq.T, sol_dual_ineq)
    sol_dual_lb = -np.minimum(stat_cond, 0)
    sol_dual_ub = np.maximum(stat_cond, 0)

    # Return solution as a quadProgSolution object
    return quadProgSolution(OPTIMAL, objval, sol_prim, sol_dual_eq,
                            sol_dual_ineq, sol_dual_lb, sol_dual_ub)


def isPSD(A, tol=1e-8):
    eigs, _ = sp.linalg.eigh(A)
    return np.all(eigs > -tol)
