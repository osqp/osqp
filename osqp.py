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
    Stores solution of a QP.
    """

    def __init__(self, status, objval, sol_prim, sol_dual_eq,
                 sol_dual_ineq, sol_dual_lb, sol_dual_ub,
                 LU, piv, z, u):
        self.status = status
        self.objval = objval
        self.sol_prim = sol_prim
        self.sol_dual_eq = sol_dual_eq
        self.sol_dual_ineq = sol_dual_ineq
        self.sol_dual_lb = sol_dual_lb
        self.sol_dual_ub = sol_dual_ub
        self.lu_kkt = LU
        self.piv_kkt = piv
        self.ADMM_z_iter = z
        self.ADMM_u_iter = u


def isPSD(A, tol=1e-8):
    """
    Check if the given matrix is positive semidefinite.
    """
    eigs, _ = sp.linalg.eigh(A)
    return np.all(eigs > -tol)


def project(xbar, lb, ub):
    """
    Project first nx (size of lb) elements on interval [lb, ub],
    and the rest on positive orthant.
    """
    nx = np.size(lb)
    # Project x part to [lb, ub] interval
    xbar[:nx] = np.minimum(np.maximum(xbar[:nx], lb), ub)
    # Project slack variables to positive ortant
    xbar[nx:] = np.maximum(xbar[nx:], 0)

    return xbar


# Base QP Solver
def OSQP(Q, c, Aeq, beq, Aineq, bineq, lb, ub,
         max_iter=500, rho=1.6, alpha=1.0, print_level=2, scaling=False,
         prev_sol=None):
    """
    Operator splitting solver for a QP given
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

    # Form compact (c) matrices for standard QP from:
    #       minimize	1/2 z' Qc z + cc'z
    #       subject to	Ac z == bc
    #                   z \in Z
    Qc = spla.block_diag(Q, np.zeros((nineq, nineq)))
    cc = np.append(c, np.zeros(nineq))
    Ac = np.vstack([np.hstack([Aeq, np.zeros((neq, nineq))]),
                    np.hstack([Aineq, np.eye(nineq)])])
    bc = np.append(beq, bineq)

    # Scaling (s): Normalize rows of Ac.
    if scaling:
        scaler = np.sqrt(np.square(Ac).sum(1))  # norm of each row of Ac
        As = Ac / scaler[:, None]
        bs = bc / scaler
    else:
        As = Ac
        bs = bc

    # Warm starting
    if prev_sol:    # If the previous solution is passed as an argument
        # Reuse the factorization from previous solution
        LU = prev_sol.lu_kkt
        piv = prev_sol.piv_kkt
        # Set initial condition to previous solution
        z = prev_sol.ADMM_z_iter
        u = prev_sol.ADMM_u_iter
    else:
        # Factorize KKT matrix using LU decomposition (for now)
        KKT = np.vstack([np.hstack([Qc + rho * np.eye(nvar), As.T]),
                         np.hstack([As, -1. / rho * np.eye(nineq + neq)])])
        LU, piv = spla.lu_factor(KKT)
        # Set initial conditions to zero
        z = np.zeros(nvar)
        u = np.zeros(neq + nineq + nvar)

    print "Splitting QP Solver"
    print "-------------------\n"
    if print_level > 1:
        print "iter |\t   cost\n"

    # Run ADMM: alpha \in (0, 2) is a relaxation parameter.
    #           Nominal ADMM is obtained for alpha=1.0
    for i in range(max_iter):

        # Update RHS of KKT system
        qtemp = -cc + rho * (z + np.dot(As.T, bs) -
                             np.dot(As.T, u[:neq + nineq]) - u[neq + nineq:])
        qbar = np.append(qtemp, np.zeros(nineq + neq))

        # x update
        x = spla.lu_solve((LU, piv), qbar)[:nvar]  # Select first nvar elements
        # z update
        z_old = z
        z = project(alpha*x + (1.-alpha)*z_old + u[neq + nineq:], lb, ub)
        # u update
        u = u + alpha*np.append(np.dot(As, x), x) - \
            np.append(np.zeros(neq + nineq), z - (1.-alpha)*z_old) - \
            alpha*np.append(bs, np.zeros(nvar))

        # Print cost function depending on print level
        if print_level > 1:
            if (i + 1 == 1) | (print_level == 2) & \
                    (np.mod(i + 1, np.floor(np.float(max_iter) / 20.0)) == 0) \
                    | (print_level == 3):
                            xtemp = z[:nx]
                            f = .5 * np.dot(np.dot(xtemp.T, Q), xtemp) + \
                                np.dot(c.T, xtemp)
                            print "%4s | \t%7.2f" % (i + 1, f)
    print "Optimization Done\n"

    # Rescale (r) dual variables
    if scaling:
        dual_vars = rho * u[:neq + nineq] / scaler
    else:
        dual_vars = rho * u[:neq + nineq]

    # TODO: Scale Qc matrix as well

    # TODO: Stopping criterion

    # TODO: What is a status of the obtained solution?

    # TODO: Solution polishing

    # Recover primal solution
    sol_prim = z[:nx]
    objval = .5 * np.dot(np.dot(sol_prim, Q), sol_prim) + np.dot(c, sol_prim)

    # Recover dual solution
    sol_dual_eq = dual_vars[:neq]
    sol_dual_ineq = dual_vars[neq:]
    stat_cond_resid = np.dot(Q, sol_prim) + c + \
        np.dot(Aeq.T, sol_dual_eq) + np.dot(Aineq.T, sol_dual_ineq)
    sol_dual_lb = np.maximum(stat_cond_resid, 0)
    sol_dual_ub = -np.minimum(stat_cond_resid, 0)

    # Return solution as a quadProgSolution object
    return quadProgSolution(OPTIMAL, objval, sol_prim, sol_dual_eq,
                            sol_dual_ineq, sol_dual_lb, sol_dual_ub,
                            LU, piv, z, u)
