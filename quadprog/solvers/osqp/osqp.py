import numpy as np
from scipy import linalg as spla
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spalinalg
from quadprog.results import quadprogResults  # Import results class
import quadprog.problem as qp   # Import statuses
import time   # Time execution
import ipdb		# ipdb.set_trace()


# def isPSD(A, tol=1e-8):
#     """
#     Check if the given matrix is positive semidefinite.
#     """
#     eigs, _ = sp.linalg.eigh(A)
#     return np.all(eigs > -tol)


# def polishedSolution(Q, c, Aeq, beq, Aineq, bineq, lb, ub, QP_sol):
#     """
#     Try to reconstruct the actual solution of a QP by guessing which
#     bounds are active. It boils down to solving a linear system.
#     """
#     # Check which of the ub-x, x-lb and dual variables corresponding to bound
#     # constraints has the largest value
#     x_range = ub - lb
#     dv_range = max(np.max(QP_sol.sol_dual_lb), np.max(QP_sol.sol_dual_ub))
#     K = np.array([(ub - QP_sol.x) / x_range, (QP_sol.x - lb) / x_range,
#                  QP_sol.sol_dual_lb / dv_range, QP_sol.sol_dual_ub / dv_range])
#     K_maxind = np.argmax(K, axis=0)
#
#     # Deduce which variables are active




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
def solve(Q, c, Aeq, beq, Aineq, bineq, lb, ub, **kwargs):
    """
    Operator splitting solver for a QP given
    in the following form
            minimize	1/2 x' Q x + c'x
            subject to	Aeq x == beq
                        Aineq x <= bineq
                        lb <= x <= ub
    """

    # Set passed options and set default values
    max_iter = kwargs.pop('max_iter', 5000)
    rho = kwargs.pop('rho', 1.6)
    alpha = kwargs.pop('alpha', 1.0)
    eps_abs = kwargs.pop('eps_abs', 0.0)
    eps_rel = kwargs.pop('eps_rel', 1e-7)
    print_level = kwargs.pop('print_level', 2)
    scaling = kwargs.pop('scaling', False)
    # prev_sol = kwargs.pop('prev_sol', False)  TODO: Add warm starting

    # Start timer
    t = time.time()

    # Get dimensions
    nx = c.shape[0]
    neq = beq.shape[0]
    nineq = bineq.shape[0]
    nvar = nx + nineq  # Num of variables in standard form: x and s variables

    # Form compact (c) matrices for standard QP from:
    #       minimize	1/2 z' Qc z + cc'z
    #       subject to	Ac z == bc
    #                   z \in Z
    Qc = spspa.block_diag((Q, spspa.csc_matrix((nineq, nineq))))
    cc = np.append(c, np.zeros(nineq))
    Ac = spspa.vstack([spspa.hstack([Aeq, spspa.csc_matrix((neq, nineq))]),
                       spspa.hstack([Aineq, spspa.eye(nineq)])])
    bc = np.append(beq, bineq)
    M = neq + nineq

    # Scaling (s): Normalize rows of Ac.
    if scaling:
        scaler = np.sqrt(np.square(Ac).sum(1))  # norm of each row of Ac
        As = Ac / scaler[:, None]
        bs = bc / scaler
    else:
        As = Ac
        bs = bc

    # # Warm starting (TODO: add later)
    # if prev_sol:    # If the previous solution is passed as an argument
    #     # Reuse the factorization from previous solution
    #     LU = prev_sol.lu_kkt
    #     piv = prev_sol.piv_kkt
    #     # Set initial condition to previous solution
    #     z = prev_sol.ADMM_z_iter
    #     u = prev_sol.ADMM_u_iter
    # else:

    # Factorize KKT matrix using LU decomposition (for now)
    KKT = spspa.vstack([spspa.hstack([Qc + rho * spspa.eye(nvar), As.T]),
                       spspa.hstack([As, -1. / rho * spspa.eye(M)])])
    luKKT = spalinalg.splu(KKT.tocsc())  # Perform sparse LU factorization

    # Set initial conditions to zero
    z = np.zeros(nvar)
    u = np.zeros(M + nvar)

    print "Splitting QP Solver"
    print "-------------------\n"
    if print_level > 1:
        print "iter | \t   cost\t |   prim_res\t |   dual_res\n"

    # Run ADMM: alpha \in (0, 2) is a relaxation parameter.
    #           Nominal ADMM is obtained for alpha=1.0
    for i in range(max_iter):

        # Update RHS of KKT system
        qtemp = -cc + rho * (z + As.T.dot(bs) - As.T.dot(u[:M]) - u[M:])
        qbar = np.append(qtemp, np.zeros(nineq + neq))

        # x update
        x = luKKT.solve(qbar)[:nvar]  # Select first nvar elements
        # z update
        z_old = z
        z = project(alpha*x + (1.-alpha)*z_old + u[neq + nineq:], lb, ub)
        # u update
        u = u + alpha*np.append(As.dot(x), x) - \
            np.append(np.zeros(neq + nineq), z - (1.-alpha)*z_old) - \
            alpha*np.append(bs, np.zeros(nvar))

        # Compute primal and dual residuals
        resid_prim = spla.norm(np.append(Ac.dot(x) - bc, x - z)) # eq constr violation
        resid_dual = rho*spla.norm(z - z_old)

        # Check the stopping criterion
        eps_prim = (neq + nineq + nvar) * eps_abs \
            + eps_rel * np.max([spla.norm(Ac.dot(x)) + spla.norm(x),
                                spla.norm(z),
                                spla.norm(bc)])

        eps_dual = nvar * eps_abs + eps_rel * rho * \
            (spla.norm(As.T.dot(u[:M])) + spla.norm(u[M:]))
        if resid_prim <= eps_prim and resid_dual <= eps_dual:
            # Stop the algorithm
            break

        # Print cost function depending on print level
        if print_level > 1:
            if (i + 1 == 1) | (print_level == 2) & \
                    (np.mod(i + 1, np.floor(np.float(max_iter) / 20.0)) == 0) \
                    | (print_level == 3):
                        xtemp = z[:nx]
                        f = .5 * np.dot(xtemp.T, Q.dot(xtemp)) + \
                            c.T.dot(xtemp)
                        print "%4s | \t%7.2f\t |   %8.4f\t |   %8.4f" \
                            % (i+1, f, resid_prim, resid_dual)

    # End timer
    cputime = time.time() - t
    total_iter = i

    print "Optimization Done in %.2fs\n" % cputime

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
    x = z[:nx]
    objval = .5 * np.dot(x, Q.dot(x)) + np.dot(c, x)

    # Recover dual solution
    sol_dual_eq = dual_vars[:neq]
    sol_dual_ineq = dual_vars[neq:]
    stat_cond_resid = Q.dot(x) + c + \
        Aeq.T.dot(sol_dual_eq) + Aineq.T.dot(sol_dual_ineq)
    sol_dual_lb = np.maximum(stat_cond_resid, 0)
    sol_dual_ub = -np.minimum(stat_cond_resid, 0)

    # Return status
    status = qp.OPTIMAL

    # Store solution as a quadprogResults object
    QP_sol = quadprogResults(status, objval, x, sol_dual_eq,
                             sol_dual_ineq, sol_dual_lb, sol_dual_ub,
                             cputime, total_iter)

    # # Solution polishing (try to guess the active set and solve a lin. sys.)
    # if status == qp.OPTIMAL:
    #     QP_sol = polishedSolution(Q, c, Aeq, beq, Aineq, bineq, lb, ub, QP_sol)

    # Return solution
    return QP_sol
