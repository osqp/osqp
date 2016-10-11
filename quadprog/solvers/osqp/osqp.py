import numpy as np
from scipy import linalg as spla
# import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spalinalg
import time   # Time execution
import ipdb		# ipdb.set_trace()

# Solver Constants
OPTIMAL = "optimal"
# OPTIMAL_INACCURATE = "optimal_inaccurate"
# INFEASIBLE = "infeasible"
# INFEASIBLE_INACCURATE = "infeasible_inaccurate"
# UNBOUNDED = "unbounded"
# UNBOUNDED_INACCURATE = "unbounded_inaccurate"
# SOLVER_ERROR = "solver_error"


class results(object):
    """
    Stores OSQP results
    """

    def __init__(self, status, objval, x, sol_dual_eq,
                 sol_dual_ineq, sol_dual_lb, sol_dual_ub, cputime, total_iter):
        self.status = status
        self.objval = objval
        self.x = x
        self.sol_dual_eq = sol_dual_eq
        self.sol_dual_ineq = sol_dual_ineq
        self.sol_dual_lb = sol_dual_lb
        self.sol_dual_ub = sol_dual_ub
        self.cputime = cputime
        self.total_iter = total_iter


class problem(object):
    """
    QP problem of the form
        minimize	1/2 x' Q x + c'x
        subject to	Aeq x == beq
                    Aineq x <= bineq
                    lb <= x <= ub

    Attributes
    ----------
    Q, c
    Aeq, beq
    Aineq, bineq
    lb, ub
    """

    def __init__(self, Q, c, Aeq, beq, Aineq, bineq, lb=None, ub=None):
        # Set problem dimensions
        self.nx = c.size
        self.neq = beq.size
        self.nineq = bineq.size
        # Set problem data
        self.Q = Q.tocsr()
        self.c = c
        self.Aeq = Aeq.tocsr()
        self.beq = beq
        self.Aineq = Aineq.tocsr()
        self.bineq = bineq
        self.lb = lb if lb is not None else -np.inf*np.ones(self.nx)
        self.ub = ub if ub is not None else np.inf*np.ones(self.nx)

    def objval(self, x):
        # Compute quadratic objective value for the given x
        return .5 * np.dot(x, self.Q.dot(x)) + np.dot(self.c, x)


class options(object):
    """
    OSQP Solver Options

    Attributes (General)
    ----------
    max_iter [5000]            - Maximum number of iterations
    rho  [1.0]                 - Step in ADMM procedure
    alpha [1.0]                - Relaxation parameter
    eps_abs  [1e-8]            - Absolute tolerance
    eps_rel  [1e-06]           - Relative tolerance
    print_level [2]            - Printing level
    scaling  [False]           - Prescaling/Equilibration
    polish_tol [1e-5]          - Polishing tolerance to detect active constrs
    splitting [2]              - Splitting option
    warm_start [False]         - Reuse solution from previous solve


    KKT Solution
    ------------
    kkt_method  ['direct']        - KKT solution method: 'direct' or 'indirect'

    -> Direct method options:
    kkt_dir_reuse_factor [True]   - KKT factorization from previous solve

    -> Indirect method options (dynamic rho update):
    kkt_ind_alg ['cg']            - Algorithm
    kkt_ind_mu [10]               - rho update trigger ratio between residuals
    kkt_ind_tau [2]               - rho update factor
    """

    def __init__(self, **kwargs):

        # Set general options
        self.max_iter = kwargs.pop('max_iter', 5000)
        self.rho = kwargs.pop('rho', 1.6)
        self.alpha = kwargs.pop('alpha', 1.0)
        self.eps_abs = kwargs.pop('eps_abs', 1e-6)
        self.eps_rel = kwargs.pop('eps_rel', 1e-6)
        self.print_level = kwargs.pop('print_level', 2)
        self.scaling = kwargs.pop('scaling', False)
        self.polish = kwargs.pop('polish', True)
        self.polish_tol = kwargs.pop('polish_tol', 1e-05)
        self.splitting = kwargs.pop('splitting', 2)
        self.warm_start = kwargs.pop('warm_start', False)

        # Set KKT system solution options
        self.kkt_method = kwargs.pop('kkt_method', 'direct')
        self.kkt_dir_reuse_factor = kwargs.pop('kkt_dir_reuse_factor',
                                               True)
        self.kkt_ind_alg = kwargs.pop('kkt_ind_alg', 'cg')
        self.kkt_ind_tau = kwargs.pop('kkt_ind_tau', 10.)
        self.kkt_ind_mu = kwargs.pop('kkt_ind_mu', 10.)


# class prev_solution(object):
#     """
#
#     """
#     def __init__(self, splitting, factor, z_prev, u_prev):
#         self.splitting = splitting
#         self.factor = factor
#         self.z_prev = z_prev
#         self.u_prev = u_prev


class OSQP(object):
    """
    OSQP Solver Object class

    Attributes
    ----------
    problem          - QP problem
    options          - Solver options
    kkt_factor       - KKT Matrix factorization (direct method)
    z_prev           - Previous primal solution
    u_prev           - Previous dual solution
    """

    def __init__(self, **kwargs):
        """
        Define OSQP Solver by passing solver options
        """
        self.options = options(**kwargs)
        self.kkt_factor = None
        # self.scaler = None
        self.z_prev = None
        self.u_prev = None

    def problem(self, Q, c, Aeq, beq, Aineq, bineq, lb=None, ub=None):
        """
        Defines QP problem of the form
            minimize	1/2 x' Q x + c'x
            subject to	Aeq x == beq
                        Aineq x <= bineq
                        lb <= x <= ub
        """
        self.problem = problem(Q, c, Aeq, beq, Aineq, bineq, lb, ub)

    def set_problem_data(self, **kwargs):
        """
        Set QP problem data. Reset factorization if Q, Aeq or Aineq
        is changed.
        """
        if 'Q' in kwargs.keys() and kwargs['Q'].tocsr() != self.problem.Q:
            self.problem.Q = kwargs['Q'].tocsr()
            # Reset factorization
            self.kkt_factor = None
        if 'Aeq' in kwargs.keys() and \
                kwargs['Aeq'].tocsr() != self.problem.Aeq:
            self.problem.Aeq = kwargs['Aeq'].tocsr()
            # Reset factorization
            self.kkt_factor = None
        if 'Aineq' in kwargs.keys() and \
                kwargs['Aineq'].tocsr() != self.problem.Aineq:
            self.problem.Aineq = kwargs['Aineq'].tocsr()
            # Reset factorization
            self.kkt_factor = None
        self.problem.c = kwargs.pop('c', self.problem.c)
        self.problem.beq = kwargs.pop('beq', self.problem.beq)
        self.problem.bineq = kwargs.pop('bineq', self.problem.bineq)
        self.problem.lb = kwargs.pop('lb', self.problem.lb)
        self.problem.ub = kwargs.pop('ub', self.problem.ub)

    def set_option(self, **kwargs):
        """
        Set solver options. Reset factorization if rho or splitting
        is changed.
        """

        # General options
        if 'rho' in kwargs.keys() and kwargs['rho'] != self.options.rho:
            self.options.rho = kwargs['rho']
            # Reset factorization
            self.kkt_factor = None
        if 'splitting' in kwargs.keys() and \
                kwargs['splitting'] != self.options.splitting:
            self.options.splitting = kwargs['splitting']
            # Reset factorization
            self.kkt_factor = None
        self.options.max_iter = kwargs.pop('max_iter', self.options.max_iter)
        self.options.alpha = kwargs.pop('alpha', self.options.alpha)
        self.options.eps_abs = kwargs.pop('eps_abs', self.options.eps_abs)
        self.options.eps_rel = kwargs.pop('eps_rel', self.options.eps_rel)
        self.options.print_level = kwargs.pop('print_level',
                                              self.options.print_level)
        self.options.scaling = kwargs.pop('scaling', self.options.scaling)
        self.options.polish = kwargs.pop('polish', self.options.polish)
        self.options.polish_tol = kwargs.pop('polish_tol',
                                             self.options.polish_tol)

        #  KKT System solution options
        self.options.kkt_method = kwargs.pop('kkt_method',
                                             self.options.kkt_method)
        self.options.kkt_dir_reuse_factor = kwargs.pop(
            'kkt_dir_reuse_factor',
            self.options.kkt_dir_reuse_factor)
        self.options.kkt_ind_alg = kwargs.pop(
            'kkt_ind_alg',
            'cg')
        self.options.kkt_ind_mu = kwargs.pop(
            'kkt_ind_mu',
            self.options.kkt_ind_mu)
        self.options.kkt_ind_tau = kwargs.pop(
            'kkt_ind_tau',
            self.options.kkt_ind_tau)

    def project(self, xbar):
        """
        Project a vector xbar onto constraint set.
        """
        if self.options.splitting == 1:
            # Project first nx elements on interval [lb, ub],
            # and the rest on positive orthant.
            xbar[:self.problem.nx] = np.minimum(
                                        np.maximum(xbar[:self.problem.nx],
                                        self.problem.lb), self.problem.ub)
            xbar[self.problem.nx:] = np.maximum(xbar[self.problem.nx:], 0.0)
        elif self.options.splitting == 2:
            # Project first nx elements on interval [lb, ub],
            # next neq elements on zero, and the rest on positive
            # orthant.
            xbar[:self.problem.nx] = np.minimum(
                                        np.maximum(xbar[:self.problem.nx],
                                        self.problem.lb), self.problem.ub)
            xbar[self.problem.nx:self.problem.nx+self.problem.neq] = 0.0
            xbar[self.problem.nx+self.problem.neq:] = np.maximum(
                                xbar[self.problem.nx+self.problem.neq:], 0.0)
        else:
            assert False, "Unknown splitting"

        return xbar

    def polish(self, solution):
        """
        Try to reconstruct the actual solution of a QP by guessing which
        constraints are active. The problem boils down to solving a linear
        system.
        """
        sol_x = solution.x
        sol_lb = solution.sol_dual_lb
        sol_ub = solution.sol_dual_ub

        # Try to guess from an approximate solution which bounds are active
        bool_lb = np.logical_and(sol_x < (self.problem.lb+self.problem.ub)/2.,
                                 sol_lb > sol_x - self.problem.lb)
        bool_ub = np.logical_and(sol_x > (self.problem.lb+self.problem.ub)/2.,
                                 sol_ub > self.problem.ub - sol_x)
        ind_lb = np.where(bool_lb)[0]
        ind_ub = np.where(bool_ub)[0]
        # All the other elements of x are free
        ind_free = np.where(np.logical_not(np.logical_or(bool_lb, bool_ub)))[0]

        # Try to guess which inequality constraints are active
        ineq_act = solution.sol_dual_ineq > \
            (self.problem.bineq - self.problem.Aineq.dot(sol_x))
        ind_act = np.where(ineq_act)[0]                     # Aineq x = bineq
        ind_inact = np.where(np.logical_not(ineq_act))[0]   # Aineq x < bineq

        # Solve the corresponding linear system
        KKT = spspa.vstack([
            spspa.hstack([self.problem.Q, self.problem.Aeq.T,
                         self.problem.Aineq.T, spspa.eye(self.problem.nx)]),
            spspa.hstack([self.problem.Aeq,
                         spspa.csr_matrix((self.problem.neq, self.problem.neq +
                                           self.problem.nineq +
                                           self.problem.nx))]),
            spspa.hstack([self.problem.Aineq[ind_act],
                          spspa.csr_matrix((len(ind_act), self.problem.neq +
                                            self.problem.nineq +
                                            self.problem.nx))]),
            spspa.hstack([spspa.csr_matrix((len(ind_inact),
                                            self.problem.nx+self.problem.neq)),
                          spspa.eye(self.problem.nineq).tocsr()[ind_inact],
                          spspa.csr_matrix((len(ind_inact),
                                            self.problem.nx))]),
            spspa.hstack([spspa.csr_matrix((len(ind_free), self.problem.nx +
                                            self.problem.neq +
                                            self.problem.nineq)),
                          spspa.eye(self.problem.nx).tocsr()[ind_free]]),
            spspa.hstack([spspa.eye(self.problem.nx).tocsr()[ind_lb],
                          spspa.csr_matrix((len(ind_lb), self.problem.neq +
                                            self.problem.nineq +
                                            self.problem.nx))]),
            spspa.hstack([spspa.eye(self.problem.nx).tocsr()[ind_ub],
                          spspa.csr_matrix((len(ind_ub), self.problem.neq +
                                            self.problem.nineq +
                                            self.problem.nx))]),
            ]).tocsr()
        rhs = np.hstack([-self.problem.c, self.problem.beq,
                        self.problem.bineq[ind_act],
                        np.zeros(len(ind_inact) + len(ind_free)),
                        self.problem.lb[ind_lb], self.problem.ub[ind_ub]])
        try:
            pol_sol = spalinalg.spsolve(KKT, rhs)
        except:
            # Failed to factorize KKT matrix
            print "Polishing failed. Failed to factorize KKT matrix."
            return solution

        # If the KKT matrix is singular, spsolve return an array of NaNs
        if any(np.isnan(pol_sol)):
            # Terminate
            print "Polishing failed. KKT matrix is singular."
            return solution

        # Check if the above solution satisfies constraints
        pol_x = pol_sol[:self.problem.nx]
        pol_dual_eq = pol_sol[self.problem.nx:self.problem.nx+self.problem.neq]
        pol_dual_ineq = pol_sol[self.problem.nx+self.problem.neq:
                                self.problem.nx + self.problem.neq +
                                self.problem.nineq]
        pol_dual_lb = np.zeros(self.problem.nx)
        pol_dual_lb[ind_lb] = -pol_sol[self.problem.nx + self.problem.neq +
                                       self.problem.nineq:][ind_lb]
        pol_dual_ub = np.zeros(self.problem.nx)
        pol_dual_ub[ind_ub] = pol_sol[self.problem.nx + self.problem.neq +
                                      self.problem.nineq:][ind_ub]
        if all(pol_x > self.problem.lb - self.options.polish_tol) and\
                all(pol_x < self.problem.ub + self.options.polish_tol)\
                and all(pol_dual_ineq > -self.options.polish_tol)\
                and all(pol_dual_lb > -self.options.polish_tol)\
                and all(pol_dual_ub > -self.options.polish_tol):
                # Return the computed high-precision solution
                print "Polishing successful!"
                solution.x = pol_x
                solution.sol_dual_eq = pol_dual_eq
                solution.sol_dual_ineq = pol_dual_ineq
                solution.sol_dual_lb = pol_dual_lb
                solution.sol_dual_ub = pol_dual_ub
                solution.objval = self.problem.objval(pol_x)
        else:
            print "Polishing failed!"

        return solution

    # def solve(Q, c, Aeq, beq, Aineq, bineq, lb, ub, **kwargs):
    def solve(self):
        """
        Operator splitting solver for a QP given
        in the following form
                minimize	1/2 x' Q x + c'x
                subject to	Aeq x == beq
                            Aineq x <= bineq
                            lb <= x <= ub
        """

        print "Operator Splitting QP Solver"
        print "----------------------------\n"

        # Choose splitting based on the options
        if self.options.splitting == 1:
            solution = self.solve_splitting1()
        elif self.options.splitting == 2:
            solution = self.solve_splitting2()
        else:
            assert False, "Unknown splitting"

        return solution

    def solve_splitting1(self):
        """"
        Solve splitting problem with splitting of the form (after introducing
        slack variables for inequalities)
            minimize	1/2 x' Q x + c'x
            subject to	[A] x - [0]z = [b]
                        [I]     [I]    [0]
                        z in C
        where C is a set of lower and upper bounds
        """

        # Start timer
        t = time.time()

        # Number of variables and constraints in standard form
        nvar = self.problem.nx + self.problem.nineq     # (x,s)
        nconstr = self.problem.neq + self.problem.nineq

        # Check whether the problem matrices have already been constructed
        if self.options.kkt_dir_reuse_factor and self.kkt_factor is not None:
            kkt_factor = self.kkt_factor
            As = self.As
            # scaler = self.scaler
        else:
            # Form compact (c) matrices for standard QP from:
            #       minimize	1/2 z' Qc z + cc'z
            #       subject to	Ac z == bc
            #                   z \in Z
            Qc = spspa.block_diag((self.problem.Q,
                                   spspa.csc_matrix((self.problem.nineq,
                                                     self.problem.nineq))))
            Ac = spspa.vstack([spspa.hstack([self.problem.Aeq,
                                            spspa.csc_matrix(
                                                (self.problem.neq,
                                                 self.problem.nineq))]),
                               spspa.hstack([self.problem.Aineq,
                                            spspa.eye(self.problem.nineq)])])
            # Scaling (s): Normalize rows of Ac.
            if self.options.scaling:
                scaler = np.sqrt(np.square(Ac).sum(1))  # norm of rows of Ac
                As = Ac / scaler[:, None]
            else:
                As = Ac
            # Factorize KKT matrix using sparse LU decomposition
            KKT = spspa.vstack([
                spspa.hstack([Qc + self.options.rho*spspa.eye(nvar), As.T]),
                spspa.hstack([As, -1./self.options.rho*spspa.eye(nconstr)])])
            kkt_factor = spalinalg.splu(KKT.tocsc())
            if self.options.kkt_dir_reuse_factor:
                # Store factorization
                self.factor = kkt_factor
                self.As = As

        cc = np.append(self.problem.c, np.zeros(self.problem.nineq))
        bc = np.append(self.problem.beq, self.problem.bineq)
        if self.options.scaling:
            bs = bc / scaler
        else:
            bs = bc

        # Set initial conditions
        if self.options.warm_start and self.z_prev is not None \
                and self.u_prev is not None:
            z = self.z_prev
            u = self.u_prev
        else:
            z = np.zeros(nvar)
            u = np.zeros(nconstr + nvar)

        # Print parameters
        print "Splitting method 1"
        print "KKT solution: " + self.options.kkt_method + "\n"

        if self.options.print_level > 1:
            print "Iter | \t   Cost\t    Prim Res\t    Dual Res"

        # Run ADMM: alpha \in (0, 2) is a relaxation parameter.
        #           Nominal ADMM is obtained for alpha=1.0
        for i in xrange(self.options.max_iter):

            # Update RHS of KKT system
            qtemp = -cc + self.options.rho * \
                    (z + As.T.dot(bs) - As.T.dot(u[:nconstr]) - u[nconstr:])
            qbar = np.append(qtemp, np.zeros(nconstr))

            # x update
            x = kkt_factor.solve(qbar)[:nvar]  # Select first nvar elements
            # z update
            z_old = z
            z = self.project(self.options.alpha*x +
                             (1.-self.options.alpha)*z_old + u[nconstr:])
            # u update
            u = u + self.options.alpha * np.append(As.dot(x), x) \
                - np.append(np.zeros(nconstr),
                            z - (1.-self.options.alpha)*z_old) \
                - self.options.alpha*np.append(bs, np.zeros(nvar))

            # Compute primal and dual residuals
            resid_prim = spla.norm(np.append(Ac.dot(x) - bc, x - z))  # eq
            resid_dual = self.options.rho*spla.norm(z - z_old)

            # Check the stopping criterion
            eps_prim = np.sqrt(nconstr + nvar) * self.options.eps_abs \
                + self.options.eps_rel * \
                np.max([spla.norm(Ac.dot(x)) + spla.norm(x),
                        spla.norm(z),
                        spla.norm(bc)])
            eps_dual = np.sqrt(nvar) * self.options.eps_abs + \
                self.options.eps_rel * self.options.rho * \
                (spla.norm(As.T.dot(u[:nconstr])) + spla.norm(u[nconstr:]))

            if resid_prim <= eps_prim and resid_dual <= eps_dual:
                # Print the progress in last iterations
                if self.options.print_level > 1:
                    f = self.problem.objval(z[:self.problem.nx])
                    print "%4s | \t%7.2f\t    %8.4f\t    %8.4f" \
                        % (i+1, f, resid_prim, resid_dual)
                # Stop the algorithm
                break

            # Print cost function depending on print level
            if self.options.print_level > 1:
                if (i + 1 == 1) | (self.options.print_level == 2) & \
                        (np.mod(i + 1,
                         np.floor(np.float(self.options.max_iter)/20.0)) == 0)\
                        | (self.options.print_level == 3):
                            f = self.problem.objval(z[:self.problem.nx])
                            print "%4s | \t%7.2f\t    %8.4f\t    %8.4f" \
                                % (i+1, f, resid_prim, resid_dual)

        # End timer
        cputime = time.time() - t
        total_iter = i

        print "Optimization Done in %.3fs\n" % cputime

        # Rescale (r) dual variables
        if self.options.scaling:
            dual_vars = self.options.rho * u[:nconstr] / scaler
        else:
            dual_vars = self.options.rho * u[:nconstr]

        # TODO: Scale Qc matrix as well

        # TODO: What is a status of the obtained solution?

        # Recover primal solution
        sol_x = z[:self.problem.nx]
        objval = self.problem.objval(sol_x)

        # Recover dual solution
        sol_dual_eq = dual_vars[:self.problem.neq]
        sol_dual_ineq = dual_vars[self.problem.neq:]
        stat_cond_resid = self.problem.Q.dot(sol_x) + self.problem.c + \
            self.problem.Aeq.T.dot(sol_dual_eq) + \
            self.problem.Aineq.T.dot(sol_dual_ineq)
        sol_dual_lb = np.maximum(stat_cond_resid, 0)
        sol_dual_ub = -np.minimum(stat_cond_resid, 0)

        # Return status
        status = OPTIMAL

        # Store solution as a quadprogResults object
        solution = results(status, objval, sol_x, sol_dual_eq,
                           sol_dual_ineq, sol_dual_lb, sol_dual_ub,
                           cputime, total_iter)

        # Solution polishing
        if status == OPTIMAL:
            solution = self.polish(solution)
            # Store last iterates for warm starting
            if self.options.warm_start:
                self.z_prev = z
                self.u_prev = u

        # Return solution
        return solution

    def solve_splitting2(self):
        """"
        Solve splitting problem with splitting of the form (after introducing
        slack variables for both equality and inequality constraints)
            minimize	1/2 x' Q x + c'x  + I_{Ax=b}(x) + I_Z(z)
            subject to	x == z
        where Z = [lb, ub]^nx x {0}^neq x R+^nineq.
        Slack variable for equality constraints is introduced for dealing with
        an issue of KKT matrix being singular when Aeq does not have full rank.
        """

        # Start timer
        t = time.time()

        # Number of variables (x,s1,s2) and constraints
        nvar = self.problem.nx + self.problem.neq + self.problem.nineq
        nconstr = self.problem.neq + self.problem.nineq

        # Check whether the problem matrices have already been constructed
        if self.options.kkt_dir_reuse_factor and self.kkt_factor is not None:
            kkt_factor = self.kkt_factor
            # scaler = self.scaler
        else:
            # Form compact (c) matrices for standard QP from:
            #       minimize	1/2 z' Qc z + cc'z
            #       subject to	Ac z == bc
            #                   z \in Z
            Qc = spspa.block_diag((self.problem.Q,
                                   spspa.csc_matrix((nconstr, nconstr))))
            Ac = spspa.vstack([
                    spspa.hstack([self.problem.Aeq, spspa.eye(self.problem.neq),
                                  spspa.csc_matrix((self.problem.neq,
                                                    self.problem.nineq))]),
                    spspa.hstack([self.problem.Aineq,
                                  spspa.csc_matrix((self.problem.nineq,
                                                    self.problem.neq)),
                                  spspa.eye(self.problem.nineq)])])
            # Factorize KKT matrix using sparse LU decomposition
            KKT = spspa.vstack([
                spspa.hstack([Qc + self.options.rho * spspa.eye(nvar), Ac.T]),
                spspa.hstack([Ac, spspa.csc_matrix((nconstr, nconstr))])])
            kkt_factor = spalinalg.splu(KKT.tocsc())
            if self.options.kkt_dir_reuse_factor:
                # Store factorization
                self.kkt_factor = kkt_factor

        cc = np.append(self.problem.c, np.zeros(nconstr))
        bc = np.append(self.problem.beq, self.problem.bineq)

        # Set initial conditions
        if self.options.warm_start and self.z_prev is not None \
            and self.u_prev is not None:
            z = self.z_prev
            u = self.u_prev
        else:
            z = np.zeros(nvar)
            u = np.zeros(nvar)

        # Print parameters
        print "Splitting method 2"
        print "KKT solution: " + self.options.kkt_method + "\n"

        if self.options.print_level > 1:
            print "Iter | \t   Cost\t    Prim Res\t    Dual Res"

        # Run ADMM: alpha \in (0, 2) is a relaxation parameter.
        #           Nominal ADMM is obtained for alpha=1.0
        for i in xrange(self.options.max_iter):
            # x update
            rhs = np.append(self.options.rho * (z - u) - cc, bc)
            x = kkt_factor.solve(rhs)[:nvar]
            # z update
            z_old = z
            z = self.project(self.options.alpha*x +
                             (1.-self.options.alpha)*z_old + u)
            # u update
            u = u + self.options.alpha*x + (1.-self.options.alpha)*z_old - z

            # Compute primal and dual residuals
            resid_prim = spla.norm(x - z)
            resid_dual = self.options.rho*spla.norm(z - z_old)

            # Check the stopping criterion
            eps_prim = np.sqrt(nconstr) * self.options.eps_abs \
                + self.options.eps_rel * np.max([spla.norm(x), spla.norm(z)])
            eps_dual = np.sqrt(nvar) * self.options.eps_abs + \
                self.options.eps_rel * self.options.rho * spla.norm(u)

            if resid_prim <= eps_prim and resid_dual <= eps_dual:
                # Print the progress in last iterations
                if self.options.print_level > 1:
                    f = self.problem.objval(z[:self.problem.nx])
                    print "%4s | \t%7.2f\t    %8.4f\t    %8.4f" \
                        % (i+1, f, resid_prim, resid_dual)
                # Stop the algorithm
                break

            # Print cost function depending on print level
            if self.options.print_level > 1:
                if (i + 1 == 1) | (self.options.print_level == 2) & \
                        (np.mod(i + 1,
                         np.floor(np.float(self.options.max_iter)/20.0)) == 0)\
                        | (self.options.print_level == 3):
                            f = self.problem.objval(z[:self.problem.nx])
                            print "%4s | \t%7.2f\t    %8.4f\t    %8.4f" \
                                % (i+1, f, resid_prim, resid_dual)

        # End timer
        cputime = time.time() - t
        total_iter = i

        print "Optimization done in %.3fs\n" % cputime

        # Recover primal solution
        sol_x = z[:self.problem.nx]
        objval = self.problem.objval(sol_x)

        # Recover dual solution
        dual_vars = -self.options.rho * u
        sol_dual_eq = dual_vars[self.problem.nx:
                                self.problem.nx+self.problem.neq]
        sol_dual_ineq = dual_vars[-self.problem.nineq:]
        stat_cond_resid = self.problem.Q.dot(sol_x) + self.problem.c + \
            self.problem.Aeq.T.dot(sol_dual_eq) + \
            self.problem.Aineq.T.dot(sol_dual_ineq)
        sol_dual_lb = np.maximum(stat_cond_resid, 0)
        sol_dual_ub = -np.minimum(stat_cond_resid, 0)

        # Return status
        status = OPTIMAL

        # Store solution as a quadprogResults object
        solution = results(status, objval, sol_x, sol_dual_eq,
                           sol_dual_ineq, sol_dual_lb, sol_dual_ub,
                           cputime, total_iter)

        if status == OPTIMAL:
            if self.options.polish:
                # Solution polishing
                solution = self.polish(solution)
            # Store last iterates for warm starting
            if self.options.warm_start:
                self.z_prev = z
                self.u_prev = u

        # Return solution
        return solution
