import numpy as np
from scipy import linalg as spla
# import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spalinalg
import time   # Time execution
import ipdb		# ipdb.set_trace()

# Solver Constants
OPTIMAL = "optimal"
OPTIMAL_INACCURATE = "optimal_inaccurate"
MAXITER_REACHED = "maxiter_reached"
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
    eps_abs  [1e-06]            - Absolute tolerance
    eps_rel  [1e-06]           - Relative tolerance
    print_level [2]            - Printing level
    scaling  [False]           - Prescaling/Equilibration
    polish_tol [1e-5]          - Polishing tolerance to detect active constrs
    splitting [2]              - Splitting option
    warm_start [False]         - Reuse solution from previous solve


    Scaling
    ------------
    scale_problem [True]       - Scale Optimization Problem
    scaling_steps [10]         - Number of Steps for Scaling Method
    scaled [False]             - Has the system been scaled yet?

    KKT Solution
    ------------
    kkt_method  ['direct']        - KKT solution method: 'direct' (only direct)

    -> Direct method options:
    kkt_dir_reuse_factor [True]   - KKT factorization from previous solve
    """
    #  -> Indirect method options (dynamic rho update):
    #  kkt_ind_alg ['cg']            - Algorithm
    #  kkt_ind_tol [1e-5]            - Indirect algorithm tolerance
    #  kkt_ind_maxiter [100]         - Indirect algorithm maximum number of iter
    #  kkt_ind_mu [10]               - rho update trigger ratio between residuals
    #  kkt_ind_tau [2]               - rho update factor
    def __init__(self, **kwargs):

        # Set general options
        self.max_iter = kwargs.pop('max_iter', 5000)
        self.rho = kwargs.pop('rho', 1.0)
        self.alpha = kwargs.pop('alpha', 1.0)
        self.eps_abs = kwargs.pop('eps_abs', 1e-6)
        self.eps_rel = kwargs.pop('eps_rel', 1e-6)
        self.print_level = kwargs.pop('print_level', 2)
        if kwargs.pop('verbose', 1) == 0:
            self.print_level = 0
        self.scaling = kwargs.pop('scaling', False)
        self.polish = kwargs.pop('polish', True)
        self.polish_tol = kwargs.pop('polish_tol', 1e-05)
        self.splitting = kwargs.pop('splitting', 2)
        self.warm_start = kwargs.pop('warm_start', False)

        # Set scaling options
        self.scale_problem = kwargs.pop('scale_problem', True)
        self.scale_steps = kwargs.pop('scale_steps', 10)

        # Set KKT system solution options
        self.kkt_method = kwargs.pop('kkt_method', 'direct')
        self.kkt_dir_reuse_factor = kwargs.pop('kkt_dir_reuse_factor',
                                               True)
        #  self.kkt_ind_alg = kwargs.pop('kkt_ind_alg', 'cg')
        #  self.kkt_ind_tol = kwargs.pop('kkt_ind_tol', 1e-5)
        #  self.kkt_ind_maxiter = kwargs.pop('kkt_ind_maxiter', 100)
        #  self.kkt_ind_tau = kwargs.pop('kkt_ind_tau', 2.)
        #  self.kkt_ind_mu = kwargs.pop('kkt_ind_mu', 10.)


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
    scaled_problem   - scaled QP problem
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
        #  self.options.kkt_ind_alg = kwargs.pop(
            #  'kkt_ind_alg',
            #  'cg')
        #  self.options.kkt_ind_tol = kwargs.pop(
            #  'kkt_ind_tol',
            #  self.options.kkt_ind_tol)
        #  self.options.kkt_ind_maxiter = kwargs.pop(
            #  'kkt_ind_maxiter',
            #  self.options.kkt_ind_maxiter)
        #  self.options.kkt_ind_mu = kwargs.pop(
            #  'kkt_ind_mu',
            #  self.options.kkt_ind_mu)
        #  self.options.kkt_ind_tau = kwargs.pop(
            #  'kkt_ind_tau',
            #  self.options.kkt_ind_tau)

    #  def scale_problem(self):
        #  """
        #  Perform diagonal scaling via equilibration
        #  """
        #  # Define reduced KKT matrix to scale
        #  KKT = spspa.vstack([
            #  spspa.hstack([self.problem.Q + self.options.rho *
                          #  spspa.eye(nx), Ac.T]),
            #  spspa.hstack([Ac,
                          #  -1./self.options.rho * spspa.eye(nconstr)])])
        

    def project(self, xbar):
        """
        Project a vector xbar onto constraint set.
        """
        # Project first nx elements on interval [lb, ub],
        # next neq elements on zero, and the rest on positive
        # orthant.
        xbar[:self.problem.nx] = np.minimum(
                np.maximum(xbar[:self.problem.nx],
                           self.problem.lb), self.problem.ub)
        xbar[self.problem.nx:self.problem.nx+self.problem.neq] = 0.0
        xbar[self.problem.nx+self.problem.neq:] = np.maximum(
                xbar[self.problem.nx+self.problem.neq:], 0.0)
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
        solution = self.solve_admm()

        return solution

    def solve_admm(self):
        """" 
        Solve splitting problem with splitting of the form (after introducing
        slack variables for both equality and inequality constraints)
            minimize	1/2 x' Q x + c'x  + I_{Ax=b}(x) + I_Z(z)
            subject to	x == z
        where Z = [lb, ub]^nx x {0}^neq x R+^nineq.
        Slack variable for equality constraints is introduced for dealing with
        an issue of KKT matrix being singular when Aeq does not have full rank.
        """

        # Print parameters
        #  print "Splitting method 2"
        #  print "KKT solution: " + self.options.kkt_method + "\n"

        # Start timer
        t = time.time()

        # Number of variables (x,s1,s2) and constraints
        nx = self.problem.nx
        neq = self.problem.neq
        nineq = self.problem.nineq
        nvar = nx + neq + nineq
        nconstr = neq + nineq

        # Check whether the problem matrices have already been constructed.
        # If direct method is used and factorization stored, get it
        if self.options.kkt_dir_reuse_factor and \
                self.kkt_factor is not None:
            kkt_factor = self.kkt_factor
            # scaler = self.scaler
        else:
            # Construct reduced KKT matrix
            Ac = spspa.vstack([self.problem.Aeq, self.problem.Aineq])
            KKT = spspa.vstack([
                spspa.hstack([self.problem.Q + self.options.rho *
                              spspa.eye(nx), Ac.T]),
                spspa.hstack([Ac,
                              -1./self.options.rho * spspa.eye(nconstr)])])
            kkt_factor = spalinalg.splu(KKT.tocsc())
            if self.options.kkt_dir_reuse_factor:
                # Store factorization
                self.kkt_factor = kkt_factor

        # Construct augmented b vector
        bc = np.append(self.problem.beq, self.problem.bineq)

        # Set initial conditions
        if self.options.warm_start and self.z_prev is not None \
                and self.u_prev is not None:
                z = self.z_prev
                u = self.u_prev
        else:
            z = np.zeros(nvar)
            u = np.zeros(nvar)

        if self.options.print_level > 1:
            print "Iter \t  Objective \tPrim Res \tDual Res"

        # Run ADMM: alpha \in (0, 2) is a relaxation parameter.
        #           Nominal ADMM is obtained for alpha=1.0
        for i in xrange(self.options.max_iter):
            # x update
            rhs = np.append(self.options.rho * (z[:nx] - u[:nx]) -
                            self.problem.c, bc - z[nx:] + u[nx:])
            sol_kkt = kkt_factor.solve(rhs)
            x = np.append(sol_kkt[:nx], z[nx:] - u[nx:] -
                          1./self.options.rho * sol_kkt[nx:])
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
                    f = self.problem.objval(z[:nx])
                    print "%4s \t%1.7e  \t%1.2e  \t%1.2e" \
                        % (i+1, f, resid_prim, resid_dual)
                # Stop the algorithm
                break

            # Print cost function depending on print level
            if self.options.print_level > 1:
                if (i + 1 == 1) | (self.options.print_level == 2) & \
                        (np.mod(i + 1,
                         np.floor(np.float(self.options.max_iter)/20.0)) == 0)\
                        | (self.options.print_level == 3):
                            f = self.problem.objval(z[:nx])
                            print "%4s \t%1.7e  \t%1.2e  \t%1.2e" \
                                % (i+1, f, resid_prim, resid_dual)

        # End timer
        cputime = time.time() - t
        total_iter = i

        # Return status
        print "\n"
        if i == self.options.max_iter - 1:
            print "Maximum number of iterations exceeded!"
            status = MAXITER_REACHED
        else:
            print "Optimal solution found"
            status = OPTIMAL

        print "Elapsed time: %.3fs\n" % cputime

        # Recover primal solution
        sol_x = z[:nx]
        objval = self.problem.objval(sol_x)

        # Recover dual solution
        dual_vars = -self.options.rho * u
        sol_dual_eq = dual_vars[nx:nx+neq]
        sol_dual_ineq = dual_vars[nx+neq:]
        sol_dual_lb = np.maximum(dual_vars[:nx], 0)
        sol_dual_ub = -np.minimum(dual_vars[:nx], 0)

        # Store solution as a quadprogResults object
        solution = results(status, objval, sol_x, sol_dual_eq,
                           sol_dual_ineq, sol_dual_lb, sol_dual_ub,
                           cputime, total_iter)

        # Polish only if optimal solution reached
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
