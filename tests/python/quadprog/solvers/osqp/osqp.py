import numpy as np
from scipy import linalg as spla
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spalinalg
import numpy.linalg as nplinalg
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

    def __init__(self, status, objval, x, dual, cputime, total_iter):
        self.status = status
        self.objval = objval
        self.x = x
        self.dual = dual
        self.cputime = cputime
        self.total_iter = total_iter


class problem(object):
    """
    QP problem of the form
        minimize	1/2 x' P x + q' x
        subject to	lA <= A x <= uA

    Attributes
    ----------
    P, q
    A, lA, uA
    """

    def __init__(self, P, q, A, lA=None, uA=None):
        # Set problem dimensions
        self.n = q.size
        self.m = lA.size
        # Set problem data
        self.P = P.tocsr()
        self.q = q
        self.A = A.tocsr()
        self.lA = lA if lA is not None else -np.inf*np.ones(self.m)
        self.uA = uA if uA is not None else np.inf*np.ones(self.m)

    def objval(self, x):
        # Compute quadratic objective value for the given x
        return .5 * np.dot(x, self.P.dot(x)) + np.dot(self.q, x)


class options(object):
    """
    OSQP Solver Options

    Attributes (General)
    ----------
    max_iter [5000]            - Maximum number of iterations
    rho  [1.6]                 - Step in ADMM procedure
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
    scale_steps [3]          - Number of Steps for Scaling Method
    scale_norm [2]           - Scaling norm in SK algorithm

    KKT Solution
    ------------
    kkt_method  ['direct']        - KKT solution method: 'direct' (only direct)
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
        self.rho = kwargs.pop('rho', 1.6)
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
        self.scale_norm = kwargs.pop('scale_norm', 2)

        # Set KKT system solution options
        self.kkt_method = kwargs.pop('kkt_method', 'direct')
        # self.kkt_dir_reuse_factor = kwargs.pop('kkt_dir_reuse_factor', True)
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

class scaler_matrices(object):
    """
    Matrices for Scaling
    """
    def __init__(self):
        self.D = None
        self.E = None
        self.Dinv = None
        self.Einv = None


class solver_solution(object):
    """
    Solver solution vectors z, u
    """
    def __init__(self):
        self.z = None
        self.u = None


class OSQP(object):
    """
    OSQP Solver Object class

    Attributes
    ----------
    problem          - QP problem
    scaled_problem   - Scaled QP problem
    scaler_matrices  - Diagonal scaling matrices
    options          - Solver options
    kkt_factor       - KKT Matrix factorization (direct method)
    status           - Solver status
    total_iter       - Total number of iterations
    cputime          - Elapsed cputime
    solution         - Solution
    """

    def __init__(self, **kwargs):
        """
        Define OSQP Solver by passing solver options
        """
        self.options = options(**kwargs)
        self.kkt_factor = None
        self.scaled_problem = None
        self.scaler_matrices = scaler_matrices()
        self.solution = solver_solution()

    def problem(self, P, q, A, lA=None, uA=None):
        """
        Defines QP problem of the form
            minimize	1/2 x' P x + q'x
            subject to	lA <= A x <= uA
        """
        self.problem = problem(P, q, A, lA, uA)

    def set_problem_data(self, **kwargs):
        """
        Set QP problem data. Reset factorization if P or A is changed.
        """
        if 'P' in kwargs.keys():
            self.problem.P = kwargs['P'].tocsr()
            # Reset factorization
            self.kkt_factor = None
        if 'A' in kwargs.keys():
            self.problem.A = kwargs['A'].tocsr()
            # Reset factorization
            self.kkt_factor = None
        self.problem.q = kwargs.pop('q', self.problem.q)
        self.problem.lA = kwargs.pop('lA', self.problem.lA)
        self.problem.uA = kwargs.pop('uA', self.problem.uA)

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

        # Set scaling options
        self.scale_problem = kwargs.pop('scale_problem',
                                        self.options.scale_problem)
        self.scale_steps = kwargs.pop('scale_steps',
                                      self.options.scale_steps)
        self.scale_norm = kwargs.pop('scale_norm',
                                     self.options.scale_norm)

        #  KKT System solution options
        self.options.kkt_method = kwargs.pop('kkt_method',
                                             self.options.kkt_method)
        # self.options.kkt_dir_reuse_factor = kwargs.pop(
        #     'kkt_dir_reuse_factor',
        #     self.options.kkt_dir_reuse_factor)
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


    def solve(self):
        """
        Operator splitting solver for a QP given
        in the following form
                minimize	1/2 x' P x + q'x
                subject to	lA <= A x <= uA
        """

        print "Operator Splitting QP Solver"
        print "----------------------------\n"

        # Start timer
        t = time.time()

        # Scale problem
        self.scale_problem()

        # Choose splitting based on the options
        self.solve_admm()

        # Polish solution if optimal
        # if (self.status == OPTIMAL) & self.options.polish:
            # self.polish()

        # Rescale solution back
        self.rescale_solution()

        # End timer
        self.cputime = time.time() - t
        print "Elapsed time: %.3fs\n" % self.cputime

        # ipdb.set_trace()
        # Return QP solution
        solution = self.get_qp_solution()

        return solution

    def get_qp_solution(self):

        # Recover primal solution
        sol_x = self.solution.z[:self.problem.n]
        objval = self.problem.objval(sol_x)

        # Recover dual solution
        dual = self.options.rho * self.solution.u

        # Store solution as a quadprogResults object
        solution = results(self.status, objval, sol_x, dual,
                           self.cputime, self.total_iter)

        # Return solution
        return solution

    def scale_problem(self):
        """
        Perform symmetric diagonal scaling via equilibration
        """
        # Predefine scaling vector
        # nx = self.problem.nx
        # nvar = self.problem.nx + self.problem.neq + self.problem.nineq
        # nconstr = self.problem.neq + self.problem.nineq
        #  d = np.multiply(sp.rand(nvar), np.ones(nvar))
        n = self.problem.n
        m = self.problem.m

        if self.options.scale_problem:

            # Check if the problem has already been scaled
            if self.scaled_problem is not None and \
                    self.scaler_matrices.D is not None and \
                    self.scaler_matrices.E is not None and \
                    self.scaler_matrices.Dinv is not None and \
                    self.scaler_matrices.Einv is not None:
                return

            # Stack up equalities and inequalities
            # Ac = spspa.vstack([self.problem.Aeq, self.problem.Aineq])
            # bc = np.hstack([self.problem.beq, self.problem.bineq])

            #  if self.problem.Q.count_nonzero():  # If there are nonzero elements
            d = np.ones(n + m)

            # Define reduced KKT matrix to scale
            KKT = spspa.vstack([
                spspa.hstack([self.problem.P, self.problem.A.T]),
                spspa.hstack([self.problem.A, spspa.csc_matrix((m, m))])])

            # KKT = spspa.vstack([
            #     spspa.hstack([self.problem.P + spspa.eye(n), self.problem.A.T]),
            #     spspa.hstack([self.problem.A, -spspa.eye(m)])])

            # Run Scaling
            KKT2 = KKT.copy()
            if self.options.scale_norm == 2:
                KKT2.data = np.square(KKT2.data)  # Elementwise square
            elif self.options.scale_norm == 1:
                KKT2.data = np.absolute(KKT2.data)  # Elementwise absolute value

            # ipdb.set_trace()


            # Perform Scalings as in GLPK solver: https://en.wikibooks.org/wiki/GLPK/Scaling
            # 1: Check if problem is well scaled
            #  if False:
            if (np.min(KKT2.data) >= 0.1) & (np.max(KKT2.data) <= 10):
                print "Problem already well scaled. No Scaling Required\n"
            else:
                #  print "Perform geometric scaling of KKT matrix: %i Steps\n" % \
                    #  self.options.scale_steps
                #  for i in range(self.options.scale_steps):
                    #  di = np.zeros(nvar)
                    #  for j in range(nvar):  # Iterate over all rows of KKT
                        #  maxj = np.max((KKT2.tocsc())[j, :]) + 1e-08
                        #  minj = np.min((KKT2.tocsc())[j, :]) + 1e-08
                        #  di[j] = 1./np.sqrt(maxj * minj)
                    #  d = np.multiply(di, d)
                    #  # DEBUG STUFF
                    #  S = spspa.diags(d)
                    #  KKTeq = S.dot(KKT.dot(S))
                    #  print "Norm of first row of KKT %.4e" % \
                        #  nplinalg.norm((KKT.todense())[1, :])
                    #  print "Norm of first row of KKTeq %.4e" % \
                        #  nplinalg.norm((KKTeq.todense())[1, :])
                    #  condKKT = nplinalg.cond(KKT.todense())
                    #  condKKTeq = nplinalg.cond(KKTeq.todense())
                    #  print "Condition number of KKT matrix %.4e" % condKKT
                    #  print "Condition number of KKTeq matrix %.4e" % condKKTeq
                    #  ipdb.set_trace()

                print "Perform symmetric scaling of KKT matrix: %i Steps\n" % \
                    self.options.scale_steps

                # Iterate Scaling
                for i in range(self.options.scale_steps):
                    #  print np.max(KKT2.dot(d))
                    #  ipdb.set_trace()
                    # Regularize components
                    KKT2d = KKT2.dot(d)
                    # Prevent division by 0
                    d = (n + m)*np.reciprocal(KKT2d + 1e-6)
                    # Prevent too large elements
                    d = np.minimum(np.maximum(d, -1e+10), 1e+10)
                    #  d = np.reciprocal(KKT2d)
                    # print "Scaling step %i\n" % i

                    # # DEBUG STUFF
                    S = spspa.diags(d)
                    KKTeq = S.dot(KKT.dot(S))
                    print "Norm of first row of KKT %.4e" % \
                        nplinalg.norm((KKT.todense())[1, :])
                    print "Norm of first row of KKTeq %.4e" % \
                        nplinalg.norm((KKTeq.todense())[1, :])
                    condKKT = nplinalg.cond(KKT.todense())
                    condKKTeq = nplinalg.cond(KKTeq.todense())
                    print "Condition number of KKT matrix %.4e" % condKKT
                    print "Condition number of KKTeq matrix %.4e" % condKKTeq
                    # ipdb.set_trace()
            #  else:  # Q matrix is zero (LP)
                #  print "Perform scaling of Ac constraints matrix: %i Steps\n" % \
                    #  self.options.scale_steps
                #  # Run scaling
                #  Ac2 = Ac.copy()
                #  Ac2.data = np.square(Ac2.data)
                #  d1 = np.ones(nconstr)
                #  d2 = np.ones(nx)

                #  # Iterate scaling
                #  for i in range(self.options.scale_steps):
                    #  #  print(np.max(Ac2.dot(d1)))
                    #  # Regularize components
                    #  Ac2d2 = np.minimum(np.maximum(Ac2.dot(d2), -1e+10), 1e+10)
                    #  # Avoid dividing by 0
                    #  d1 = np.reciprocal(Ac2d2 + 1e-10)
                    #  # Regularize components
                    #  Ac2Td1 =  np.minimum(np.maximum(Ac2.T.dot(d1), -1e+10),
                            #  1e+10)
                    #  # Avoid dividing by 0
                    #  d2 = np.reciprocal(Ac2Td1 + 1e-10)
                    #  # DEBUG STUFF
                    #  D1 = spspa.diags(d1)
                    #  D2 = spspa.diags(d2)
                    #  Aceq = D1.dot(Ac.dot(D2))
                    #  condAc = nplinalg.cond(Ac.todense())
                    #  condAceq = nplinalg.cond(Aceq.todense())
                    #  print "Condition number of Ac matrix %.4f" % condAc
                    #  print "Condition number of Aceq matrix %.4f" % condAceq
                #  d = np.append(d1, d2)

            # DEBUG STUFF
            # d = sp.rand(n + m)
            # d = 2.*np.ones(n + m)

            # Obtain Scaler Matrices
            d = np.power(d, 1./self.options.scale_norm)
            D = spspa.diags(d[:self.problem.n])
            if m == 0:
                # spspa.diags() will throw an error if fed with an empty array
                E = spspa.csc_matrix((0, 0))
            else:
                E = spspa.diags(d[self.problem.n:])
            #  E = spspa.diags(np.ones(self.problem.neq + self.problem.nineq))

            # Scale problem Matrices
            P = D.dot(self.problem.P.dot(D))
            A = E.dot(self.problem.A.dot(D))
            q = D.dot(self.problem.q)
            lA = E.dot(self.problem.lA)
            uA = E.dot(self.problem.uA)
            # lA = np.multiply(E.diagonal(), self.problem.lA)
            # uA = np.multiply(E.diagonal(), self.problem.uA)
            # lA = np.multiply(np.reciprocal(E.diagonal()),
            #                  self.problem.lA)
            # uA = np.multiply(np.reciprocal(E.diagonal()),
            #                  self.problem.uA)

            # Assign scaled problem
            self.scaled_problem = problem(P, q, A, lA, uA)

            # Assign scaler matrices
            self.scaler_matrices.D = D
            self.scaler_matrices.Dinv = \
                spspa.diags(np.reciprocal(D.diagonal()))
            self.scaler_matrices.E = E
            if m == 0:
                self.scaler_matrices.Einv = E
            else:
                self.scaler_matrices.Einv = \
                    spspa.diags(np.reciprocal(E.diagonal()))

            # ipdb.set_trace()
            #  # DEBUG STUFF
            #  Dinv = self.scaler_matrices.Dinv
            #  Einv = self.scaler_matrices.Einv
            #  #  Dinv = spspa.linalg.inv(D.tocsr())
            #  #  Einv = spspa.csr_matrix(spspa.linalg.inv(F.tocsr()))
            #  Qtest = Dinv.dot(Q.dot(Dinv))
            #  Actest = Einv.dot(Ac.dot(Dinv))
            #  ctest = Dinv.dot(c)
            #  btest = Einv.dot(b)
            #  lbtest = D.dot(lb)
            #  ubtest = D.dot(ub)
            #  ipdb.set_trace()
        else:
            d = np.ones(n + m)

            # Obtain Scaler Matrices
            self.scaler_matrices.D = spspa.diags(d[:self.problem.n])
            self.scaler_matrices.Dinv = self.scaler_matrices.D
            if m == 0:
                self.scaler_matrices.E = spspa.csc_matrix((0, 0))
            else:
                self.scaler_matrices.E = spspa.diags(np.ones(m))
            self.scaler_matrices.Einv = self.scaler_matrices.E

            # Assign scaled problem to same one
            self.scaled_problem = self.problem

    def project(self, xbar):
        """
        Project a vector xbar onto constraint set.
        """
        # Project only last m elements between bounds lA, uA
        xproj = np.minimum(np.maximum(xbar,
                           self.scaled_problem.lA), self.scaled_problem.uA)
        return xproj

    def polish(self):
        """
        Try to reconstruct the actual solution of a QP by guessing which
        constraints are active. The problem boils down to solving a linear
        system.
        """
        nx = self.scaled_problem.nx
        neq = self.scaled_problem.neq
        nineq = self.scaled_problem.nineq
        tol = self.options.polish_tol

        # Recover primal solution
        sol_x = self.solution.z[:nx]

        # Recover dual solution
        dual_vars = -self.options.rho * self.solution.u
        sol_ineq = dual_vars[nx+neq:]
        sol_lb = np.maximum(dual_vars[:nx], 0)
        sol_ub = -np.minimum(dual_vars[:nx], 0)

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
        ineq_act = sol_ineq > \
            (self.problem.bineq - self.problem.Aineq.dot(sol_x))
        ind_act = np.where(ineq_act)[0]                     # Aineq x = bineq
        ind_inact = np.where(np.logical_not(ineq_act))[0]   # Aineq x < bineq

        # Solve the corresponding linear system
        KKT = spspa.vstack([
            spspa.hstack([self.problem.Q, self.problem.Aeq.T,
                         self.problem.Aineq.T, spspa.eye(nx)]),
            spspa.hstack([self.problem.Aeq,
                         spspa.csr_matrix((neq, neq + nineq + nx))]),
            spspa.hstack([self.problem.Aineq[ind_act],
                          spspa.csr_matrix((len(ind_act), neq + nineq + nx))]),
            spspa.hstack([spspa.csr_matrix((len(ind_inact), nx + neq)),
                          spspa.eye(nineq).tocsr()[ind_inact],
                          spspa.csr_matrix((len(ind_inact), nx))]),
            spspa.hstack([spspa.csr_matrix((len(ind_free), nx + neq + nineq)),
                          spspa.eye(nx).tocsr()[ind_free]]),
            spspa.hstack([spspa.eye(nx).tocsr()[ind_lb],
                          spspa.csr_matrix((len(ind_lb), neq + nineq + nx))]),
            spspa.hstack([spspa.eye(nx).tocsr()[ind_ub],
                          spspa.csr_matrix((len(ind_ub), neq + nineq + nx))]),
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

        # If the KKT matrix is singular, spsolve return an array of NaNs
        if any(np.isnan(pol_sol)):
            # Terminate
            print "Polishing failed. KKT matrix is singular."

        # Check if the above solution satisfies constraints
        pol_x = pol_sol[:nx]
        pol_dual_eq = pol_sol[nx:nx + neq]
        pol_dual_ineq = pol_sol[nx + neq:nx + neq + nineq]
        pol_dual_lb = np.zeros(nx)
        pol_dual_lb[ind_lb] = -pol_sol[nx + neq + nineq:][ind_lb]
        pol_dual_ub = np.zeros(nx)
        pol_dual_ub[ind_ub] = pol_sol[nx + neq + nineq:][ind_ub]
        if all(pol_x > self.problem.lb - tol)\
                and all(pol_x < self.problem.ub + tol)\
                and all(pol_dual_ineq > -tol)\
                and all(pol_dual_lb > -tol)\
                and all(pol_dual_ub > -tol):
                # Return the computed high-precision solution
                print "Polishing successful!"

                # Substitute new z and u in the solution
                # Get slack variable
                if self.scaled_problem.Aineq.shape[0]:  # Aineq not null
                    sineq = self.scaled_problem.bineq - \
                        self.scaled_problem.Aineq.dot(pol_x)
                else:
                    sineq = np.zeros(nineq)

                # Reconstruct primal variable
                self.solution.z = np.hstack([pol_x, np.zeros(neq), sineq])

                # reconstruct dual variable
                self.solution.u = -1./self.options.rho * \
                    np.hstack([pol_dual_lb - pol_dual_ub,
                              pol_dual_eq,
                              pol_dual_ineq])
                #  solution.x = pol_x
                #  solution.sol_dual_eq = pol_dual_eq
                #  solution.sol_dual_ineq = pol_dual_ineq
                #  solution.sol_dual_lb = pol_dual_lb
                #  solution.sol_dual_ub = pol_dual_ub
                #  solution.objval = self.problem.objval(pol_x)
        else:
            print "Polishing failed! Constraints not satisfied"

    def scale_solution(self):
        """
        Scale given solution with diagonal scaling
        """
        self.solution.z[:self.problem.n] = self.scaler_matrices.Dinv.dot(
                self.solution.z[:self.problem.n])
        self.solution.z[self.problem.n:] = self.scaler_matrices.Einv.dot(
                self.solution.z[self.problem.n:])
        self.solution.u = \
            self.scaler_matrices.Einv.dot(self.solution.u)

    def rescale_solution(self):
        """
        Rescale solution back to user-given units
        """
        self.solution.z[:self.problem.n] = \
            self.scaler_matrices.D.dot(self.solution.z[:self.problem.n])
        self.solution.z[self.problem.n:] = \
            self.scaler_matrices.E.dot(self.solution.z[self.problem.n:])
        self.solution.u = \
            self.scaler_matrices.E.dot(self.solution.u)

    def norm_pri_res(self, z):
        pri_res = np.minimum(z[self.scaled_problem.n:] - self.scaled_problem.lA, 0) + \
            np.maximum(z[self.scaled_problem.n:] - self.scaled_problem.uA, 0)
        return np.linalg.norm(pri_res)

    def norm_dua_res(self, z_prev, z, x):
        temp_vec = (2 - self.options.alpha)*z_prev - z - \
            (1 - self.options.alpha)*x
        dua_res = temp_vec[:self.scaled_problem.n] + \
            self.scaled_problem.A.T.dot(temp_vec[self.scaled_problem.n:])
        dua_res *= self.options.rho
        return np.linalg.norm(dua_res)

    def solve_admm(self):
        """"
        Solve splitting problem with splitting of the form (after introducing
        slack variables for both equality and inequality constraints)
            minimize	1/2 x' Pbar x + qbar'x  + I_{Abar z=0}(z) + I_Z(z)
            subject to	x == z
        where Z = R^n x [lA, uA]^{m}.
        """

        # Print parameters
        #  print "Splitting method 2"
        #  print "KKT solution: " + self.options.kkt_method + "\n"

        # Number of variables (x,s1,s2) and constraints
        n = self.scaled_problem.n
        m = self.scaled_problem.m

        # Factorize KKT matrix if this has not been done yet
        if self.kkt_factor is None:
            # Construct reduced KKT matrix
            KKT = spspa.vstack([
                spspa.hstack([self.scaled_problem.P + self.options.rho *
                              spspa.eye(n), self.scaled_problem.A.T]),
                spspa.hstack([self.scaled_problem.A,
                              -1./self.options.rho * spspa.eye(m)])])
            self.kkt_factor = spalinalg.splu(KKT.tocsc())
            # if self.options.kkt_dir_reuse_factor:
            #     # Store factorization
            #     self.kkt_factor = kkt_factor

        # # Construct augmented b vector
        # bc = np.append(self.scaled_problem.beq, self.scaled_problem.bineq)

        # Set initial conditions
        if self.options.warm_start and self.solution.z is not None \
                and self.solution.u is not None:
            self.scale_solution()
            z = self.solution.z
            u = self.solution.u
        else:
            z = np.zeros(n + m)
            u = np.zeros(m)

        if self.options.print_level > 1:
            print "Iter \t  Objective       \tPrim Res \tDual Res"

        # Run ADMM: alpha \in (0, 2) is a relaxation parameter.
        #           Nominal ADMM is obtained for alpha=1.0
        for i in xrange(self.options.max_iter):
            # x update
            rhs = np.append(self.options.rho * z[:n] -
                            self.scaled_problem.q, z[n:] - u)
            sol_kkt = self.kkt_factor.solve(rhs)
            x = np.append(sol_kkt[:n], z[n:] - u +
                          1./self.options.rho * sol_kkt[n:])
            # z update
            z_old = np.copy(z)
            z[:n] = self.options.alpha*x[:n] + \
                (1.-self.options.alpha)*z_old[:n]  # First part not projected
            z[n:] = self.project(self.options.alpha*x[n:] +
                                 (1.-self.options.alpha)*z_old[n:] + u)
            # u update
            u = u + self.options.alpha*x[n:] + \
                (1.-self.options.alpha)*z_old[n:] - z[n:]

            # # DEBUG
            # print "x = "
            # print x
            # print "z = "
            # print z
            # print "u = "
            # print u

            # Compute primal and dual residuals
            # ipdb.set_trace()
            norm_pri_res = self.norm_pri_res(z)
            norm_dua_res = self.norm_dua_res(z_old, z, x)

            # Check the stopping criterion
            eps_prim = np.sqrt(m) * self.options.eps_abs \
                + self.options.eps_rel * np.max([np.linalg.norm(x[n:]),
                                                 np.linalg.norm(
                                                    self.scaled_problem.lA),
                                                 np.linalg.norm(
                                                    self.scaled_problem.uA)])
            eps_dual = np.sqrt(n) * self.options.eps_abs + \
                self.options.eps_rel * self.options.rho * \
                np.linalg.norm(self.scaled_problem.A.T.dot(u))

            if norm_pri_res <= eps_prim and norm_dua_res <= eps_dual:
                # Print the progress in last iterations
                if self.options.print_level > 1:
                    f = self.scaled_problem.objval(z[:n])
                    print "%4s \t % 1.7e  \t%1.2e  \t%1.2e" \
                        % (i+1, f, norm_pri_res, norm_dua_res)
                # Stop the algorithm
                break

            # Print cost function depending on print level
            if self.options.print_level > 1:
                if (i + 1 == 1) | (self.options.print_level == 2) & \
                        (np.mod(i + 1,
                         np.floor(np.float(self.options.max_iter)/20.0)) == 0)\
                        | (self.options.print_level == 3):
                            f = self.scaled_problem.objval(z[:n])
                            print "%4s \t % 1.7e  \t%1.2e  \t%1.2e" \
                                % (i+1, f, norm_pri_res, norm_dua_res)

        # Total iterations
        self.total_iter = i

        # Return status
        print "\n"
        if i == self.options.max_iter - 1:
            print "Maximum number of iterations exceeded!"
            self.status = MAXITER_REACHED
        else:
            print "Optimal solution found"
            self.status = OPTIMAL

        # Save z and u solution
        self.solution.z = z
        self.solution.u = u
