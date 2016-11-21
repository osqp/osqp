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

    def __init__(self, status, objval, # pri_res, dua_res,
                 x, dual, cputime, total_iter):
        self.status = status
        self.objval = objval
        # self.pri_res = pri_res
        # self.dua_res = dua_res
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
    eps_abs  [1e-06]           - Absolute tolerance
    eps_rel  [1e-06]           - Relative tolerance
    delta    [1e-07]           - Regularization parameter for polishing
    print_level [2]            - Printing level
    scaling  [False]           - Prescaling/Equilibration
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
        self.delta = kwargs.pop('delta', 1e-07)
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
        self.x = None
        self.u = None
        self.pri_res = None
        self.dua_res = None


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
        self.scaled_problem.q = self.scaler_matrices.D.dot(self.problem.q)

        self.problem.lA = kwargs.pop('lA', self.problem.lA)
        self.scaled_problem.lA = self.scaler_matrices.E.dot(self.problem.lA)

        self.problem.uA = kwargs.pop('uA', self.problem.uA)
        self.scaled_problem.uA = self.scaler_matrices.E.dot(self.problem.uA)


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
        self.options.delta = kwargs.pop('delta', self.options.delta)

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
        if (self.status == OPTIMAL) & self.options.polish:
            self.polish()

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
        sol_x = self.solution.x[:self.problem.n]
        objval = self.problem.objval(sol_x)

        # Recover dual solution
        dual = self.options.rho * self.solution.u

        # # Compute residuals
        # pri_res = self.norm_pri_res(self.solution.x)
        # dua_res = self.norm_dua_res2(sol_x, dual)

        # Store solution as a quadprogResults object
        solution = results(self.status, objval, # pri_res, dua_res,
                           sol_x, dual, self.cputime, self.total_iter)

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

            #  ipdb.set_trace()


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
                    #  condKKT = nplinalg.cond(KKT.todense1())
                    #  condKKTeq = nplinalg.cond(KKTeq.todense())
                    #  print "Condition number of KKT matrix %.4e" % condKKT
                    #  print "Condition number of KKTeq matrix %.4e" % condKKTeq
                    #  ipdb.set_trace()

                print "Perform symmetric scaling of KKT matrix: %i Steps\n" % \
                    self.options.scale_steps


                # condKKT = nplinalg.cond(KKT.todense())
                # print "Condition number of KKT matrix %.4e" % condKKT

                # Iterate Scaling
                for i in range(self.options.scale_steps):
                    #  print np.max(KKT2.dot(d))
                    #  ipdb.set_trace()
                    # Regularize components
                    KKT2d = KKT2.dot(d)
                    # Prevent division by 0
                    d = (n + m)*np.reciprocal(KKT2d + 1e-08)
                    # ipdb.set_trace()
                    # Prevent too large elements
                    d = np.minimum(np.maximum(d, -1e+10), 1e+10)
                    #  d = np.reciprocal(KKT2d)
                    # print "Scaling step %i\n" % i

                    # # DEBUG STUFF
                    # S = spspa.diags(d)
                    # KKTeq = S.dot(KKT.dot(S))
                    # #  print "Norm of first row of KKT %.4e" % \
                    #     #  nplinalg.norm((KKT.todense())[1, :])
                    # #  print "Norm of first row of KKTeq %.4e" % \
                    #     #  nplinalg.norm((KKTeq.todense())[1, :])
                    # condKKTeq = nplinalg.cond(KKTeq.todense())
                    # print "Condition number of KKTeq matrix %.4e" % condKKTeq
                    #  ipdb.set_trace()
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

        n = self.scaled_problem.n
        m = self.scaled_problem.m

        # Recover Ax and lambda from the ADMM solution
        sol_Ax = self.solution.x[n:]
        sol_lambda = self.options.rho * self.solution.u

        # Try to guess the active bounds
        ind_lAct = np.where(sol_Ax - self.scaled_problem.lA < -sol_lambda)[0]
        ind_uAct = np.where(self.scaled_problem.uA - sol_Ax < sol_lambda)[0]

        # Solve the corresponding linear system
        Ared = spspa.vstack([self.scaled_problem.A[ind_lAct],
                             self.scaled_problem.A[ind_uAct]])
        mred = Ared.shape[0]
        KKT = spspa.vstack([
                spspa.hstack([self.scaled_problem.P + self.options.delta *
                              spspa.eye(n), Ared.T]),
                spspa.hstack([Ared, -self.options.delta *
                              spspa.eye(mred)])]).tocsr()
        rhs = np.hstack([-self.scaled_problem.q,
                         self.scaled_problem.lA[ind_lAct],
                         self.scaled_problem.uA[ind_uAct]])
        try:
            pol_sol = spalinalg.spsolve(KKT, rhs)
        except:
            # Failed to factorize KKT matrix
            print "Polishing failed. Failed to factorize KKT matrix."
            return

        # If the KKT matrix is singular, spsolve return an array of NaNs
        if any(np.isnan(pol_sol)):
            # Terminate
            print "Polishing failed. KKT matrix is singular."
            return

        # Recover primal and dual polished solution
        pol_x = pol_sol[:n]
        pol_Ax = self.scaled_problem.A.dot(pol_x)
        pol_lambda = np.zeros(m)
        pol_lambda[ind_lAct] = pol_sol[n:n + len(ind_lAct)]
        pol_lambda[ind_uAct] = pol_sol[n + len(ind_lAct):]

        # Compute primal and dual residuals
        pol_pri_res = np.minimum(pol_Ax - self.scaled_problem.lA, 0) + \
            np.maximum(pol_Ax - self.scaled_problem.uA, 0)
        pol_dua_res = self.scaled_problem.P.dot(pol_x) + \
            self.scaled_problem.q + Ared.T.dot(pol_sol[n:])
        pol_pri_res_norm = np.linalg.norm(pol_pri_res)
        pol_dua_res_norm = np.linalg.norm(pol_dua_res)

        # Check if polishing was successful
        if pol_pri_res_norm < self.solution.pri_res and \
            pol_dua_res_norm < self.solution.pri_res:
                self.solution.x[:self.problem.n] = pol_x
                self.solution.u = pol_lambda / self.options.rho
                if self.options.print_level > 1:
                    f = self.scaled_problem.objval(pol_x)
                    print "PLSH \t % 1.7e  \t%1.2e  \t%1.2e\n" \
                        % (f, pol_pri_res_norm, pol_dua_res_norm)

    def scale_solution(self):
        """
        Scale given solution with diagonal scaling
        """
        self.solution.x[:self.problem.n] = self.scaler_matrices.Dinv.dot(
                self.solution.x[:self.problem.n])
        self.solution.x[self.problem.n:] = self.scaler_matrices.Einv.dot(
                self.solution.x[self.problem.n:])
        self.solution.u = \
            self.scaler_matrices.Einv.dot(self.solution.u)

    def rescale_solution(self):
        """
        Rescale solution back to user-given units
        """
        self.solution.x[:self.problem.n] = \
            self.scaler_matrices.D.dot(self.solution.x[:self.problem.n])
        self.solution.x[self.problem.n:] = \
            self.scaler_matrices.E.dot(self.solution.x[self.problem.n:])
        self.solution.u = \
            self.scaler_matrices.E.dot(self.solution.u)

    def norm_pri_res(self, x):
        pri_res = np.minimum(x[self.scaled_problem.n:] -
                             self.scaled_problem.lA, 0) + \
            np.maximum(x[self.scaled_problem.n:] - self.scaled_problem.uA, 0)
        return np.linalg.norm(pri_res)

    def norm_dua_res(self, z_prev, z, x):
        temp_vec = (2 - self.options.alpha)*z_prev - z - \
            (1 - self.options.alpha)*x
        dua_res = temp_vec[:self.scaled_problem.n] + \
            self.scaled_problem.A.T.dot(temp_vec[self.scaled_problem.n:])
        dua_res *= self.options.rho
        return np.linalg.norm(dua_res)

    def norm_dua_res2(self, x, lmbd):
        dua_res = self.scaled_problem.P.dot(x) + \
                  self.scaled_problem.q + self.scaled_problem.A.T.dot(lmbd)
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
        if self.options.warm_start and self.solution.x is not None \
                and self.solution.u is not None:
            self.scale_solution()
            z = self.solution.x
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

            # Compute primal and dual residuals
            norm_pri_res = self.norm_pri_res(x)
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

            if norm_pri_res < eps_prim and norm_dua_res < eps_dual:
                # Print the progress in last iterations
                if self.options.print_level > 1:
                    f = self.scaled_problem.objval(x[:n])
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
        # print "\n"
        if i == self.options.max_iter - 1:
            # print "Maximum number of iterations exceeded!"
            self.status = MAXITER_REACHED
        else:
            # print "Optimal solution found"
            self.status = OPTIMAL

        # Save z and u solution
        self.solution.x = x
        self.solution.u = u
        self.solution.pri_res = norm_pri_res
        self.solution.pri_res = norm_dua_res
