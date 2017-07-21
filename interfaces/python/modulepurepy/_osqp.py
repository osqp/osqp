"""
OSQP Solver pure python implementation: low level module
"""
from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
import pdb   # Debugger

# Solver Constants
OSQP_SOLVED = 1
OSQP_MAX_ITER_REACHED = -2
OSQP_PRIMAL_INFEASIBLE = -3
OSQP_DUAL_INFEASIBLE = -4
OSQP_UNSOLVED = -10

# Printing interval
PRINT_INTERVAL = 100

# OSQP Infinity
OSQP_INFTY = 1e+20

# OSQP Nan
OSQP_NAN = 1e+20  # Just as placeholder. Not real value

# Auto rho (only linear function)
# (old values)
# AUTO_RHO_OFFSET = 1.07838081E-03
# AUTO_RHO_SLOPE = 2.31511262

#  AUTO_RHO_OFFSET = 0.0
#  AUTO_RHO_SLOPE = 2.4474028467925546

# Old (More complicated fit)
# AUTO_RHO_BETA0 = 2.1877627217504649
# AUTO_RHO_BETA1 = 0.57211669508170027
# AUTO_RHO_BETA2 = -0.71622847416411806

# New
# AUTO_RHO_BETA0 = 2.9287845053694284
# AUTO_RHO_BETA1 = 0.36309102743796728
# AUTO_RHO_BETA2 = -0.59595092004777972

# With regularization on v (1e-03)
# AUTO_RHO_BETA0 = 4.7967320978661059
# AUTO_RHO_BETA1 = -0.014575489596745135
# AUTO_RHO_BETA2 = -0.38435004096785225

# With regularization on v (1e-04)
#  AUTO_RHO_BETA0 = 3.1723875550135223
#  AUTO_RHO_BETA1 = 0.29811867735531827
#  AUTO_RHO_BETA2 = -0.55976668580992439

# With iter less than 20
# AUTO_RHO_BETA0 = 6.4047899119449241
# AUTO_RHO_BETA1 = -0.67858183687448892
# AUTO_RHO_BETA2 = -0.037034234262609413





# No regularization. interval [1, 1.2] (depends on traces)
#  AUTO_RHO_BETA0 = 2.2377322735057317
#  AUTO_RHO_BETA1 = 0.73909558577990619
#  AUTO_RHO_BETA2 = -0.81428271821694909


# only n and m, interval 2.0
#  AUTO_RHO_BETA0 = 132.31670550204416
#  AUTO_RHO_BETA1 = 3.6821990789623533
#  AUTO_RHO_BETA2 = -5.3493318062852033

#  # only n and m, interval 2.0
#  AUTO_RHO_BETA0 = 99.46657940983809
#  AUTO_RHO_BETA1 = 3.7594273667640685
#  AUTO_RHO_BETA2 = -5.3658885099270002

# only n and m, interval 1.5
#  AUTO_RHO_BETA0 = 63.9204222926816
#  AUTO_RHO_BETA1 = 4.2480325664123226
#  AUTO_RHO_BETA2 = -5.7924560461638848


#  AUTO_RHO_BETA0 = 1.0865058613182395
#  AUTO_RHO_BETA1 = 0.12750326228757933
#  AUTO_RHO_BETA2 = -0.65234442259175496

AUTO_RHO_BETA0 = 0.43764484761141698
AUTO_RHO_BETA1 = 0.26202391082629206
AUTO_RHO_BETA2 = -0.46598879917320213



AUTO_RHO_MAX = 1e06
AUTO_RHO_MIN = 1e-06


# Scaling
SCALING_REG = 1e-08
#  MAX_SCALING = 1e06
#  MIN_SCALING = 1e-06


class workspace(object):
    """
    OSQP solver workspace

    Attributes
    ----------
    data                   - scaled QP problem
    info                   - solver information
    priv                   - private structure for linear system solution
    scaling                - scaling matrices
    settings               - settings structure
    solution               - solution structure


    Additional workspace variables
    -------------------------------
    first_run              - flag to indicate if it is the first run
    timer                  - saved time instant for timing purposes
    x                      - primal iterate
    x_prev                 - previous primal iterate
    xz_tilde               - x_tilde and z_tilde iterates stacked together
    y                      - dual iterate
    z                      - z iterate
    z_prev                 - previous z iterate

    Primal infeasibility related workspace variables
    -----------------------------------------
    delta_y                - difference of consecutive y
    Atdelta_y              - A' * delta_y

    Dual infeasibility related workspace variables
    -----------------------------------------
    delta_x                - difference of consecutive x
    Pdelta_x               - P * delta_x
    Adelta_x               - A * delta_x

    """


class problem(object):
    """
    QP problem of the form
        minimize	1/2 x' P x + q' x
        subject to	l <= A x <= u

    Attributes
    ----------
    P, q
    A, l, u
    """

    def __init__(self, dims, Pdata, Pindices, Pindptr, q,
                 Adata, Aindices, Aindptr,
                 l, u):
        # Set problem dimensions
        (self.n, self.m) = dims

        # Set problem data
        self.P = spspa.csc_matrix((Pdata, Pindices, Pindptr),
                                  shape=(self.n, self.n))
        self.q = q
        self.A = spspa.csc_matrix((Adata, Aindices, Aindptr),
                                  shape=(self.m, self.n))
        self.l = l if l is not None else -np.inf*np.ones(self.m)
        self.u = u if u is not None else np.inf*np.ones(self.m)

    def objval(self, x):
        # Compute quadratic objective value for the given x
        return .5 * np.dot(x, self.P.dot(x)) + np.dot(self.q, x)


class settings(object):
    """
    OSQP solver settings

    Attributes
    ----------
    -> These cannot be changed without running setup
    rho  [1.6]                 - Step in ADMM procedure
    sigma    [1e-06]           - Regularization parameter for polish
    scaling  [True]            - Prescaling/Equilibration
    scaling_iter [3]           - Number of Steps for Scaling Method

    -> These can be changed without running setup
    max_iter [5000]                     - Maximum number of iterations
    eps_abs  [1e-05]                    - Absolute tolerance
    eps_rel  [1e-05]                    - Relative tolerance
    eps_prim_inf  [1e-06]                    - Primal infeasibility tolerance
    eps_dual_inf  [1e-06]                    - Dual infeasibility tolerance
    alpha [1.6]                         - Relaxation parameter
    delta [1.0]                         - Regularization parameter for polish
    verbose  [True]                     - Verbosity
    scaled_termination [False]             - Evalute scaled termination criteria
    early_terminate  [True]             - Evalute termination criteria
    early_terminate_interval  [25]    - Interval for evaluating termination criteria
    warm_start [False]                  - Reuse solution from previous solve
    polish  [True]                      - Solution polish
    pol_refine_iter  [3]                - Number of iterative refinement iterations
    auto_rho  [True]                    - Automatic rho computation
    """

    def __init__(self, **kwargs):

        self.rho = kwargs.pop('rho', 0.1)
        self.sigma = kwargs.pop('sigma', 1e-06)
        self.scaling = kwargs.pop('scaling', True)
        self.scaling_iter = kwargs.pop('scaling_iter', 15)
        self.max_iter = kwargs.pop('max_iter', 2500)
        self.eps_abs = kwargs.pop('eps_abs', 1e-3)
        self.eps_rel = kwargs.pop('eps_rel', 1e-3)
        self.eps_prim_inf = kwargs.pop('eps_prim_inf', 1e-4)
        self.eps_dual_inf = kwargs.pop('eps_dual_inf', 1e-4)
        self.alpha = kwargs.pop('alpha', 1.6)
        self.delta = kwargs.pop('delta', 1e-6)
        self.verbose = kwargs.pop('verbose', True)
        self.scaled_termination = kwargs.pop('scaled_termination', False)
        self.early_terminate = kwargs.pop('early_terminate', True)
        self.early_terminate_interval = kwargs.pop('early_terminate_interval', 25)
        self.warm_start = kwargs.pop('warm_start', False)
        self.polish = kwargs.pop('polish', True)
        self.pol_refine_iter = kwargs.pop('pol_refine_iter', 3)
        self.auto_rho = kwargs.pop('auto_rho', True)



class scaling(object):
    """
    Matrices for diagonal scaling

    Attributes
    ----------
    D     - matrix in R^{n \\times n}
    E     - matrix in R^{m \\times n}
    Dinv  - inverse of D
    Einv  - inverse of E
    """
    def __init__(self):
        self.D = None
        self.E = None
        self.Dinv = None
        self.Einv = None


class solution(object):
    """
    Solver solution vectors z, u
    """
    def __init__(self):
        self.x = None
        self.y = None


class info(object):
    """
    Solver information

    Attributes
    ----------
    iter            - number of iterations taken
    status          - status string, e.g. 'Solved'
    status_val      - status as c_int, defined in constants.h
    status_polish   - polish status: successful (1), not (0)
    obj_val         - primal objective
    pri_res         - norm of primal residual
    dua_res         - norm of dual residual
    setup_time      - time taken for setup phase (milliseconds)
    solve_time      - time taken for solve phase (milliseconds)
    polish_time     - time taken for polish phase (milliseconds)
    run_time        - total time  (milliseconds)
    """
    def __init__(self):
        self.iter = 0
        self.status_val = OSQP_UNSOLVED
        self.status_polish = 0
        self.polish_time = 0.0


class pol(object):
    """
    Polishing structure containing active constraints at the solution

    Attributes
    ----------
    ind_low         - indices of lower-active constraints
    ind_upp         - indices of upper-active constraints
    n_low           - number of lower-active constraints
    n_upp           - number of upper-active constraints
    Ared            - Part of A containing only active rows
    x               - polished x
    z               - polished z
    y_red           - polished y (part corresponding to active constraints)
    """
    def __init__(self):
        self.ind_low = None
        self.ind_upp = None
        self.n_low = None
        self.n_upp = None
        self.Ared = None
        self.x = None
        self.z = None
        self.y_red = None


class priv(object):
    """
    Linear systems solver
    """

    def __init__(self, work):
        """
        Initialize private structure for KKT system solution
        """
        # Construct reduced KKT matrix
        KKT = spspa.vstack([
              spspa.hstack([work.data.P + work.settings.sigma *
                            spspa.eye(work.data.n), work.data.A.T]),
              spspa.hstack([work.data.A,
                           -1./work.settings.rho * spspa.eye(work.data.m)])])

        # Initialize private structure
        self.kkt_factor = spla.splu(KKT.tocsc())

    def solve(self, rhs):
        """
        Solve linear system with given factorization
        """
        return self.kkt_factor.solve(rhs)


class results(object):
    """
    Results structure

    Attributes
    ----------
    x           - primal solution
    y           - dual solution
    info        - info structure
    """
    def __init__(self, solution, info):
        self.x = solution.x
        self.y = solution.y
        self.info = info


class OSQP(object):
    """OSQP solver lower level interface
    Attributes
    ----------
    work    - workspace
    """
    def __init__(self):
        self._version = "0.1.2"

    @property
    def version(self):
        """Return solver version
        """
        return self._version

    def scale_data(self):
        """
        Perform symmetric diagonal scaling via equilibration
        """
        n = self.work.data.n
        m = self.work.data.m

        # Initialize scaling
        d = np.ones(n + m)
        d_temp = np.ones(n + m)

        # Define reduced KKT matrix to scale
        KKT = spspa.vstack([
              spspa.hstack([self.work.data.P, self.work.data.A.T]),
              spspa.hstack([self.work.data.A,
                            spspa.csc_matrix((m, m))])]).tocsc()

        # Iterate Scaling
        for i in range(self.work.settings.scaling_iter):

            # Ruiz equilibration
            for j in range(n + m):
                norm_col_j = np.linalg.norm(np.asarray(KKT[:, j].todense()),
                                            np.inf)
                #  print("norm col %i = %.4e" % (j, norm_col_j))
                #  norm_row_j = np.linalg.norm(KKT_temp[j, :].A1, np.inf)
                #  print("norm row %i = %.4e" % (j, norm_row_j))
                #  norm_col_j = np.linalg.norm(KKT_temp[j, :], np.inf)
                #  norm_col_j = np.linalg.norm(KKT_temp[j, :], 2)
                #  norm_col_j = np.linalg.norm(KKT_temp[j, :], 1)
                #  d[j] = 1./(np.sqrt(norm_col_j) + SCALING_REG)
                if norm_col_j > SCALING_REG:
                    d_temp[j] = 1./(np.sqrt(norm_col_j))

            S_temp = spspa.diags(d_temp)
            d = np.multiply(d, d_temp)
            KKT = S_temp.dot(KKT.dot(S_temp))

            #  # DEBUG: Check scaling
            #  D = spspa.diags(d[:n])
            #
            #  if m == 0:
            #      # spspa.diags() will throw an error if fed with an empty array
            #      E = spspa.csc_matrix((0, 0))
            #      A = \
            #          E.dot(self.work.data.A.dot(D)).todense()
            #      cond_A = 1.
            #  else:
            #      E = spspa.diags(d[n:])
            #      A = \
            #          E.dot(self.work.data.A.dot(D)).todense()
            #      cond_A = np.linalg.cond(A)
            #  cond_KKT = np.linalg.cond(KKT.todense())
            #  P = \
            #      D.dot(self.work.data.P.dot(D)).todense()
            #  cond_P = np.linalg.cond(P)
            #
            #  # Get rato between columns and rows
            #  n_plus_m = n + m
            #  max_norm_rows = 0.0
            #  min_norm_rows = np.inf
            #  for j in range(n_plus_m):
            #     norm_row_j = np.linalg.norm(np.asarray(KKT[j, :].todense()))
            #     max_norm_rows = np.maximum(norm_row_j,
            #                                max_norm_rows)
            #     min_norm_rows = np.minimum(norm_row_j,
            #                                min_norm_rows)
            #
            #  # Compute residuals
            #  res_rows = max_norm_rows / min_norm_rows
            #
            #  np.set_printoptions(suppress=True, linewidth=500, precision=3)
            #  print("\nIter %i" % i)
            #  print("cond(KKT) = %.4e" % cond_KKT)
            #  print("cond(P) = %.4e" % cond_P)
            #  print("cond(A) = %.4e" % cond_A)
            #  print("res_rows = %.4e / %.4e = %.4e" %
            #       (max_norm_rows, min_norm_rows, res_rows))
            #
            #
        # Obtain Scaler Matrices
        D = spspa.diags(d[:self.work.data.n])
        if m == 0:
            # spspa.diags() will throw an error if fed with an empty array
            E = spspa.csc_matrix((0, 0))
        else:
            E = spspa.diags(d[self.work.data.n:])


        # Scale problem Matrices
        P = D.dot(self.work.data.P.dot(D)).tocsc()
        A = E.dot(self.work.data.A.dot(D)).tocsc()
        q = D.dot(self.work.data.q)
        l = E.dot(self.work.data.l)
        u = E.dot(self.work.data.u)

        #  import ipdb; ipdb.set_trace()

        # Assign scaled problem
        self.work.data = problem((n, m), P.data, P.indices, P.indptr, q,
                                 A.data, A.indices, A.indptr, l, u)

        # Assign scaling matrices
        self.work.scaling = scaling()
        self.work.scaling.D = D
        self.work.scaling.Dinv = \
            spspa.diags(np.reciprocal(D.diagonal()))
        self.work.scaling.E = E
        if m == 0:
            self.work.scaling.Einv = E
        else:
            self.work.scaling.Einv = \
                spspa.diags(np.reciprocal(E.diagonal()))

    def compute_rho(self):
        """
        Automatically compute rho value
        """

        if self.work.data.m == 0:
            self.work.settings.rho = AUTO_RHO_MAX

        n = self.work.data.n
        m = self.work.data.m
        sigma = self.work.settings.sigma
        #  self.work.settings.rho = AUTO_RHO_BETA0 * \
        #      np.power(n, AUTO_RHO_BETA1) * \
        #      np.power(m, AUTO_RHO_BETA2)


        #  Old stuff with traces
        # Compute tr(P)
        trP = self.work.data.P.diagonal().sum()

        #  Compute tr(AtA) = fro(A) ^ 2
        trAtA = spspa.linalg.norm(self.work.data.A) ** 2
#

        # DEBUG: saturate values of norms
        # trP = np.maximum(trP, 1e-03)
        # trAtA = np.maximum(trAtA, 1e-03)

        #  self.work.settings.rho = AUTO_RHO_BETA0 * \
        #      np.power(trP, AUTO_RHO_BETA1) * \
        #      np.power(trAtA, AUTO_RHO_BETA2)


        self.work.settings.rho = AUTO_RHO_BETA0 * \
                np.power((trP + sigma * n)/n, AUTO_RHO_BETA1) * \
                np.power((trAtA)/m, AUTO_RHO_BETA2)

        #  import ipdb; ipdb.set_trace()
        # Old linear ratio
        # # Compute ratio
        # ratio = trP / trAtA
        #
        # # Compute rho
        # self.work.settings.rho = AUTO_RHO_OFFSET + AUTO_RHO_SLOPE * ratio

        # Constrain rho between max and min
        self.work.settings.rho = \
            np.minimum(np.maximum(self.work.settings.rho,
                                  AUTO_RHO_MIN),
                       AUTO_RHO_MAX)

    def print_setup_header(self, data, settings):
        """Print solver header
        """
        print("-------------------------------------------------------")
        print("      OSQP v%s  -  Operator Splitting QP Solver" % \
            self.version)
        print("              Pure Python Implementation")
        print("     (c) Bartolomeo Stellato, Goran Banjac")
        print("   University of Oxford  -  Stanford University 2017")
        print("-------------------------------------------------------")

        print("Problem:  variables n = %d, constraints m = %d" % \
            (data.n, data.m))
        print("Settings: eps_abs = %.2e, eps_rel = %.2e," % \
            (settings.eps_abs, settings.eps_rel))
        print("          eps_prim_inf = %.2e, eps_dual_inf = %.2e," % \
            (settings.eps_prim_inf, settings.eps_dual_inf))
        print("          rho = %.2e " % settings.rho, end='')
        if settings.auto_rho:
            print("(auto)")
        else:
            print("")
        print("          sigma = %.2e, alpha = %.2e," % \
            (settings.sigma, settings.alpha))
        print("          max_iter = %d" % settings.max_iter)
        if settings.scaling:
            print("          scaling: on, ", end='')
        else:
            print("          scaling: off, ", end='')
        if settings.scaled_termination:
            print("scaled_termination: on")
        else:
            print("scaled_termination: off")
        if settings.warm_start:
            print("          warm_start: on, ", end='')
        else:
            print("          warm_start: off, ", end='')
        if settings.polish:
            print("polish: on")
        else:
            print("polish: off")
        print("")

    def print_header(self):
        """
        Print header before the iterations
        """
        print("Iter    Obj  Val     Pri  Res     Dua  Res       Time")

    def update_status_string(self, status):
        if status == OSQP_SOLVED:
            return "Solved"
        elif status == OSQP_PRIMAL_INFEASIBLE:
            return "Primal infeasible"
        elif status == OSQP_UNSOLVED:
            return "Unsolved"
        elif status == OSQP_DUAL_INFEASIBLE:
            return "Dual infeasible"
        elif status == OSQP_MAX_ITER_REACHED:
            return "Maximum iterations reached"

    def cold_start(self):
        """
        Cold start optimization variables to zero
        """
        self.work.x = np.zeros(self.work.data.n)
        self.work.z = np.zeros(self.work.data.m)
        self.work.y = np.zeros(self.work.data.m)

    def update_xz_tilde(self):
        """
        First ADMM step: update xz_tilde
        """
        # Compute rhs and store it in xz_tilde
        self.work.xz_tilde[:self.work.data.n] = \
            self.work.settings.sigma * self.work.x_prev - self.work.data.q
        self.work.xz_tilde[self.work.data.n:] = \
            self.work.z_prev - 1./self.work.settings.rho * self.work.y

        # Solve linear system
        self.work.xz_tilde = self.work.priv.solve(self.work.xz_tilde)

        # Update z_tilde
        self.work.xz_tilde[self.work.data.n:] = \
            self.work.z_prev + 1./self.work.settings.rho * \
            (self.work.xz_tilde[self.work.data.n:] - self.work.y)

    def update_x(self):
        """
        Update x variable in second ADMM step
        """
        self.work.x = \
            self.work.settings.alpha * self.work.xz_tilde[:self.work.data.n] +\
            (1. - self.work.settings.alpha) * self.work.x_prev
        self.work.delta_x = self.work.x - self.work.x_prev

    def project_z(self):
        """
        Project z variable in set C (for now C = [l, u])
        """
        # proj_z = np.minimum(np.maximum(self.work.z,
        #                     self.work.data.l), self.work.data.u)
        # set_trace()
        self.work.z = np.minimum(np.maximum(self.work.z,
                                 self.work.data.l), self.work.data.u)

    def update_z(self):
        """
        Update z variable in second ADMM step
        """
        self.work.z = \
            self.work.settings.alpha * self.work.xz_tilde[self.work.data.n:] +\
            (1. - self.work.settings.alpha) * self.work.z_prev +\
            1./self.work.settings.rho * self.work.y

        self.project_z()

    def update_y(self):
        """
        Third ADMM step: update dual variable y
        """
        self.work.delta_y = self.work.settings.rho * \
            (self.work.settings.alpha * self.work.xz_tilde[self.work.data.n:] +
                (1. - self.work.settings.alpha) * self.work.z_prev -
                self.work.z)
        self.work.y += self.work.delta_y

    def compute_pri_res(self, polish):
        """
        Compute primal residual ||Ax - z||
        """
        if self.work.data.m == 0:  # No constraints
            return 0.
        if polish:
            pri_res = np.maximum(self.work.data.l - self.work.pol.z, 0) + \
                np.maximum(self.work.pol.z - self.work.data.u, 0)
        else:
            pri_res = self.work.data.A.dot(self.work.x) - self.work.z

        if self.work.settings.scaling and not self.work.settings.scaled_termination:
            pri_res = self.work.scaling.Einv.dot(pri_res)

        return la.norm(pri_res, np.inf)

    def compute_dua_res(self, polish):
        """
        Compute dual residual ||Px + q + A'y||
        """
        if polish:
            dua_res = self.work.data.P.dot(self.work.pol.x) + \
                self.work.data.q +\
                self.work.pol.Ared.T.dot(self.work.pol.y_red)
        else:
            dua_res = self.work.data.P.dot(self.work.x) +\
                self.work.data.q +\
                self.work.data.A.T.dot(self.work.y)

        if self.work.settings.scaling and not self.work.settings.scaled_termination:
            dua_res = self.work.scaling.Dinv.dot(dua_res)

        return la.norm(dua_res, np.inf)

    def is_primal_infeasible(self):
        """
        Check primal infeasibility
                ||A'*v||_2 = 0
        with v = delta_y/||delta_y||_2 given that following condition holds
            u'*(v)_{+} + l'*(v)_{-} < 0
        """

        # # DEBUG
        # if (self.work.info.iter % PRINT_INTERVAL == 0):
        #     print "\n\nValues with r_pri"
        #     r_pri = self.work.z - self.work.data.A.dot(self.work.x)
        #
        #     lhs = 0.
        #     for i in range(self.work.data.m):
        #         if self.work.data.u[i] < self.constant('OSQP_INFTY')*1e-03:
        #             lhs += self.work.data.u[i] * max(r_pri[i], 0)
        #
        #         if self.work.data.l[i] > -self.constant('OSQP_INFTY')*1e-03:
        #             lhs += self.work.data.l[i] * min(r_pri[i], 0)
        #     # lhs = self.work.data.u.dot(np.maximum(r_pri, 0)) + \
        #     #     self.work.data.l.dot(np.minimum(r_pri, 0))
        #     print("u' * (v)_{+} + l' * v_{-} = %6.2e" % (lhs))
        #     Atr_pri = self.work.data.A.T.dot(r_pri)
        #     print("||A'*v|| = %6.2e" % (la.norm(Atr_pri)))
        #

            # lhsp = self.work.data.u.dot(np.maximum(self.work.delta_y, 0))
            # lhsm = self.work.data.l.dot(np.minimum(self.work.delta_y, 0))
            # print("Values with delta_y")
            # lhs = 0.
            # lhsp = 0.
            # lhsm = 0.
            # for i in range(self.work.data.m):
            #     if self.work.data.u[i] < self.constant('OSQP_INFTY')*1e-05:
            #         lhsp += self.work.data.u[i] * max(self.work.delta_y[i], 0)
            #
            #     if self.work.data.l[i] > -self.constant('OSQP_INFTY')*1e-03:
            #         lhsm += self.work.data.l[i] * min(self.work.delta_y[i], 0)
            # lhs = lhsp + lhsm
            # print("u' * (v_{+}) = %6.2e" % lhsp)
            # print("l' * (v_{-}) = %6.2e" % lhsm)
            # print("u' * (v_{+}) + l' * (v_{-}) = %6.2e" % (lhs))
            # self.work.Atdelta_y = self.work.data.A.T.dot(self.work.delta_y)
            # print("||A'*v|| = %6.2e" % (la.norm(self.work.Atdelta_y)))
            # pdb.set_trace()

        # Prevent 0 division
        # if la.norm(self.work.delta_y) > eps_prim_inf*eps_prim_inf:
        #     # self.work.delta_y /= la.norm(self.work.delta_y)
        #     # lhs = self.work.data.u.dot(np.maximum(self.work.delta_y, 0)) + \
        #     #     self.work.data.l.dot(np.minimum(self.work.delta_y, 0))
        #     # if  lhs < -eps_prim_inf:
        #     #     self.work.Atdelta_y = self.work.data.A.T.dot(self.work.delta_y)
        #     #     return la.norm(self.work.Atdelta_y) < eps_prim_inf

        eps_prim_inf = self.work.settings.eps_prim_inf
        norm_delta_y = la.norm(self.work.delta_y, np.inf)
        if norm_delta_y > eps_prim_inf:
            self.work.delta_y /= norm_delta_y
            lhs = self.work.data.u.dot(np.maximum(self.work.delta_y, 0)) + \
                    self.work.data.l.dot(np.minimum(self.work.delta_y, 0))
            if lhs < -eps_prim_inf:
                self.work.Atdelta_y = self.work.data.A.T.dot(self.work.delta_y)
                if self.work.settings.scaling and not self.work.settings.scaled_termination:
                        self.work.Atdelta_y = self.work.scaling.Dinv.dot(self.work.Atdelta_y)
                return la.norm(self.work.Atdelta_y, np.inf) < eps_prim_inf

        return False

    def is_dual_infeasible(self):
        """
        Check dual infeasibility
            ||P*v||_inf = 0
        with v = delta_x / ||delta_x||_inf given that the following
        conditions hold
            q'* v < 0 and
                        | 0     if l_i, u_i \in R
            (A * v)_i = { >= 0  if u_i = +inf
                        | <= 0  if l_i = -inf
        """
        eps_dual_inf = self.work.settings.eps_dual_inf
        norm_delta_x = la.norm(self.work.delta_x, np.inf)
        # Prevent 0 division
        if norm_delta_x > eps_dual_inf:
            # Normalize delta_x
            self.work.delta_x /= norm_delta_x

            # First check q'* delta_x < 0
            if self.work.data.q.dot(self.work.delta_x) < -eps_dual_inf:

                # Compute P * delta_x
                self.work.Pdelta_x = self.work.data.P.dot(self.work.delta_x)

                # Scale if necessary
                if self.work.settings.scaling and not self.work.settings.scaled_termination:
                    self.work.Pdelta_x = self.work.scaling.Dinv.dot(self.work.Pdelta_x)

                # Check if ||P * delta_x|| = 0
                if la.norm(self.work.Pdelta_x, np.inf) < eps_dual_inf:

                    # Compute A * delta_x
                    self.work.Adelta_x = self.work.data.A.dot(
                        self.work.delta_x)

                    # Scale if necessary
                    if self.work.settings.scaling and not self.work.settings.scaled_termination:
                        self.work.Adelta_x = self.work.scaling.Einv.dot(self.work.Adelta_x)

                    for i in range(self.work.data.m):
                        # De Morgan's Law applied to negate
                        # conditions on A * delta_x
                        if ((self.work.data.u[i] < OSQP_INFTY*1e-06) and
                            (self.work.Adelta_x[i] > eps_dual_inf)) or \
                            ((self.work.data.l[i] > -OSQP_INFTY*1e-06) and
                             (self.work.Adelta_x[i] < -eps_dual_inf)):

                            # At least one condition not satisfied
                            return False

                    # All conditions passed -> dual infeasible
                    return True

        # No all checks managed to pass. Problem not dual infeasible
        return False

    def update_info(self, iter, polish):
        """
        Update information at iterations
        """
        if polish == 1:
            self.work.pol.obj_val = self.work.data.objval(self.work.pol.x)
            self.work.pol.pri_res = self.compute_pri_res(1)
            self.work.pol.dua_res = self.compute_dua_res(1)
        else:
            self.work.info.iter = iter
            self.work.info.obj_val = self.work.data.objval(self.work.x)
            self.work.info.pri_res = self.compute_pri_res(0)
            self.work.info.dua_res = self.compute_dua_res(0)
            self.work.info.solve_time = time.time() - self.work.timer

    def print_summary(self):
        """
        Print status summary at each ADMM iteration
        """
        print("%4i %12.4e %12.4e %12.4e %9.2fs" % \
            (self.work.info.iter,
             self.work.info.obj_val,
             self.work.info.pri_res,
             self.work.info.dua_res,
             self.work.info.setup_time + self.work.info.solve_time))

    def print_polish(self):
        """
        Print polish information
        """
        print("PLSH %12.4e %12.4e %12.4e %9.2fs" % \
            (self.work.info.obj_val,
             self.work.info.pri_res,
             self.work.info.dua_res,
             self.work.info.setup_time + self.work.info.solve_time +
             self.work.info.polish_time))

    def check_termination(self):
        """
        Check residuals for algorithm convergence and update solver status
        """
        pri_check = 0
        dua_check = 0
        prim_inf_check = 0
        dual_inf_check = 0

        eps_abs = self.work.settings.eps_abs
        eps_rel = self.work.settings.eps_rel

        if self.work.data.m == 0:  # No constraints -> always  primal feasible
            pri_check = 1
        else:
            # Compute primal tolerance
            if self.work.settings.scaling and not self.work.settings.scaled_termination:
                Einv = self.work.scaling.Einv
                max_rel_eps = np.max([
                    la.norm(Einv.dot(self.work.data.A.dot(self.work.x)), np.inf),
                    la.norm(Einv.dot(self.work.z), np.inf)])
            else:
                max_rel_eps = np.max([
                    la.norm(self.work.data.A.dot(self.work.x), np.inf),
                    la.norm(self.work.z, np.inf)])

            eps_pri = eps_abs + eps_rel * max_rel_eps

            if self.work.info.pri_res < eps_pri:
                pri_check = 1
            else:
                # Check infeasibility
                prim_inf_check = self.is_primal_infeasible()

        # Compute dual tolerance

        if self.work.settings.scaling and not self.work.settings.scaled_termination:
            Dinv = self.work.scaling.Dinv
            max_rel_eps = np.max([
                la.norm(Dinv.dot(self.work.data.A.T.dot(self.work.y)), np.inf),
                la.norm(Dinv.dot(self.work.data.P.dot(self.work.x)), np.inf),
                la.norm(Dinv.dot(self.work.data.q), np.inf)])
        else:
            max_rel_eps = np.max([
                la.norm(self.work.data.A.T.dot(self.work.y), np.inf),
                la.norm(self.work.data.P.dot(self.work.x), np.inf),
                la.norm(self.work.data.q, np.inf)])

        eps_dua = eps_abs + eps_rel * max_rel_eps

        if self.work.info.dua_res < eps_dua:
            dua_check = 1
        else:
            # Check dual infeasibility
            dual_inf_check = self.is_dual_infeasible()

        # Compare residuals and determine solver status
        if pri_check & dua_check:
            self.work.info.status_val = OSQP_SOLVED
            return 1
        elif prim_inf_check:
            self.work.info.status_val = OSQP_PRIMAL_INFEASIBLE
            self.work.info.obj_val = OSQP_INFTY
            return 1
        elif dual_inf_check:
            self.work.info.status_val = OSQP_DUAL_INFEASIBLE
            self.work.info.obj_val = -OSQP_INFTY
            return 1

    def print_footer(self):
        """
        Print footer at the end of the optimization
        """
        print("")  # Add space after iterations
        print("Status: %s" % self.work.info.status)
        if self.work.settings.polish and \
                self.work.info.status_val == OSQP_SOLVED:
                    if self.work.info.status_polish == 1:
                        print("Solution polish: Successful")
                    elif self.work.info.status_polish == -1:
                        print("Solution polish: Unsuccessful")
        print("Number of iterations: %d" % self.work.info.iter)
        if self.work.info.status_val == OSQP_SOLVED:
            print("Optimal objective: %.4f" % self.work.info.obj_val)
        if self.work.info.run_time > 1e-03:
            print("Run time: %.3fs" % (self.work.info.run_time))
        else:
            print("Run time: %.3fms" % (1e03*self.work.info.run_time))


        print("")  # Print last space

    def store_solution(self):
        """
        Store current primal and dual solution in solution structure
        """

        if (self.work.info.status_val is not OSQP_PRIMAL_INFEASIBLE) and \
                (self.work.info.status_val is not OSQP_DUAL_INFEASIBLE):
            self.work.solution.x = self.work.x
            self.work.solution.y = self.work.y

            # Unscale solution
            if self.work.settings.scaling:
                self.work.solution.x = \
                    self.work.scaling.D.dot(self.work.solution.x)
                self.work.solution.y = \
                    self.work.scaling.E.dot(self.work.solution.y)
        else:
            self.work.solution.x = np.array([None] * self.work.data.n)
            self.work.solution.y = np.array([None] * self.work.data.m)

    #
    #   Main Solver API
    #

    def setup(self, dims, Pdata, Pindices, Pindptr, q,
              Adata, Aindices, Aindptr,
              l, u, **stgs):
        """
        Perform OSQP solver setup QP problem of the form
            minimize	1/2 x' P x + q' x
            subject to	l <= A x <= u

        """
        (n, m) = dims
        self.work = workspace()

        # Start timer
        self.work.timer = time.time()

        # Unscaled problem data
        self.work.data = problem((n, m), Pdata, Pindices, Pindptr, q,
                                 Adata, Aindices, Aindptr,
                                 l, u)

        # Initialize workspace variables
        self.work.x = np.zeros(n)
        self.work.z = np.zeros(m)
        self.work.xz_tilde = np.zeros(n + m)
        self.work.x_prev = np.zeros(n)
        self.work.z_prev = np.zeros(m)
        self.work.y = np.zeros(m)
        self.work.delta_y = np.zeros(m)       # Delta_y for primal infeasibility

        # Flag indicating first run
        self.work.first_run = 1

        # Settings
        self.work.settings = settings(**stgs)

        # Scale problem
        if self.work.settings.scaling:
            self.scale_data()

        # Compute auto_rho in case
        if self.work.settings.auto_rho:
            self.compute_rho()

        # Factorize KKT
        self.work.priv = priv(self.work)

        # Solution
        self.work.solution = solution()

        # Info
        self.work.info = info()

        # Polishing structure
        self.work.pol = pol()

        # End timer
        self.work.info.setup_time = time.time() - self.work.timer

        # Print setup header
        if self.work.settings.verbose:
            self.print_setup_header(self.work.data, self.work.settings)

    def solve(self):
        """
        Solve QP problem using OSQP
        """
        # Start timer
        self.work.timer = time.time()

        # Print header
        if self.work.settings.verbose:
            self.print_header()

        # Cold start if not warm start
        if not self.work.settings.warm_start:
            self.cold_start()

        # ADMM algorithm
        for iter in range(1, self.work.settings.max_iter + 1):
            # Update x_prev, z_prev
            self.work.x_prev = np.copy(self.work.x)
            self.work.z_prev = np.copy(self.work.z)

            # Admm steps
            # First step: update \tilde{x} and \tilde{z}
            self.update_xz_tilde()

            # Second step: update x and z
            self.update_x()

            self.update_z()

            # Third step: update y
            self.update_y()


            if self.work.settings.early_terminate:
                # Update info
                self.update_info(iter, 0)

                # Print summary
                if (self.work.settings.verbose) & \
                        ((iter % PRINT_INTERVAL == 0) | (iter == 1)):
                    self.print_summary()

                # Break if converged
                if self.check_termination():
                    break

        if not self.work.settings.early_terminate:
            # Update info
            self.update_info(self.work.settings.max_iter, 0)

            # Print summary
            if (self.work.settings.verbose):
                self.print_summary()

            # Break if converged
            self.check_termination()

        # Print summary for last iteration
        if (self.work.settings.verbose) & (iter % PRINT_INTERVAL != 0):
            self.print_summary()

        # If max iterations reached, update status accordingly
        if iter == self.work.settings.max_iter:
            self.work.info.status_val = OSQP_MAX_ITER_REACHED

        # Update status string
        self.work.info.status = \
            self.update_status_string(self.work.info.status_val)

        # Update solve time
        self.work.info.solve_time = time.time() - self.work.timer

        # Solution polish
        if self.work.settings.polish and \
                self.work.info.status_val == OSQP_SOLVED:
                    self.polish()

        # Update total times
        if self.work.first_run:
            self.work.info.run_time = self.work.info.setup_time + \
                self.work.info.solve_time + self.work.info.polish_time
        else:
            self.work.info.run_time = self.work.info.solve_time + \
                                      self.work.info.polish_time

        # Print footer
        if self.work.settings.verbose:
            self.print_footer()

        # Store solution
        self.store_solution()

        # Eliminate first run flag
        if self.work.first_run:
            self.work.first_run = 0

        # Store results structure
        return results(self.work.solution, self.work.info)

    #
    #   Auxiliary API Functions
    #

    def update_lin_cost(self, q_new):
        """
        Update linear cost without requiring factorization
        """
        # Copy cost vector
        self.work.data.q = np.copy(q_new)

        # Scaling
        if self.work.settings.scaling:
            self.work.data.q = self.work.scaling.D.dot(self.work.data.q)

    def update_bounds(self, l_new, u_new):
        """
        Update counstraint bounds without requiring factorization
        """

        # Check if bounds are correct
        if not np.greater_equal(u_new, l_new).all():
            raise ValueError("Lower bound must be lower than" +
                             " or equal to upper bound!")

        # Update vectors
        self.work.data.l = np.copy(l_new)
        self.work.data.u = np.copy(u_new)

        # Scale vectors
        if self.work.settings.scaling:
            self.work.data.l = self.work.scaling.E.dot(self.work.data.l)
            self.work.data.u = self.work.scaling.E.dot(self.work.data.u)

    def update_lower_bound(self, l_new):
        """
        Update lower bound without requiring factorization
        """
        # Update lower bound
        self.work.data.l = l_new

        # Scale vector
        if self.work.settings.scaling:
            self.work.data.l = self.work.scaling.E.dot(self.work.data.l)

        # Check values
        if not np.greater(self.work.data.u, self.work.data.l).all():
            raise ValueError("Lower bound must be lower than" +
                             " or equal to upper bound!")

    def update_upper_bound(self, u_new):
        """
        Update upper bound without requiring factorization
        """
        # Update upper bound
        self.work.data.u = u_new

        # Scale vector
        if self.work.settings.scaling:
            self.work.data.u = self.work.scaling.E.dot(self.work.data.u)

        # Check values
        if not np.greater(self.work.data.u, self.work.data.l).all():
            raise ValueError("Lower bound must be lower than" +
                             " or equal to upper bound!")

    def warm_start(self, x, y):
        """
        Warm start primal and dual variables
        """
        # Update warm_start setting to true
        self.work.settings.warm_start = True

        # Copy primal and dual variables into the iterates
        self.work.x = x
        self.work.y = y

        # Scale iterates
        self.work.x = self.work.scaling.Dinv.dot(self.work.x)
        self.work.y = self.work.scaling.Einv.dot(self.work.y)

        # Update z iterate as well
        self.work.z = self.work.data.A.dot(self.work.x)


    def warm_start_x(self, x):
        """
        Warm start primal variable
        """
        # Update warm_start setting to true
        self.work.settings.warm_start = True

        # Copy primal and dual variables into the iterates
        self.work.x = x

        # Scale iterates
        self.work.x = self.work.scaling.Dinv.dot(self.work.x)

        # Update z iterate as well
        self.work.z = self.work.data.A.dot(self.work.x)

        # Cold start y
        self.work.y = np.zeros(self.work.data.m)

    def warm_start_y(self, y):
        """
        Warm start dual variable
        """
        # Update warm_start setting to true
        self.work.settings.warm_start = True

        # Copy primal and dual variables into the iterates
        self.work.y = y

        # Scale iterates
        self.work.y = self.work.scaling.Einv.dot(self.work.y)

        # Cold start x and z
        self.work.x = np.zeros(self.work.data.n)
        self.work.z = np.zeros(self.work.data.m)


    #
    #   Update Problem Settings
    #
    def update_max_iter(self, max_iter_new):
        """
        Update maximum number of iterations
        """
        # Check that maxiter is positive
        if max_iter_new <= 0:
            raise ValueError("max_iter must be positive")

        # Update max_iter
        self.work.settings.max_iter = max_iter_new

    def update_eps_abs(self, eps_abs_new):
        """
        Update absolute tolerance
        """
        if eps_abs_new <= 0:
            raise ValueError("eps_abs must be positive")

        self.work.settings.eps_abs = eps_abs_new

    def update_eps_rel(self, eps_rel_new):
        """
        Update relative tolerance
        """
        if eps_rel_new <= 0:
            raise ValueError("eps_rel must be positive")

        self.work.settings.eps_rel = eps_rel_new

    def update_alpha(self, alpha_new):
        """
        Update relaxation parameter alpga
        """
        if not (alpha_new >= 0 | alpha_new <= 2):
            raise ValueError("alpha must be between 0 and 2")

        self.work.settings.alpha = alpha_new

    def update_delta(self, delta_new):
        """
        Update delta parameter for polish
        """
        if delta_new <= 0:
            raise ValueError("delta must be positive")

        self.work.settings.delta = delta_new

    def update_polish(self, polish_new):
        """
        Update polish parameter
        """
        if (polish_new is not True) & (polish_new is not False):
            raise ValueError("polish should be either True or False")

        self.work.settings.polish = polish_new
        self.work.info.polish_time = 0.0

    def update_pol_refine_iter(self, pol_refine_iter_new):
        """
        Update number iterative refinement iterations in polish
        """
        if pol_refine_iter_new < 0:
            raise ValueError("pol_refine_iter must be nonnegative")

        self.work.settings.pol_refine_iter = pol_refine_iter_new

    def update_verbose(self, verbose_new):
        """
        Update verbose parameter
        """
        if (verbose_new is not True) & (verbose_new is not False):
            raise ValueError("verbose should be either True or False")

        self.work.settings.verbose = verbose_new

    def update_scaled_termination(self, scaled_termination_new):
        """
        Update scaled_termination parameter
        """
        if (scaled_termination_new is not True) & (scaled_termination_new is not False):
            raise ValueError("scaled_termination should be either True or False")

        self.work.settings.scaled_termination = scaled_termination_new

    def update_early_terminate(self, early_terminate_new):
        """
        Update early_terminate parameter
        """
        if (early_terminate_new is not True) & (early_terminate_new is not False):
            raise ValueError("early_terminate should be either True or False")

        self.work.settings.early_terminate = early_terminate_new

    def update_early_terminate_interval(self, early_terminate_interval_new):
        """
        Update early_terminate_interval parameter
        """
        if (early_terminate_interval_new is not True) & (early_terminate_interval_new is not False):
            raise ValueError("early_terminate_interval should be either True or False")

        self.work.settings.early_terminate_interval = early_terminate_interval_new

    def update_warm_start(self, warm_start_new):
        """
        Update warm_start parameter
        """
        if (warm_start_new is not True) & (warm_start_new is not False):
            raise ValueError("warm_start should be either True or False")

        self.work.settings.warm_start = warm_start_new

    def constant(self, constant_name):
        """
        Return solver constant
        """
        if constant_name == "OSQP_INFTY":
            return OSQP_INFTY
        if constant_name == "OSQP_NAN":
            return OSQP_NAN
        if constant_name == "OSQP_SOLVED":
            return OSQP_SOLVED
        if constant_name == "OSQP_UNSOLVED":
            return OSQP_UNSOLVED
        if constant_name == "OSQP_PRIMAL_INFEASIBLE":
            return OSQP_PRIMAL_INFEASIBLE
        if constant_name == "OSQP_DUAL_INFEASIBLE":
            return OSQP_DUAL_INFEASIBLE
        if constant_name == "OSQP_MAX_ITER_REACHED":
            return OSQP_MAX_ITER_REACHED

        raise ValueError('Constant not recognized!')


    def iter_refin(self, KKT_factor, z, b):
        """
        Iterative refinement of the solution of a linear system
            1. (K + dK) * dz = b - K*z
            2. z <- z + dz
        """
        for i in range(self.work.settings.pol_refine_iter):
            rhs = b - np.hstack([
                            self.work.data.P.dot(z[:self.work.data.n]) +
                            self.work.pol.Ared.T.dot(z[self.work.data.n:]),
                            self.work.pol.Ared.dot(z[:self.work.data.n])])
            dz = KKT_factor.solve(rhs)
            z += dz
        return z

    def polish(self):
        """
        Solution polish:
        Solve equality constrained QP with assumed active constraints.
        """
        # Start timer
        self.work.timer = time.time()

        # Guess which linear constraints are lower-active, upper-active, free
        self.work.pol.ind_low = np.where(self.work.z -
                                         self.work.data.l < -self.work.y)[0]
        self.work.pol.ind_upp = np.where(self.work.data.u -
                                         self.work.z < self.work.y)[0]
        self.work.pol.n_low = len(self.work.pol.ind_low)
        self.work.pol.n_upp = len(self.work.pol.ind_upp)

        # Form Ared from the assumed active constraints
        self.work.pol.Ared = spspa.vstack([
                                self.work.data.A[self.work.pol.ind_low],
                                self.work.data.A[self.work.pol.ind_upp]])

        # # Terminate if there are no active constraints
        # if self.work.pol.Ared.shape[0] == 0:
        #     return

        # Form and factorize reduced KKT
        KKTred = spspa.vstack([
              spspa.hstack([self.work.data.P + self.work.settings.delta *
                            spspa.eye(self.work.data.n),
                            self.work.pol.Ared.T]),
              spspa.hstack([self.work.pol.Ared, -self.work.settings.delta *
                            spspa.eye(self.work.pol.Ared.shape[0])])])
        KKTred_factor = spla.splu(KKTred.tocsc())

        # Form reduced RHS
        rhs_red = np.hstack([-self.work.data.q,
                             self.work.data.l[self.work.pol.ind_low],
                             self.work.data.u[self.work.pol.ind_upp]])

        # Solve reduced KKT system
        pol_sol = KKTred_factor.solve(rhs_red)

        # Perform iterative refinement to compensate for the reg. error
        if self.work.settings.pol_refine_iter > 0:
            pol_sol = self.iter_refin(KKTred_factor, pol_sol, rhs_red)

        # Store the polished solution
        self.work.pol.x = pol_sol[:self.work.data.n]
        self.work.pol.z = self.work.data.A.dot(self.work.pol.x)
        self.work.pol.y_red = pol_sol[self.work.data.n:]

        # Compute primal and dual residuals of the polished solution
        self.update_info(0, 1)

        # Update polish time
        self.work.info.polish_time = time.time() - self.work.timer

        # Check if polish was successful
        pol_success = (self.work.pol.pri_res < self.work.info.pri_res) and \
                      (self.work.pol.dua_res < self.work.info.dua_res) or \
                      (self.work.pol.pri_res < self.work.info.pri_res) and \
                      (self.work.info.dua_res < 1e-10) or \
                      (self.work.pol.dua_res < self.work.info.dua_res) and \
                      (self.work.info.pri_res < 1e-10)

        if pol_success:
            # Update solver information
            self.work.info.obj_val = self.work.pol.obj_val
            self.work.info.pri_res = self.work.pol.pri_res
            self.work.info.dua_res = self.work.pol.dua_res
            self.work.info.status_polish = 1

            # Update ADMM iterations
            self.work.x = self.work.pol.x
            self.work.z = self.work.pol.z
            self.work.y = np.zeros(self.work.data.m)
            if self.work.pol.Ared.shape[0] > 0:
                self.work.y[self.work.pol.ind_low] = \
                    self.work.pol.y_red[:self.work.pol.n_low]
                self.work.y[self.work.pol.ind_upp] = \
                    self.work.pol.y_red[self.work.pol.n_low:]

            # Print summary
            if self.work.settings.verbose:
                self.print_polish()
        else:
            self.work.info.status_polish = -1
