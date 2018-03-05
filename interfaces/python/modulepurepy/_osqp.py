"""
OSQP Solver pure python implementation: low level module
"""
from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution

# Solver Constants
OSQP_DUAL_INFEASIBLE_INACCURATE = 4
OSQP_PRIMAL_INFEASIBLE_INACCURATE = 3
OSQP_SOLVED_INACCURATE = 2
OSQP_SOLVED = 1
OSQP_MAX_ITER_REACHED = -2
OSQP_PRIMAL_INFEASIBLE = -3
OSQP_DUAL_INFEASIBLE = -4
OSQP_UNSOLVED = -10

# Parameter bounds
RHO_MIN = 1e-06
RHO_MAX = 1e+06
RHO_EQ_OVER_RHO_INEQ = 1e+03
RHO_TOL = 1e-04


# Printing interval
PRINT_INTERVAL = 200

# OSQP Infinity
OSQP_INFTY = 1e+20

# OSQP Nan
OSQP_NAN = 1e+20  # Just as placeholder. Not real value

# Linear system solver options
SUITESPARSE_LDL_SOLVER = 0

# Scaling
MIN_SCALING = 1e-04
MAX_SCALING = 1e+04


class workspace(object):
    """
    OSQP solver workspace

    Attributes
    ----------
    data                   - scaled QP problem
    info                   - solver information
    linsys_solver          - structure for linear system solution
    scaling                - scaling matrices
    settings               - settings structure
    solution               - solution structure


    Additional workspace variables
    ------------------------------
    first_run              - flag to indicate if it is the first run
    timer                  - saved time instant for timing purposes
    x                      - primal iterate
    x_prev                 - previous primal iterate
    xz_tilde               - x_tilde and z_tilde iterates stacked together
    y                      - dual iterate
    z                      - z iterate
    z_prev                 - previous z iterate

    Vectorized rho parameter
    ------------------------
    rho_vec                - vector of rho values for each constraint
    rho_inv_vec            - vector of reciprocal rho values
    constr_type            - type of constraints: loose (-1), eq (1), ineq (0)

    Primal infeasibility related workspace variables
    ------------------------------------------------
    delta_y                - difference of consecutive y
    Atdelta_y              - A' * delta_y

    Dual infeasibility related workspace variables
    ----------------------------------------------
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


class settings(object):
    """
    OSQP solver settings

    Attributes
    ----------
    -> These cannot be changed without running setup
    sigma    [1e-06]           - Regularization parameter for polish
    scaling  [10]            - Scaling/Equilibration iterations (0 disabled)

    -> These can be changed without running setup
    rho  [1.6]                 - Step in ADMM procedure
    max_iter [4000]                     - Maximum number of iterations
    eps_abs  [1e-05]                    - Absolute tolerance
    eps_rel  [1e-05]                    - Relative tolerance
    eps_prim_inf  [1e-06]                    - Primal infeasibility tolerance
    eps_dual_inf  [1e-06]                    - Dual infeasibility tolerance
    alpha [1.6]                         - Relaxation parameter
    delta [1.0]                         - Regularization parameter for polish
    verbose  [True]                     - Verbosity
    scaled_termination [False]             - Evalute scaled termination criteria
    check_termination  [True]             - Interval for termination checking
    warm_start [False]                  - Reuse solution from previous solve
    polish  [False]                     - Solution polish
    polish_refine_iter  [3]                - Iterative refinement iterations
    """

    def __init__(self, **kwargs):

        self.rho = kwargs.pop('rho', 0.1)
        self.sigma = kwargs.pop('sigma', 1e-06)
        self.scaling = kwargs.pop('scaling', 10)
        self.max_iter = kwargs.pop('max_iter', 4000)
        self.eps_abs = kwargs.pop('eps_abs', 1e-3)
        self.eps_rel = kwargs.pop('eps_rel', 1e-3)
        self.eps_prim_inf = kwargs.pop('eps_prim_inf', 1e-4)
        self.eps_dual_inf = kwargs.pop('eps_dual_inf', 1e-4)
        self.alpha = kwargs.pop('alpha', 1.6)
        self.linsys_solver = kwargs.pop('linsys_solver',
                                        SUITESPARSE_LDL_SOLVER)
        self.delta = kwargs.pop('delta', 1e-6)
        self.verbose = kwargs.pop('verbose', True)
        self.scaled_termination = kwargs.pop('scaled_termination', False)
        self.check_termination = kwargs.pop('check_termination', True)
        self.warm_start = kwargs.pop('warm_start', True)
        self.polish = kwargs.pop('polish', False)
        self.polish_refine_iter = kwargs.pop('polish_refine_iter', 3)
        self.adaptive_rho = kwargs.pop('adaptive_rho', False)
        self.adaptive_rho_interval = kwargs.pop('adaptive_rho_interval', 200)
        self.adaptive_rho_tolerance = kwargs.pop('adaptive_rho_tolerance', 5)
        self.adaptive_rho_fraction = kwargs.pop('adaptive_rho_fraction', 0.7)


class scaling(object):
    """
    Matrices for diagonal scaling

    Attributes
    ----------
    D        - matrix in R^{n \\times n}
    E        - matrix in R^{m \\times n}
    Dinv     - inverse of D
    Einv     - inverse of E
    c        - cost scaling
    cinv    - inverse of cost scaling
    """
    def __init__(self):
        self.D = None
        self.E = None
        self.Dinv = None
        self.Einv = None
        self.c = None
        self.cinv = None


class linesearch(object):
    """
    Vectors obtained from line search between the ADMM and the polished
    solution

    Attributes
    ----------
    X     - matrix in R^{N \\times n}
    Z     - matrix in R^{N \\times m}
    Y     - matrix in R^{N \\times m}
    t     - vector in R^N
    """
    def __init__(self):
        self.X = None
        self.Z = None
        self.Y = None
        self.t = None


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
    setup_time      - time taken for setup phase (seconds)
    solve_time      - time taken for solve phase (seconds)
    polish_time     - time taken for polish phase (seconds)
    run_time        - total time  (seconds)
    rho_updates     - number of rho updates
    rho_estimate    - optimal rho estimate 
    """
    def __init__(self):
        self.iter = 0
        self.status_val = OSQP_UNSOLVED
        self.status = "Unsolved"
        self.status_polish = 0
        self.polish_time = 0.0
        self.rho_updates = 0.0


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
    y               - polished y
    """
    def __init__(self):
        self.ind_low = None
        self.ind_upp = None
        self.n_low = None
        self.n_upp = None
        self.Ared = None
        self.x = None
        self.z = None
        self.y = None


class linsys_solver(object):
    """
    Linear systems solver
    """

    def __init__(self, work):
        """
        Initialize structure for KKT system solution
        """
        # Construct reduced KKT matrix
        KKT = spspa.vstack([
              spspa.hstack([work.data.P + work.settings.sigma *
                            spspa.eye(work.data.n), work.data.A.T]),
              spspa.hstack([work.data.A, -spspa.diags(work.rho_inv_vec)])])

        # Initialize structure
        self.kkt_factor = spla.splu(KKT.tocsc())
        #  self.lu, self.piv = sp.linalg.lu_factor(KKT.todense())

    def solve(self, rhs):
        """
        Solve linear system with given factorization
        """
        return self.kkt_factor.solve(rhs)
        #  return sp.linalg.lu_solve((self.lu, self.piv), rhs)


class results(object):
    """
    Results structure

    Attributes
    ----------
    x           - primal solution
    y           - dual solution
    info        - info structure
    """
    def __init__(self, solution, info, linesearch):
        self.x = solution.x
        self.y = solution.y
        self.info = info
        self.linesearch = linesearch


class OSQP(object):
    """OSQP solver lower level interface
    Attributes
    ----------
    work    - workspace
    """
    def __init__(self):
        self._version = "0.3.0"

    @property
    def version(self):
        """Return solver version
        """
        return self._version

    def _norm_KKT_cols(self, P, A):
        """
        Compute the norm of the KKT matrix from P and A
        """

        # First half
        norm_P_cols = spspa.linalg.norm(P, np.inf, axis=0)
        norm_A_cols = spspa.linalg.norm(A, np.inf, axis=0)
        norm_first_half = np.maximum(norm_P_cols, norm_A_cols)

        # Second half (norm cols of A')
        norm_second_half = spspa.linalg.norm(A, np.inf, axis=1)

        return np.hstack((norm_first_half, norm_second_half))

    def _limit_scaling(self, norm_vec):
        """
        Norm vector for scaling
        """

        if isinstance(norm_vec, (list, tuple, np.ndarray)):   # Array

            n = len(norm_vec)
            new_norm_vec = np.zeros(n)

            for i in range(n):
                if norm_vec[i] < MIN_SCALING:
                    new_norm_vec[i] = 1.
                elif norm_vec[i] > MAX_SCALING:
                    new_norm_vec[i] = MAX_SCALING
                else:
                    new_norm_vec[i] = norm_vec[i]
        else:   # Scalar
            if norm_vec < MIN_SCALING:
                new_norm_vec = 1.
            elif norm_vec > MAX_SCALING:
                new_norm_vec = MAX_SCALING
            else:
                new_norm_vec = norm_vec

        return new_norm_vec

    def scale_data(self):
        """
        Perform symmetric diagonal scaling via equilibration
        """
        n = self.work.data.n
        m = self.work.data.m

        # Initialize scaling
        s_temp = np.ones(n + m)
        c = 1.0  # Cost scaling

        # Define data
        P = self.work.data.P
        q = self.work.data.q
        A = self.work.data.A
        l = self.work.data.l
        u = self.work.data.u

        # Initialize scaler matrices
        D = spspa.eye(n)
        if m == 0:
            # spspa.diags() will throw an error if fed with an empty array
            E = spspa.csc_matrix((0, 0))
        else:
            E = spspa.eye(m)

        # Iterate Scaling
        for i in range(self.work.settings.scaling):

            # First Step Ruiz
            norm_cols = self._norm_KKT_cols(P, A)
            norm_cols = self._limit_scaling(norm_cols)  # Limit scaling
            sqrt_norm_cols = np.sqrt(norm_cols)         # Compute sqrt
            s_temp = np.reciprocal(sqrt_norm_cols)      # Elementwise recipr

            # Obtain Scaler Matrices
            D_temp = spspa.diags(s_temp[:self.work.data.n])
            if m == 0:
                # spspa.diags() will throw an error if fed with an empty array
                E_temp = spspa.csc_matrix((0, 0))
            else:
                E_temp = spspa.diags(s_temp[self.work.data.n:])

            # Scale data in place
            P = D_temp.dot(P.dot(D_temp)).tocsc()
            A = E_temp.dot(A.dot(D_temp)).tocsc()
            q = D_temp.dot(q)
            l = E_temp.dot(l)
            u = E_temp.dot(u)

            # Update equilibration matrices D and E
            D = D_temp.dot(D)
            E = E_temp.dot(E)

            # Second Step cost normalization
            norm_P_cols = spla.norm(P, np.inf, axis=0).mean()
            inf_norm_q = np.linalg.norm(q, np.inf)
            inf_norm_q = self._limit_scaling(inf_norm_q)
            scale_cost = np.maximum(inf_norm_q, norm_P_cols)
            #  import ipdb; ipdb.set_trace()
            scale_cost = self._limit_scaling(scale_cost)
            scale_cost = 1. / scale_cost

            # scale_cost = 1. / np.maximum(np.minimum(
            #     scale_cost, MAX_SCALING), MIN_SCALING)
            # print("trace P", P.todense().trace()[0, 0])
            # print("sum_norm_P_cols", spla.norm(P, np.inf, axis=0).sum())
            # print("norm_P_cols", norm_P_cols)
            # print("inf_norm_q", inf_norm_q)
            # print("Scale cost = %.2e" % scale_cost)

            # norm_cost = self._limit_scaling(norm_cost)
            c_temp = scale_cost

            # c_temp = 1.0

            # Normalize cost
            P = c_temp * P
            q = c_temp * q

            # Update scaling
            c = c_temp * c

        if self.work.settings.verbose:
            print("Final cost scaling = %.10f" % c)

        # import ipdb; ipdb.set_trace()

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
        self.work.scaling.c = c
        self.work.scaling.cinv = 1. / c

    def set_rho_vec(self):
        """
        Set values of rho vector based on constraint types
        """
        self.work.settings.rho = np.minimum(np.maximum(self.work.settings.rho,
                                            RHO_MIN), RHO_MAX)

        # Find indices of loose bounds, equality constr and one-sided constr
        loose_ind = np.where(np.logical_and(
                            self.work.data.l < -OSQP_INFTY*MIN_SCALING,
                            self.work.data.u > OSQP_INFTY*MIN_SCALING))[0]
        eq_ind = np.where(self.work.data.u - self.work.data.l < RHO_TOL)[0]
        ineq_ind = np.setdiff1d(np.setdiff1d(np.arange(self.work.data.m),
                                loose_ind), eq_ind)

        # Type of constraints
        self.work.constr_type[loose_ind] = -1
        self.work.constr_type[eq_ind] = 1
        self.work.constr_type[ineq_ind] = 0

        self.work.rho_vec[loose_ind] = RHO_MIN
        self.work.rho_vec[eq_ind] = RHO_EQ_OVER_RHO_INEQ * \
            self.work.settings.rho
        self.work.rho_vec[ineq_ind] = self.work.settings.rho

        self.work.rho_inv_vec = np.reciprocal(self.work.rho_vec)

    def update_rho_vec(self):
        """
        Update values of rho_vec and refactor if constraints change.
        """
        # Find indices of loose bounds, equality constr and one-sided constr
        loose_ind = np.where(np.logical_and(
                            self.work.data.l < -OSQP_INFTY*MIN_SCALING,
                            self.work.data.u > OSQP_INFTY*MIN_SCALING))[0]
        eq_ind = np.where(self.work.data.u - self.work.data.l < RHO_TOL)[0]
        ineq_ind = np.setdiff1d(np.setdiff1d(np.arange(self.work.data.m),
                                loose_ind), eq_ind)

        # Find indices of current constraint types
        old_loose_ind = np.where(self.work.constr_type == -1)
        old_eq_ind = np.where(self.work.constr_type == 1)
        old_ineq_ind = np.where(self.work.constr_type == 0)

        # Check if type of any constraint changed
        constr_type_changed = (loose_ind != old_loose_ind).any() or \
                              (eq_ind != old_eq_ind).any() or \
                              (ineq_ind != old_ineq_ind).any()

        # Update type of constraints
        self.work.constr_type[loose_ind] = -1
        self.work.constr_type[eq_ind] = 1
        self.work.constr_type[ineq_ind] = 0

        self.work.rho_vec[loose_ind] = RHO_MIN
        self.work.rho_vec[eq_ind] = RHO_EQ_OVER_RHO_INEQ * \
            self.work.settings.rho
        self.work.rho_vec[ineq_ind] = self.work.settings.rho

        self.work.rho_inv_vec = np.reciprocal(self.work.rho_vec)

        if constr_type_changed:
            self.work.linsys_solver = linsys_solver(self.work)

    def print_setup_header(self, data, settings):
        """Print solver header
        """
        print("--------------------------------------------------------------")
        print("         OSQP v%s  -  Operator Splitting QP Solver" %
              self.version)
        print("                 Pure Python Implementation")
        print("        (c) Bartolomeo Stellato, Goran Banjac")
        print("      University of Oxford  -  Stanford University 2017")
        print("--------------------------------------------------------------")

        print("problem:  variables n = %d, constraints m = %d" %
              (data.n, data.m))
        nnz = self.work.data.P.nnz + self.work.data.A.nnz
        print("          nnz(P) + nnz(A) = %i" % nnz)
        print("settings: ", end='')
        if settings.linsys_solver == SUITESPARSE_LDL_SOLVER:
            print("linear system solver = suitesparse ldl\n          ", end='')
        print("eps_abs = %.2e, eps_rel = %.2e," %
              (settings.eps_abs, settings.eps_rel))
        print("          eps_prim_inf = %.2e, eps_dual_inf = %.2e," %
              (settings.eps_prim_inf, settings.eps_dual_inf))
        print("          rho = %.2e " % settings.rho, end='')
        if settings.adaptive_rho:
            print("(adaptive)")
        else:
            print("")
        print("          sigma = %.2e, alpha = %.2f, " %
              (settings.sigma, settings.alpha), end='')
        print("max_iter = %d" % settings.max_iter)
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
        print("iter   objective    pri res    dua res    rho       time")

    def update_status(self, status):
        self.work.info.status_val = status
        if status == OSQP_SOLVED:
            self.work.info.status = "solved"
        if status == OSQP_SOLVED_INACCURATE:
            self.work.info.status = "solved inaccurate"
        elif status == OSQP_PRIMAL_INFEASIBLE:
            self.work.info.status = "primal infeasible"
        elif status == OSQP_PRIMAL_INFEASIBLE_INACCURATE:
            self.work.info.status = "primal infeasible inaccurate"
        elif status == OSQP_UNSOLVED:
            self.work.info.status = "unsolved"
        elif status == OSQP_DUAL_INFEASIBLE:
            self.work.info.status = "dual infeasible"
        elif status == OSQP_DUAL_INFEASIBLE_INACCURATE:
            self.work.info.status = "dual infeasible inaccurate"
        elif status == OSQP_MAX_ITER_REACHED:
            self.work.info.status = "maximum iterations reached"

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
            self.work.z_prev - self.work.rho_inv_vec * self.work.y

        # Solve linear system
        self.work.xz_tilde = self.work.linsys_solver.solve(self.work.xz_tilde)

        # Update z_tilde
        self.work.xz_tilde[self.work.data.n:] = \
            self.work.z_prev + self.work.rho_inv_vec * \
            (self.work.xz_tilde[self.work.data.n:] - self.work.y)

    def update_x(self):
        """
        Update x variable in second ADMM step
        """
        self.work.x = \
            self.work.settings.alpha * self.work.xz_tilde[:self.work.data.n] +\
            (1. - self.work.settings.alpha) * self.work.x_prev
        self.work.delta_x = self.work.x - self.work.x_prev

    def project(self, z):
        """
        Project z variable in set C (for now C = [l, u])
        """
        return np.minimum(np.maximum(z, self.work.data.l), self.work.data.u)

    def project_normalcone(self, z, y):
        tmp = z + y
        z = np.minimum(np.maximum(tmp, self.work.data.l), self.work.data.u)
        y = tmp - z
        return z, y

    def update_z(self):
        """
        Update z variable in second ADMM step
        """
        self.work.z = \
            self.work.settings.alpha * self.work.xz_tilde[self.work.data.n:] +\
            (1. - self.work.settings.alpha) * self.work.z_prev +\
            self.work.rho_inv_vec * self.work.y

        self.work.z = self.project(self.work.z)

    def update_y(self):
        """
        Third ADMM step: update dual variable y
        """
        self.work.delta_y = self.work.rho_vec * \
            (self.work.settings.alpha * self.work.xz_tilde[self.work.data.n:] +
                (1. - self.work.settings.alpha) * self.work.z_prev -
                self.work.z)
        self.work.y += self.work.delta_y

    def compute_obj_val(self, x):
        # Compute quadratic objective value for the given x
        obj_val = .5 * np.dot(x, self.work.data.P.dot(x)) + \
            np.dot(self.work.data.q, x)

        if self.work.settings.scaling:
            obj_val *= self.work.scaling.cinv

        return obj_val

    def compute_pri_res(self, x, z):
        """
        Compute primal residual ||Ax - z||
        """

        # Primal residual
        Ax = self.work.data.A.dot(x)
        pri_res = Ax - z

        if self.work.settings.scaling and not \
                self.work.settings.scaled_termination:
            pri_res = self.work.scaling.Einv.dot(pri_res)

        return la.norm(pri_res, np.inf)

    def compute_pri_tol(self, eps_abs, eps_rel):
        """
        Compute primal tolerance using problem data
        """
        A = self.work.data.A
        if self.work.settings.scaling and not \
                self.work.settings.scaled_termination:
            Einv = self.work.scaling.Einv
            max_rel_eps = np.max([
                la.norm(Einv.dot(A.dot(self.work.x)), np.inf),
                la.norm(Einv.dot(self.work.z), np.inf)])
        else:
            max_rel_eps = np.max([
                la.norm(A.dot(self.work.x), np.inf),
                la.norm(self.work.z, np.inf)])

        eps_pri = eps_abs + eps_rel * max_rel_eps

        return eps_pri

    def compute_dua_res(self, x, y):
        """
        Compute dual residual ||Px + q + A'y||
        """

        dua_res = self.work.data.P.dot(x) +\
            self.work.data.q + self.work.data.A.T.dot(y)

        if self.work.settings.scaling and not \
                self.work.settings.scaled_termination:
            # Use unscaled residual
            dua_res = self.work.scaling.cinv * \
                self.work.scaling.Dinv.dot(dua_res)

        return la.norm(dua_res, np.inf)

    def compute_dua_tol(self, eps_abs, eps_rel):
        """
        Compute dual tolerance
        """
        P = self.work.data.P
        q = self.work.data.q
        A = self.work.data.A
        if self.work.settings.scaling and not \
                self.work.settings.scaled_termination:
            cinv = self.work.scaling.cinv
            Dinv = self.work.scaling.Dinv
            max_rel_eps = cinv * np.max([
                la.norm(Dinv.dot(A.T.dot(self.work.y)), np.inf),
                la.norm(Dinv.dot(P.dot(self.work.x)), np.inf),
                la.norm(Dinv.dot(q), np.inf)])
        else:
            max_rel_eps = np.max([
                la.norm(A.T.dot(self.work.y), np.inf),
                la.norm(P.dot(self.work.x), np.inf),
                la.norm(q, np.inf)])

        eps_dua = eps_abs + eps_rel * max_rel_eps

        return eps_dua

    def is_primal_infeasible(self, eps_prim_inf):
        """
        Check primal infeasibility
                ||A'*v||_2 = 0
        with v = delta_y/||delta_y||_2 given that following condition holds
            u'*(v)_{+} + l'*(v)_{-} < 0
        """

        # Rescale delta_y
        if self.work.settings.scaling and not \
                self.work.settings.scaled_termination:
            norm_delta_y = la.norm(self.work.scaling.E.dot(self.work.delta_y),
                                   np.inf)
        else:
            norm_delta_y = la.norm(self.work.delta_y, np.inf)

        if norm_delta_y > eps_prim_inf:
            lhs = self.work.data.u.dot(np.maximum(self.work.delta_y, 0)) + \
                self.work.data.l.dot(np.minimum(self.work.delta_y, 0))
            if lhs < -eps_prim_inf * norm_delta_y:
                self.work.Atdelta_y = self.work.data.A.T.dot(self.work.delta_y)
                if self.work.settings.scaling and not \
                        self.work.settings.scaled_termination:
                    self.work.Atdelta_y = \
                        self.work.scaling.Dinv.dot(self.work.Atdelta_y)
                return la.norm(self.work.Atdelta_y, np.inf) < \
                    eps_prim_inf * norm_delta_y

        return False

    def is_dual_infeasible(self, eps_dual_inf):
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
        # Rescale delta_x
        if self.work.settings.scaling and not \
                self.work.settings.scaled_termination:
            norm_delta_x = la.norm(self.work.scaling.D.dot(self.work.delta_x),
                                   np.inf)
            scale_cost = self.work.scaling.c
        else:
            norm_delta_x = la.norm(self.work.delta_x, np.inf)
            scale_cost = 1.0

        # Prevent 0 division
        if norm_delta_x > eps_dual_inf:

            # First check q'* delta_x < 0
            if self.work.data.q.dot(self.work.delta_x) < \
                    - scale_cost * eps_dual_inf * norm_delta_x:
                # Compute P * delta_x
                self.work.Pdelta_x = self.work.data.P.dot(self.work.delta_x)

                # Scale if necessary
                if self.work.settings.scaling and not \
                        self.work.settings.scaled_termination:
                    self.work.Pdelta_x = \
                        self.work.scaling.Dinv.dot(self.work.Pdelta_x)

                # Check if ||P * delta_x|| = 0
                if la.norm(self.work.Pdelta_x, np.inf) < \
                        scale_cost * eps_dual_inf * norm_delta_x:

                    # Compute A * delta_x
                    self.work.Adelta_x = self.work.data.A.dot(
                        self.work.delta_x)

                    # Scale if necessary
                    if self.work.settings.scaling and not \
                            self.work.settings.scaled_termination:
                        self.work.Adelta_x = \
                            self.work.scaling.Einv.dot(self.work.Adelta_x)

                    for i in range(self.work.data.m):
                        # De Morgan's Law applied to negate
                        # conditions on A * delta_x
                        if ((self.work.data.u[i] < OSQP_INFTY*1e-06) and
                            (self.work.Adelta_x[i] >
                             eps_dual_inf * norm_delta_x)) or \
                            ((self.work.data.l[i] > -OSQP_INFTY*1e-06) and
                             (self.work.Adelta_x[i] <
                              -eps_dual_inf * norm_delta_x)):

                            # At least one condition not satisfied
                            return False

                    # All conditions passed -> dual infeasible
                    return True

        # No all checks managed to pass. Problem not dual infeasible
        return False
    
    def compute_rho_estimate(self):
        # Iterates
        x = self.work.x
        y = self.work.y
        z = self.work.z

        # Problem data
        P = self.work.data.P
        q = self.work.data.q
        A = self.work.data.A

        # Compute normalized residuals
        pri_res = la.norm(A.dot(x) - z, np.inf)
        pri_res /= (np.max([la.norm(A.dot(x), np.inf),
                            la.norm(z, np.inf)]) + 1e-10)
        dua_res = la.norm(P.dot(x) + q + A.T.dot(y), np.inf)
        dua_res /= (np.max([la.norm(A.T.dot(y), np.inf),
                           la.norm(P.dot(x), np.inf),
                           la.norm(q, np.inf)]) + 1e-10)

        # Compute new rho
        new_rho = self.work.settings.rho * np.sqrt(pri_res/(dua_res + 1e-10))
        return min(max(new_rho, RHO_MIN), RHO_MAX)
    
    def adapt_rho(self):
        """
        Adapt rho value based on current primal and dual residuals
        """
        # Compute new rho 
        rho_new = self.compute_rho_estimate()

        # Update rho estimate
        self.work.info.rho_estimate = rho_new
        
        # Settings
        adaptive_rho_tolerance = self.work.settings.adaptive_rho_tolerance

        if rho_new > adaptive_rho_tolerance * self.work.settings.rho or \
            rho_new < 1. / adaptive_rho_tolerance * \
                self.work.settings.rho:
            # Update rho
            self.update_rho(rho_new)
            # Update rho updates count
            self.work.info.rho_updates += 1
    
    def reset_info(self, info):
        """
        Reset information after problem updates
        """
        info.solve_time = 0.0
        info.polish_time = 0.0

        self.update_status(OSQP_UNSOLVED)

        info.rho_updates = 0


    def update_info(self, iter, polish):
        """
        Update information at iterations
        """

        if polish == 1:
            self.work.pol.obj_val = self.compute_obj_val(self.work.pol.x)
            self.work.pol.pri_res = self.compute_pri_res(self.work.pol.x,
                                                         self.work.pol.z)
            self.work.pol.dua_res = self.compute_dua_res(self.work.pol.x,
                                                         self.work.pol.y)
            self.work.info.polish_time = time.time() - self.work.timer
        else:
            self.work.info.iter = iter
            self.work.info.obj_val = self.compute_obj_val(self.work.x)
            self.work.info.pri_res = self.compute_pri_res(self.work.x,
                                                          self.work.z)
            self.work.info.dua_res = self.compute_dua_res(self.work.x,
                                                          self.work.y)
            self.work.info.solve_time = time.time() - self.work.timer

    def print_summary(self):
        """
        Print status summary at each ADMM iteration
        """
        print("%4i  %11.4e   %8.2e   %8.2e   %8.2e  %8.2es" %
              (self.work.info.iter,
               self.work.info.obj_val,
               self.work.info.pri_res,
               self.work.info.dua_res,
               self.work.settings.rho,
               self.work.info.setup_time + self.work.info.solve_time))

    def print_polish(self):
        """
        Print polish information
        """
        print("plsh  %11.4e   %8.2e   %8.2e   --------  %8.2es" %
              (self.work.info.obj_val,
               self.work.info.pri_res,
               self.work.info.dua_res,
               self.work.info.setup_time + self.work.info.solve_time +
               self.work.info.polish_time))

    def check_termination(self, approximate=False):
        """
        Check residuals for algorithm convergence and update solver status

        Args
        ----
            approximate: bool to determine if termination criteria are
                         approximate or accurate

        """
        pri_check = 0
        dua_check = 0
        prim_inf_check = 0
        dual_inf_check = 0

        eps_abs = self.work.settings.eps_abs
        eps_rel = self.work.settings.eps_rel
        eps_prim_inf = self.work.settings.eps_prim_inf
        eps_dual_inf = self.work.settings.eps_dual_inf

        if approximate:
            eps_abs *= 10
            eps_rel *= 10
            eps_prim_inf *= 10
            eps_dual_inf *= 10

        if self.work.data.m == 0:  # No constraints -> always  primal feasible
            pri_check = 1
        else:
            # Compute primal tolerance
            eps_pri = self.compute_pri_tol(eps_abs, eps_rel)

            if self.work.info.pri_res < eps_pri:
                pri_check = 1
            else:
                # Check infeasibility
                prim_inf_check = self.is_primal_infeasible(eps_prim_inf)

        # Compute dual tolerance
        eps_dua = self.compute_dua_tol(eps_abs, eps_rel)

        if self.work.info.dua_res < eps_dua:
            dua_check = 1
        else:
            # Check dual infeasibility
            dual_inf_check = self.is_dual_infeasible(eps_dual_inf)

        # Compare residuals and determine solver status
        if pri_check & dua_check:
            if approximate:
                self.work.info.status_val = OSQP_SOLVED_INACCURATE
            else:
                self.work.info.status_val = OSQP_SOLVED
            return 1
        elif prim_inf_check:
            if approximate:
                self.work.info.status_val = OSQP_PRIMAL_INFEASIBLE_INACCURATE
            else:
                self.work.info.status_val = OSQP_PRIMAL_INFEASIBLE
            self.work.info.obj_val = OSQP_INFTY
            # Store original certificate
            if self.work.settings.scaling and not \
                    self.work.settings.scaled_termination:
                self.work.delta_y = self.work.scaling.E.dot(self.work.delta_y)
            return 1
        elif dual_inf_check:
            if approximate:
                self.work.info.status_val = OSQP_DUAL_INFEASIBLE_INACCURATE
            else:
                self.work.info.status_val = OSQP_DUAL_INFEASIBLE
            # Store original certificate
            if self.work.settings.scaling and not \
                    self.work.settings.scaled_termination:
                self.work.delta_x = self.work.scaling.D.dot(self.work.delta_x)
            self.work.info.obj_val = -OSQP_INFTY
            return 1

    def print_footer(self):
        """
        Print footer at the end of the optimization
        """
        print("")  # Add space after iterations
        print("status:               %s" % self.work.info.status)
        if self.work.settings.polish and \
                self.work.info.status_val == OSQP_SOLVED:
                    if self.work.info.status_polish == 1:
                        print("solution polish:      successful")
                    elif self.work.info.status_polish == -1:
                        print("solution polish:      unsuccessful")
        print("number of iterations: %d" % self.work.info.iter)
        if self.work.info.status_val == OSQP_SOLVED or \
                self.work.info.status_val == OSQP_SOLVED_INACCURATE:
            print("optimal objective:    %.4f" % self.work.info.obj_val)
            print("run time:             %.2es" % (self.work.info.run_time))
        print("optimal rho estimate: %.2es" %
                (self.work.info.rho_estimate))

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
                    self.work.scaling.cinv * \
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

        # Vectorized rho parameter
        self.work.rho_vec = np.zeros(m)
        self.work.rho_inv_vec = np.zeros(m)

        # Type of constraints
        self.work.constr_type = np.zeros(m)

        # Initialize workspace variables
        self.work.x = np.zeros(n)
        self.work.z = np.zeros(m)
        self.work.xz_tilde = np.zeros(n + m)
        self.work.x_prev = np.zeros(n)
        self.work.z_prev = np.zeros(m)
        self.work.y = np.zeros(m)
        self.work.delta_y = np.zeros(m)    # Delta_y for primal infeasibility

        # Flag indicating first run
        self.work.first_run = 1

        # Settings
        self.work.settings = settings(**stgs)

        # Scale problem
        if self.work.settings.scaling:
            self.scale_data()

        # Set type of constraints
        self.set_rho_vec()

        # Factorize KKT
        self.work.linsys_solver = linsys_solver(self.work)

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

            if self.work.settings.check_termination:
                # Update info
                self.update_info(iter, 0)

                # Print summary
                if (self.work.settings.verbose) & \
                        ((iter % PRINT_INTERVAL == 0) | (iter == 1)):
                    self.print_summary()

                # Break if converged
                if self.check_termination():
                    break

            # If not terminated, update rho in case
            if self.work.settings.adaptive_rho_interval and \
                    (iter % self.work.settings.adaptive_rho_interval == 0) \
                    and self.work.settings.adaptive_rho:
                self.adapt_rho()
                # DEBUG: Print
                #  if self.work.settings.verbose:
                #      print("rho = %.2e" % self.work.settings.rho)

        if not self.work.settings.check_termination:
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
            if not self.check_termination(approximate=True):
                self.work.info.status_val = OSQP_MAX_ITER_REACHED

        # Update status string
        self.update_status(self.work.info.status_val)

        # Update solve time
        self.work.info.solve_time = time.time() - self.work.timer

        # Update rho estimate
        self.work.info.rho_estimate = self.compute_rho_estimate()

        # Solution polish
        if self.work.settings.polish and \
                self.work.info.status_val == OSQP_SOLVED:
                    ls = self.polish()
        else:
            ls = None

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
        return results(self.work.solution, self.work.info, ls)

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
            self.work.data.q = self.work.scaling.c * \
                self.work.scaling.D.dot(self.work.data.q)

        # Reset solver info
        self.reset_info(self.work.info)

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

        # Reset solver info
        self.reset_info(self.work.info)

        # If type of any constraint changed, update rho_vec and KKT matrix
        self.update_rho_vec()

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
        if not np.greater_equal(self.work.data.u, self.work.data.l).all():
            raise ValueError("Lower bound must be lower than" +
                             " or equal to upper bound!")

        # Reset solver info
        self.reset_info(self.work.info)

        # If type of any constraint changed, update rho_vec and KKT matrix
        self.update_rho_vec()

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
        if not np.greater_equal(self.work.data.u, self.work.data.l).all():
            raise ValueError("Lower bound must be lower than" +
                             " or equal to upper bound!")

        # Reset solver info
        self.reset_info(self.work.info)

        # If type of any constraint changed, update rho_vec and KKT matrix
        self.update_rho_vec()

    def update_P(self, P_new):
        """
        Update quadratic cost matrix
        """
        if self.work.settings.scaling:
            self.work.data.P = \
                self.work.scaling.c * \
                self.work.scaling.D.dot(P_new.dot(self.work.scaling.D))
        else:
            self.work.data.P = P_new
        self.work.linsys_solver = linsys_solver(self.work)

    def update_A(self, A_new):
        """
        Update constraint matrix
        """
        if self.work.settings.scaling:
            self.work.data.A = self.work.scaling.E.dot(A_new.dot(self.work.scaling.D))
        else:
            self.work.data.A = A_new
        self.work.linsys_solver = linsys_solver(self.work)

    def update_P_A(self, P_new, A_new):
        """
        Update quadratic cost and constraint matrices
        """
        if self.work.settings.scaling:
            self.work.data.P = self.work.scaling.D.dot(P_new.dot(self.work.scaling.D))
            self.work.data.A = self.work.scaling.E.dot(A_new.dot(self.work.scaling.D))
        else:
            self.work.data.P = P_new
            self.work.data.A = A_new
        self.work.linsys_solver = linsys_solver(self.work)

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

    def update_rho(self, rho_new):
        """
        Update set-size parameter rho
        """
        if rho_new <= 0:
            raise ValueError("rho must be positive")

        # Update rho
        self.work.settings.rho = np.minimum(np.maximum(rho_new,
                                            RHO_MIN), RHO_MAX)

        # Update rho_vec and rho_inv_vec
        ineq_ind = np.where(self.work.constr_type == 0)
        eq_ind = np.where(self.work.constr_type == 1)
        self.work.rho_vec[ineq_ind] = self.work.settings.rho
        self.work.rho_vec[eq_ind] = RHO_EQ_OVER_RHO_INEQ * self.work.settings.rho
        self.work.rho_inv_vec = np.reciprocal(self.work.rho_vec)

        # Factorize KKT
        self.work.linsys_solver = linsys_solver(self.work)

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

    def update_polish_refine_iter(self, polish_refine_iter_new):
        """
        Update number iterative refinement iterations in polish
        """
        if polish_refine_iter_new < 0:
            raise ValueError("polish_refine_iter must be nonnegative")

        self.work.settings.polish_refine_iter = polish_refine_iter_new

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

    def update_check_termination(self, check_termination_new):
        """
        Update check_termination parameter
        """
        if check_termination_new <= 0:
            raise ValueError("check_termination should be greater than 0")

        self.work.settings.check_termination = check_termination_new

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
        for i in range(self.work.settings.polish_refine_iter):
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
        if self.work.settings.polish_refine_iter > 0:
            pol_sol = self.iter_refin(KKTred_factor, pol_sol, rhs_red)

        # Store the polished solution (x,z,y)
        self.work.pol.x = pol_sol[:self.work.data.n]
        self.work.pol.z = self.work.data.A.dot(self.work.pol.x)
        self.work.pol.y = np.zeros(self.work.data.m)
        y_red = pol_sol[self.work.data.n:]
        self.work.pol.y[self.work.pol.ind_low] = y_red[:self.work.pol.n_low]
        self.work.pol.y[self.work.pol.ind_upp] = y_red[self.work.pol.n_low:]

        # Ensure (z,y) satisfies normal cone constraint
        self.work.pol.z, self.work.pol.y = \
            self.project_normalcone(self.work.pol.z, self.work.pol.y)

        # Compute primal and dual residuals of the polished solution
        self.update_info(0, 1)

        # Check if polish was successful
        pol_success = (self.work.pol.pri_res < self.work.info.pri_res) and \
                      (self.work.pol.dua_res < self.work.info.dua_res) or \
                      (self.work.pol.pri_res < self.work.info.pri_res) and \
                      (self.work.info.dua_res < 1e-10) or \
                      (self.work.pol.dua_res < self.work.info.dua_res) and \
                      (self.work.info.pri_res < 1e-10)

        ls = linesearch()

        if pol_success:
            # Update solver information
            self.work.info.obj_val = self.work.pol.obj_val
            self.work.info.pri_res = self.work.pol.pri_res
            self.work.info.dua_res = self.work.pol.dua_res
            self.work.info.status_polish = 1

            # Update ADMM iterations
            self.work.x = self.work.pol.x
            self.work.z = self.work.pol.z
            self.work.y = self.work.pol.y

            # Print summary
            if self.work.settings.verbose:
                self.print_polish()

        else:
            self.work.info.status_polish = -1

            # Line search on the line connecting the ADMM and the polished sol.
            ls.t = np.linspace(0., 0.002, 1000)
            ls.X, ls.Z, ls.Y = self.line_search(
                            self.work.x, self.work.z, self.work.y,
                            self.work.pol.x, self.work.pol.z, self.work.pol.y,
                            ls.t)

        return ls

    def line_search(self, x1, z1, y1, x2, z2, y2, t):
        """
        Perform line search on the line between (x1,z1,y1) and (x2,z2,y2).
        """
        N = len(t)
        X = np.zeros((N, self.work.data.n))
        Z = np.zeros((N, self.work.data.m))
        Y = np.zeros((N, self.work.data.m))

        dx = x2 - x1
        dz = z2 - z1
        dy = y2 - y1

        for i in range(N):
            X[i, :] = x1 + t[i] * dx
            Z[i, :] = z1 + t[i] * dz
            Y[i, :] = y1 + t[i] * dy
            Z[i, :], Y[i, :] = self.project_normalcone(Z[i, :], Y[i, :])

            # Unscale optimization variables (x,z,y)
            if self.work.settings.scaling:
                X[i, :] = self.work.scaling.D.dot(X[i, :])
                Z[i, :] = self.work.scaling.Einv.dot(Z[i, :])
                Y[i, :] = self.work.scaling.E.dot(Y[i, :])

        return (X, Z, Y)
