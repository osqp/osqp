"""
OSQP solver pure python implementation
"""
from builtins import object
import osqppurepy._osqp as _osqp # Internal low level module
from warnings import warn
import numpy as np
from scipy import sparse


class OSQP(object):
    def __init__(self):
        self._model = _osqp.OSQP()

    def version(self):
        return self._model.version()

    def setup(self, P=None, q=None, A=None, l=None, u=None, **settings):
        """
        Setup OSQP solver problem of the form

        minimize     1/2 x' * P * x + q' * x
        subject to   l <= A * x <= u

        solver settings can be specified as additional keyword arguments
        """

        #
        # Get problem dimensions
        #

        if P is None:
            if q is not None:
                n = len(q)
            elif A is not None:
                n = A.shape[1]
            else:
                raise ValueError("The problem does not have any variables")
        else:
            n = P.shape[0]
        if A is None:
            m = 0
        else:
            m = A.shape[0]

        #
        # Create parameters if they are None
        #

        if (A is None and (l is not None or u is not None)) or \
                (A is not None and (l is None and u is None)):
            raise ValueError("A must be supplied together " +
                             "with at least one bound l or u")

        # Add infinity bounds in case they are not specified
        if A is not None and l is None:
            l = -np.inf * np.ones(A.shape[0])
        if A is not None and u is None:
            u = np.inf * np.ones(A.shape[0])

        # Create elements if they are not specified
        if P is None:
            P = sparse.csc_matrix((np.zeros((0,), dtype=np.double),
                                  np.zeros((0,), dtype=np.int),
                                  np.zeros((n+1,), dtype=np.int)),
                                  shape=(n, n))
        if q is None:
            q = np.zeros(n)

        if A is None:
            A = sparse.csc_matrix((np.zeros((0,), dtype=np.double),
                                  np.zeros((0,), dtype=np.int),
                                  np.zeros((n+1,), dtype=np.int)),
                                  shape=(m, n))
            l = np.zeros(A.shape[0])
            u = np.zeros(A.shape[0])

        #
        # Check vector dimensions (not checked from C solver)
        #

        # Check if second dimension of A is correct
        # if A.shape[1] != n:
        #     raise ValueError("Dimension n in A and P does not match")
        if len(q) != n:
            raise ValueError("Incorrect dimension of q")
        if len(l) != m:
            raise ValueError("Incorrect dimension of l")
        if len(u) != m:
            raise ValueError("Incorrect dimension of u")

        #
        # Check or Sparsify Matrices
        #
        if not sparse.issparse(P) and isinstance(P, np.ndarray) and \
                len(P.shape) == 2:
            raise TypeError("P is required to be a sparse matrix")
        if not sparse.issparse(A) and isinstance(A, np.ndarray) and \
                len(A.shape) == 2:
            raise TypeError("A is required to be a sparse matrix")

        # Convert matrices in CSC form and to individual pointers
        if not sparse.isspmatrix_csc(P):
            warn("Converting sparse P to a CSC " +
                 "(compressed sparse column) matrix. (It may take a while...)")
            P = P.tocsc()
        if not sparse.isspmatrix_csc(A):
            warn("Converting sparse A to a CSC " +
                 "(compressed sparse column) matrix. (It may take a while...)")
            A = A.tocsc()

        # Check if P an A have sorted indices
        if not P.has_sorted_indices:
            P.sort_indices()
        if not A.has_sorted_indices:
            A.sort_indices()

        # Convert infinity values to OSQP Infinity
        u = np.minimum(u, self._model.constant('OSQP_INFTY'))
        l = np.maximum(l, -self._model.constant('OSQP_INFTY'))

        self._model.setup((n, m), P.data, P.indices, P.indptr, q,
                          A.data, A.indices, A.indptr,
                          l, u, **settings)

    def update(self, **kwargs):
        """
        Update OSQP problem arguments

        Vectors q, l, u are supported
        """

        # get arguments
        P = kwargs.pop('P', None)
        A = kwargs.pop('A', None)
        q = kwargs.pop('q', None)
        l = kwargs.pop('l', None)
        u = kwargs.pop('u', None)

        # Get problem dimensions
        (n, m) = (self._model.work.data.n, self._model.work.data.m)

        if P is not None:
            if P.shape != (n, n):
                raise ValueError("P must have shape (n x n)")
            if A is None:
                self._model.update_P(P)

        if A is not None:
            if A.shape != (m, n):
                raise ValueError("A must have shape (m x n)")
            if P is None:
                self._model.update_A(A)

        if P is not None and A is not None:
            self._model.update_P_A(P, A)

        if q is not None:
            if len(q) != n:
                raise ValueError("q must have length n")
            self._model.update_lin_cost(q)

        if l is not None:
            if len(l) != m:
                raise ValueError("l must have length m")

            # Convert values to OSQP_INFTY
            l = np.maximum(l, -self._model.constant('OSQP_INFTY'))

            if u is None:
                self._model.update_lower_bound(l)

        if u is not None:
            if len(u) != m:
                raise ValueError("u must have length m")

            # Convert values to OSQP_INFTY
            u = np.minimum(u, self._model.constant('OSQP_INFTY'))

            if l is None:
                self._model.update_upper_bound(u)

        if l is not None and u is not None:
            self._model.update_bounds(l, u)

    def update_settings(self, **kwargs):
        """
        Update OSQP solver settings

        It is possible to change: 'max_iter', 'eps_abs', 'eps_rel', 'rho, 'alpha',
                                  'delta', 'polish', 'polish_refine_iter',
                                  'verbose', 'scaled_termination',
                                  'check_termination'
        """

        # get arguments
        max_iter = kwargs.pop('max_iter', None)
        eps_abs = kwargs.pop('eps_abs', None)
        eps_rel = kwargs.pop('eps_rel', None)
        rho = kwargs.pop('rho', None)
        alpha = kwargs.pop('alpha', None)
        delta = kwargs.pop('delta', None)
        polish = kwargs.pop('polish', None)
        polish_refine_iter = kwargs.pop('polish_refine_iter', None)
        verbose = kwargs.pop('verbose', None)
        scaled_termination = kwargs.pop('scaled_termination', None)
        check_termination = kwargs.pop('check_termination', None)
        warm_start = kwargs.pop('warm_start', None)

        # update them
        if max_iter is not None:
            self._model.update_max_iter(max_iter)

        if eps_abs is not None:
            self._model.update_eps_abs(eps_abs)

        if eps_rel is not None:
            self._model.update_eps_rel(eps_rel)

        if rho is not None:
            self._model.update_rho(rho)

        if alpha is not None:
            self._model.update_alpha(alpha)

        if delta is not None:
            self._model.update_delta(delta)

        if polish is not None:
            self._model.update_polish(polish)

        if polish_refine_iter is not None:
            self._model.update_polish_refine_iter(polish_refine_iter)

        if verbose is not None:
            self._model.update_verbose(verbose)

        if scaled_termination is not None:
            self._model.update_scaled_termination(scaled_termination)

        if check_termination is not None:
            self._model.update_check_termination(check_termination)

        if warm_start is not None:
            self._model.update_warm_start(warm_start)

        if max_iter is None and \
           eps_abs is None and \
           eps_rel is None and \
           rho is None and \
           alpha is None and \
           delta is None and \
           polish is None and \
           polish_refine_iter is None and \
           verbose is None and \
           scaled_termination is None and \
           check_termination is None and \
           warm_start is None:
            ValueError("No updatable settings has been specified!")

    def solve(self):
        """
        Solve QP Problem
        """
        # Solve QP
        return self._model.solve()

    def constant(self, constant_name):
        """
        Return solver constant
        """
        return self._model.constant(constant_name)

    def warm_start(self, x=None, y=None):
        """
        Warm start primal or dual variables
        """
        # get problem dimensions
        (n, m) = (self._model.work.data.n, self._model.work.data.m)

        if x is not None:
            if len(x)!=n:
                raise ValueError("Wrong dimension for variable x")

            if y is None:
                self._model.warm_start_x(x)

        if y is not None:
            if len(y)!=m:
                raise ValueError("Wrong dimension for variable y")

            if x is None:
                self._model.warm_start_y(y)

        if x is not None and y is not None:
            self._model.warm_start(x, y)
