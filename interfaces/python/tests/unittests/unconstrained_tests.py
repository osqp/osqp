# Test osqp python module
import osqp
# import osqppurepy as osqp
import numpy as np
from scipy import sparse
import scipy as sp

# Check solve problem with gurobi
import mathprogbasepy as mpbpy

# Unit Test
import unittest
import numpy.testing as nptest


class unconstrained_tests(unittest.TestCase):

    def setUp(self):
        """
        Setup unconstrained quadratic problem
        """
        # Simple QP problem
        sp.random.seed(4)

        self.n = 30
        self.m = 0
        P = sparse.diags(np.random.rand(self.n)) + 0.2*sparse.eye(self.n)
        self.P = P.tocsc()
        self.q = np.random.randn(self.n)
        self.A = sparse.csc_matrix((self.m, self.n))
        self.l = np.array([])
        self.u = np.array([])
        self.opts = {'verbose': False,
                     'eps_abs': 1e-05,
                     'eps_rel': 1e-05,
                     'scaling': True,
                     'scaling_norm': 2,
                     'scaling_iter': 3,
                     'rho': 0.1,
                     'alpha': 1.6,
                     'max_iter': 5000,
                     'polishing': True,
                     'warm_start': True,
                     'pol_refine_iter': 4}
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

    def test_unconstrained_problem(self):

        # Solve problem
        res = self.model.solve()

        # solve problem with gurobi
        qp_prob = mpbpy.QuadprogProblem(self.P, self.q,
                                        self.A, self.l, self.u)
        resGUROBI = qp_prob.solve(solver=mpbpy.GUROBI, verbose=False)

        # Assert close
        nptest.assert_array_almost_equal(res.x, resGUROBI.x)
