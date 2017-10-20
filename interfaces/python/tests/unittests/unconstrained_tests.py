# Test osqp python module
import osqp
# import osqppurepy as osqp
import numpy as np
from scipy import sparse
import scipy as sp

# Unit Test
import unittest
import numpy.testing as nptest


class unconstrained_tests(unittest.TestCase):

    def setUp(self):
        """
        Setup unconstrained quadratic problem
        """
        # Unconstrained QP problem
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
                     'eps_abs': 1e-08,
                     'eps_rel': 1e-08,
                     'polish': False}
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

    def test_unconstrained_problem(self):

        # Solve problem
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(
            res.x, np.array([
                -0.61981415, -0.06174194, 0.83824061, -0.0595013, -0.17810828,
                2.90550031, -1.8901713, -1.91191741, -3.73603446, 1.7530356,
                -1.67018181, 3.42221944, 0.61263403, -0.45838347, -0.13194248,
                2.95744794, 5.2902277, -1.42836238, -8.55123842, -0.79093815,
                0.43418189, -0.69323554, 1.15967924, -0.47821898, 3.6108927,
                0.03404309, 0.16322926, -2.17974795, 0.32458796, -1.97553574]))
        nptest.assert_array_almost_equal(res.y, np.array([]))
        nptest.assert_array_almost_equal(res.info.obj_val, -35.020288603855825)
