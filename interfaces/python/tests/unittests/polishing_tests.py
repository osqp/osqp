# Test osqp python module
import osqp
# import osqppurepy as osqp
import numpy as np
from scipy import sparse
import scipy as sp

# Unit Test
import unittest
import numpy.testing as nptest


class polish_tests(unittest.TestCase):

    def setUp(self):
        """
        Setup default options
        """
        self.opts = {'verbose': False,
                     'eps_abs': 1e-03,
                     'eps_rel': 1e-03,
                     'scaling': True,
                     'rho': 0.1,
                     'alpha': 1.6,
                     'max_iter': 2500,
                     'polish': True,
                     'polish_refine_iter': 4}

    def test_polish_simple(self):

        # Simple QP problem
        self.P = sparse.csc_matrix([[11, 0], [0, 0]])
        self.q = np.array([3, 4])
        self.A = sparse.csc_matrix([[-1, 0], [0, -1], [-1, -3], [2, 5], [3, 4]])
        self.u = np.array([0, 0, -15, 100, 80])
        self.l = -np.inf * np.ones(len(self.u))
        self.n = self.P.shape[0]
        self.m = self.A.shape[0]
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

        # Solve problem
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x, np.array([0., 5.]))
        nptest.assert_array_almost_equal(res.y, np.array([1.66666667, 0.,
                                                          1.33333333, 0., 0.]))
        nptest.assert_array_almost_equal(res.info.obj_val, 20.)

    def test_polish_unconstrained(self):

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
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

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

    def test_polish_random(self):

        # Random QP problem
        sp.random.seed(6)

        self.n = 30
        self.m = 50
        Pt = sp.randn(self.n, self.n)
        self.P = sparse.csc_matrix(np.dot(Pt.T, Pt))
        self.q = sp.randn(self.n)
        self.A = sparse.csc_matrix(sp.randn(self.m, self.n))
        self.l = -3 + sp.randn(self.m)
        self.u = 3 + sp.randn(self.m)
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

        # Solve problem
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(
            res.x, np.array([
                -0.58549607, 0.0030388, -0.07154039, -0.0406463, -0.13349925,
                -0.1354755, -0.17417362, 0.0165324, -0.12213118, -0.10477034,
                -0.51748662, -0.05310921, 0.07862616, 0.53663003, -0.01459859,
                0.40678716, -0.03496123, 0.25722838, 0.06335071, 0.29908295,
                -0.6223218, -0.07614658, -0.3892153, -0.18111635, 0.56301768,
                0.10429917, 0.09821862, -0.30881928, 0.24430531, 0.06597486]))
        nptest.assert_array_almost_equal(
            res.y, np.array([
                0., -2.11407101e-01, 0., 0., 0., 0., 0., 0., 0.,
                0., -3.78854588e-02, 0., -1.58346998e-02, 0., 0.,
                -6.88711599e-02, 0., 0., 0., 0., 0., 0., 0., 0.,
                6.04385132e-01, 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 1.37995470e-01, 0., 0., 0.,  -2.04427802e-02,
                0., -1.32983915e-01, 0., 2.94425952e-02, 0., 0.,
                0., 0., 0., -6.53409219e-02, 0.]))
        nptest.assert_array_almost_equal(res.info.obj_val, -3.262280663471232)
