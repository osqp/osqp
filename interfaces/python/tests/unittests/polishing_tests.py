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


class polishing_tests(unittest.TestCase):

    def setUp(self):
        """
        Setup default options
        """
        self.opts = {'verbose': False,
                     'eps_abs': 1e-03,
                     'eps_rel': 1e-03,
                     'scaling': True,
                     'scaling_norm': 2,
                     'scaling_iter': 3,
                     'rho': 0.1,
                     'alpha': 1.6,
                     'max_iter': 2500,
                     'polishing': True,
                     'pol_refine_iter': 4}

    def test_polish_simple(self):

        # Simple QP problem
        sp.random.seed(4)
        self.P = sparse.csc_matrix(np.array([[11., 0.], [0., 0.]]))
        self.q = np.array([3, 4])
        self.A = sparse.csc_matrix(np.array([[-1, 0], [0, -1], [-1, -3],
                                             [2, 5], [3, 4]]))
        self.u = np.array([0., 0., -15, 100, 80])
        self.l = -np.inf * np.ones(len(self.u))
        self.n = self.P.shape[0]
        self.m = self.A.shape[0]
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

        # Solve problem
        res = self.model.solve()

        # solve problem with gurobi
        qp_prob = mpbpy.QuadprogProblem(self.P, self.q,
                                        self.A, self.l, self.u)
        resGUROBI = qp_prob.solve(solver=mpbpy.GUROBI, verbose=False)

        # Assert close
        nptest.assert_array_almost_equal(res.x, resGUROBI.x)
        nptest.assert_array_almost_equal(res.y, resGUROBI.y)
        nptest.assert_array_almost_equal(res.info.obj_val, resGUROBI.objval)

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

        # solve problem with gurobi
        qp_prob = mpbpy.QuadprogProblem(self.P, self.q,
                                        self.A, self.l, self.u)
        resGUROBI = qp_prob.solve(solver=mpbpy.GUROBI, verbose=False)

        # Assert close
        nptest.assert_array_almost_equal(res.x, resGUROBI.x)
        nptest.assert_array_almost_equal(res.y, resGUROBI.y)
        nptest.assert_array_almost_equal(res.info.obj_val, resGUROBI.objval)

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

        # solve problem with gurobi
        qp_prob = mpbpy.QuadprogProblem(self.P, self.q,
                                        self.A, self.l, self.u)
        resGUROBI = qp_prob.solve(solver=mpbpy.GUROBI, verbose=False)

        # Assert close
        nptest.assert_array_almost_equal(res.x, resGUROBI.x)
        nptest.assert_array_almost_equal(res.y, resGUROBI.y)
        nptest.assert_array_almost_equal(res.info.obj_val, resGUROBI.objval)
