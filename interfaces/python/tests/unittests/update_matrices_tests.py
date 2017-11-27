# Test osqp python module
import osqp
import numpy as np
import scipy as sp
from scipy import sparse

# Unit Test
import unittest
import numpy.testing as nptest


class update_matrices_tests(unittest.TestCase):

    def setUp(self):
        # Simple QP problem
        sp.random.seed(1)

        self.n = 5
        self.m = 8
        p = 0.7

        Pt = sparse.random(self.n, self.n, density=p)
        Pt_new = Pt.copy()
        Pt_new.data += 0.1 * np.random.randn(Pt.nnz)

        self.P = Pt.T.dot(Pt).tocsc() + sparse.eye(self.n).tocsc()
        self.P_new = Pt_new.T.dot(Pt_new).tocsc() + sparse.eye(self.n).tocsc()
        self.q = np.random.randn(self.n)
        self.A = sparse.random(self.m, self.n, density=p).tocsc()
        self.A_new = self.A.copy()
        self.A_new.data += np.random.randn(self.A_new.nnz)
        self.l = np.zeros(self.m)
        self.u = 30 + np.random.randn(self.m)
        self.opts = {'eps_abs': 1e-08,
                     'eps_rel': 1e-08,
                     'verbose': False}
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

    def test_solve(self):
        # Solve problem
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x,
            np.array([0.85459329, 0.73472366, 0.06156, -0.06095794, -0.96167612]))
        nptest.assert_array_almost_equal(res.y,
            np.array([0., 0., 0., -2.32275314, 0., -0.93213354, 0., -0.8939565]))
        nptest.assert_array_almost_equal(res.info.obj_val, -1.5116431929127323)

    def test_update_P(self):
        # Update matrix P
        Pnew_triu = sparse.triu(self.P_new).tocsc()
        Px = Pnew_triu.data
        Px_idx = np.arange(Pnew_triu.nnz)
        self.model.update(Px=Px, Px_idx=Px_idx)
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x,
            np.array([0.79105808, 0.68008954, -0.00974931, -0.00569589, -0.92142316]))
        nptest.assert_array_almost_equal(res.y,
            np.array([0., -0.0977256, 0., -2.3051196, 0., -0.84705904, 0., -0.9014214]))
        nptest.assert_array_almost_equal(res.info.obj_val, -1.40901946656505)

    def test_update_P_allind(self):
        # Update matrix P
        Pnew_triu = sparse.triu(self.P_new).tocsc()
        Px = Pnew_triu.data
        self.model.update(Px=Px)
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x,
            np.array([0.79105808, 0.68008954, -0.00974931, -0.00569589, -0.92142316]))
        nptest.assert_array_almost_equal(res.y,
            np.array([0., -0.0977256, 0., -2.3051196, 0., -0.84705904, 0., -0.9014214]))
        nptest.assert_array_almost_equal(res.info.obj_val, -1.40901946656505)

    def test_update_A(self):
        # Update matrix A
        Ax = self.A_new.data
        Ax_idx = np.arange(self.A_new.nnz)
        self.model.update(Ax=Ax, Ax_idx=Ax_idx)
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x,
            np.array([0.44557958, 0.11209195, 0.22051994, -0.78051077, -0.01697192]))
        nptest.assert_array_almost_equal(res.y,
            np.array([-1.97318457, -1.43719371, 0., -0.05364337, -1.3354648, 0., 0., 0.]))
        nptest.assert_array_almost_equal(res.info.obj_val, -0.79990427087463)

    def test_update_A_allind(self):
        # Update matrix A
        Ax = self.A_new.data
        self.model.update(Ax=Ax)
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x,
            np.array([0.44557958, 0.11209195, 0.22051994, -0.78051077, -0.01697192]))
        nptest.assert_array_almost_equal(res.y,
            np.array([-1.97318457, -1.43719371, 0., -0.05364337, -1.3354648, 0., 0., 0.]))
        nptest.assert_array_almost_equal(res.info.obj_val, -0.79990427087463)

    def test_update_P_A_indP_indA(self):
        # Update matrices P and A
        Pnew_triu = sparse.triu(self.P_new).tocsc()
        Px = Pnew_triu.data
        Px_idx = np.arange(Pnew_triu.nnz)
        Ax = self.A_new.data
        Ax_idx = np.arange(self.A_new.nnz)
        self.model.update(Px=Px, Px_idx=Px_idx, Ax=Ax, Ax_idx=Ax_idx)
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x,
            np.array([0.45599336, 0.11471169, 0.22567378, -0.80654725, -0.01778191]))
        nptest.assert_array_almost_equal(res.y,
            np.array([-1.76495387, -1.44638239, 0., 0., -1.28476339, 0., 0., 0.]))
        nptest.assert_array_almost_equal(res.info.obj_val, -0.8249598023368026)

    def test_update_P_A_indP(self):
        # Update matrices P and A
        Pnew_triu = sparse.triu(self.P_new).tocsc()
        Px = Pnew_triu.data
        Px_idx = np.arange(Pnew_triu.nnz)
        Ax = self.A_new.data
        self.model.update(Px=Px, Px_idx=Px_idx, Ax=Ax)
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x,
            np.array([0.45599336, 0.11471169, 0.22567378, -0.80654725, -0.01778191]))
        nptest.assert_array_almost_equal(res.y,
            np.array([-1.76495387, -1.44638239, 0., 0., -1.28476339, 0., 0., 0.]))
        nptest.assert_array_almost_equal(res.info.obj_val, -0.8249598023368026)

    def test_update_P_A_indA(self):
        # Update matrices P and A
        Pnew_triu = sparse.triu(self.P_new).tocsc()
        Px = Pnew_triu.data
        Ax = self.A_new.data
        Ax_idx = np.arange(self.A_new.nnz)
        self.model.update(Px=Px, Ax=Ax, Ax_idx=Ax_idx)
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x,
            np.array([0.45599336, 0.11471169, 0.22567378, -0.80654725, -0.01778191]))
        nptest.assert_array_almost_equal(res.y,
            np.array([-1.76495387, -1.44638239, 0., 0., -1.28476339, 0., 0., 0.]))
        nptest.assert_array_almost_equal(res.info.obj_val, -0.8249598023368026)

    def test_update_P_A_allind(self):
        # Update matrices P and A
        Pnew_triu = sparse.triu(self.P_new).tocsc()
        Px = Pnew_triu.data
        Ax = self.A_new.data
        self.model.update(Px=Px, Ax=Ax)
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x,
            np.array([0.45599336, 0.11471169, 0.22567378, -0.80654725, -0.01778191]))
        nptest.assert_array_almost_equal(res.y,
            np.array([-1.76495387, -1.44638239, 0., 0., -1.28476339, 0., 0., 0.]))
        nptest.assert_array_almost_equal(res.info.obj_val, -0.8249598023368026)
