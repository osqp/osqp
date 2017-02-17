# Test osqp python module
import osqp
import scipy.sparse as spspa
import scipy as sp
import numpy as np

# Unit Test
import unittest


class infeeasibility_tests(unittest.TestCase):

    def setUp(self):
        sp.random.seed(6)
        """
        Setup infeasible problem
        """

        self.opts = {'verbose': False,
                     'eps_abs': 1e-05,
                     'eps_rel': 1e-05,
                     'scaling': True,
                     'scaling_norm': 2,
                     'scaling_iter': 3,
                     'rho': 1.6,
                     'alpha': 1.6,
                     'max_iter': 2500,
                     'polishing': False}

    def test_infeasibility_problem(self):

        # Simple QP problem
        sp.random.seed(4)

        self.n = 50
        self.m = 500
        # Generate random Matrices
        Pt = spspa.random(self.n, self.n)
        self.P = Pt.T.dot(Pt).tocsc()
        self.q = sp.randn(self.n)
        self.A = spspa.random(self.m, self.n).tolil()  # Lil for efficiency
        self.u = 3 + sp.randn(self.m)
        self.l = -3 + sp.randn(self.m)

        # Make random problem infeasible
        self.A[int(self.n/2), :] = self.A[int(self.n/2)+1, :]
        self.l[int(self.n/2)] = self.u[int(self.n/2)+1] + 10 * sp.rand()
        self.u[int(self.n/2)] = self.l[int(self.n/2)] + 0.5

        # Convert A to csc
        self.A = self.A.tocsc()

        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

        # Solve problem with OSQP
        res = self.model.solve()

        # Assert close
        self.assertEqual(res.info.status_val,
                         self.model.constant('OSQP_INFEASIBLE'))

    def test_infeasible_and_unbounded_problem(self):

        self.n = 2
        self.m = 4
        self.P = spspa.csc_matrix((2, 2))
        self.q = np.array([-1., -1.])
        self.A = spspa.csc_matrix([[1., -1.], [-1., 1.], [1., 0.], [0., 1.]])
        self.l = np.array([1., 1., 0., 0.])
        self.u = np.inf * np.ones(self.m)

        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

        res = self.model.solve()

        # Assert close
        self.assertEqual(res.info.status_val,
                         self.model.constant('OSQP_INFEASIBLE'))
