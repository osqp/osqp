# Test osqp python module
import osqp
# import osqppurepy as osqp
import numpy as np
import scipy.sparse as spspa
import scipy as sp

# Unit Test
import unittest


class warm_start_tests(unittest.TestCase):

    def setUp(self):
        """
        Setup default options
        """
        self.opts = {'verbose': False,
                     'adaptive_rho': False,
                     'eps_abs': 1e-08,
                     'eps_rel': 1e-08,
                     'polish': False,
                     'check_termination': 1}

    def test_warm_start(self):

        # Big problem
        sp.random.seed(2)
        self.n = 100
        self.m = 200
        self.A = spspa.random(self.m, self.n, density=0.9).tocsc()
        self.l = -sp.rand(self.m) * 2.
        self.u = sp.rand(self.m) * 2.

        P = spspa.random(self.n, self.n, density=0.9)
        self.P = P.dot(P.T).tocsc()
        self.q = sp.randn(self.n)

        # Setup solver
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

        # Solve problem with OSQP
        res = self.model.solve()

        # Store optimal values
        x_opt = res.x
        y_opt = res.y
        tot_iter = res.info.iter

        # Warm start with zeros and check if number of iterations is the same
        self.model.warm_start(x=np.zeros(self.n), y=np.zeros(self.m))
        res = self.model.solve()
        self.assertEqual(res.info.iter, tot_iter)

        # Warm start with optimal values and check that number of iter < 10
        self.model.warm_start(x=x_opt, y=y_opt)
        res = self.model.solve()
        self.assertLess(res.info.iter, 10)
