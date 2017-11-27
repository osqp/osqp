# Test osqp python module
import osqp
# import osqppurepy as osqp
import numpy as np
from scipy import sparse
import scipy as sp

# Unit Test
import unittest
import numpy.testing as nptest


class feasibility_tests(unittest.TestCase):

    def setUp(self):
        """
        Setup equality constrained feasibility problem

            min     0
            st      A x = l = u
        """
        # Simple QP problem
        sp.random.seed(4)

        self.n = 30
        self.m = 30
        self.P = sparse.csc_matrix((self.n, self.n))
        self.q = np.zeros(self.n)
        self.A = sparse.random(self.m, self.n, density=1.0).tocsc()
        self.u = np.random.rand(self.m)
        self.l = self.u
        self.opts = {'verbose': False,
                     'eps_abs': 1e-06,
                     'eps_rel': 1e-06,
                     'scaling': True,
                     'alpha': 1.6,
                     'max_iter': 5000,
                     'polish': False,
                     'warm_start': True,
                     'polish_refine_iter': 4}
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

    def test_feasibility_problem(self):

        # Solve problem
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(
            res.x,
            np.array([-0.0656074, 1.04194398, 0.4756959, -1.64036689,
                      -0.34180168, -0.81696303, -1.06389178, 0.44944554,
                      -0.44829675, -1.01289944, -0.12513655, 0.02267293,
                      -1.15206474, 1.06817424, 1.18143313, 0.01690332,
                      -0.11373645, -0.48115767,  0.25373436, 0.81369707,
                      0.18883475, 0.47000419, -0.24932451, 0.09298623,
                      1.88381076, 0.77536814, -1.35971433, 0.51511176,
                      0.03317466, 0.90226419]), decimal=3)
        nptest.assert_array_almost_equal(res.y, np.zeros(self.m), decimal=3)
        nptest.assert_array_almost_equal(res.info.obj_val, 0., decimal=3)
