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
                     'eps_abs': 1e-08,
                     'eps_rel': 1e-08,
                     'scaling': True,
                     'scaling_norm': 2,
                     'scaling_iter': 3,
                     'rho': 0.01,
                     'sigma': 0.01,
                     'alpha': 1.6,
                     'max_iter': 5000,
                     'polishing': False,
                     'warm_start': True,
                     'pol_refine_iter': 4}
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

    def test_feasibility_problem(self):

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
