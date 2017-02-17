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


class basic_tests(unittest.TestCase):

    def setUp(self):
        # Simple QP problem
        sp.random.seed(4)

        self.P = sparse.csc_matrix(np.array([[11., 0.], [0., 0.]]))
        self.q = np.array([3, 4])
        self.A = sparse.csc_matrix(np.array([[-1, 0], [0, -1], [-1, -3],
                                             [2, 5], [3, 4]]))
        self.uA = np.array([0., 0., -15, 100, 80])
        self.lA = -np.inf * np.ones(len(self.uA))
        self.n = self.P.shape[0]
        self.m = self.A.shape[0]
        self.opts = {'verbose': False,
                     'eps_abs': 1e-09,
                     'eps_rel': 1e-09,
                     'scaling': True,
                     'scaling_norm': 2,
                     'scaling_iter': 3,
                     'rho': 0.1,
                     'alpha': 1.6,
                     'max_iter': 3000,
                     'polishing': False,
                     'warm_start': True,
                     'pol_refine_iter': 4}
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.lA, u=self.uA,
                         **self.opts)

    def test_basic_QP(self):
        # Solve problem
        res = self.model.solve()

        # solve problem with gurobi
        qp_prob = mpbpy.QuadprogProblem(self.P, self.q,
                                        self.A, self.lA, self.uA)
        resGUROBI = qp_prob.solve(solver=mpbpy.GUROBI, verbose=False)

        # Assert close
        nptest.assert_array_almost_equal(res.x, resGUROBI.x)
        nptest.assert_array_almost_equal(res.y, resGUROBI.y)
        nptest.assert_array_almost_equal(res.info.obj_val, resGUROBI.objval)

    def test_update_q(self):
        # Update linear cost
        q_new = np.array([10, 20])
        self.model.update(q=q_new)
        res = self.model.solve()

        # solve problem with gurobi
        qp_prob = mpbpy.QuadprogProblem(self.P, q_new,
                                        self.A, self.lA, self.uA)
        resGUROBI = qp_prob.solve(solver=mpbpy.GUROBI, verbose=False)

        # Assert close
        nptest.assert_array_almost_equal(res.x, resGUROBI.x)
        nptest.assert_array_almost_equal(res.y, resGUROBI.y)
        nptest.assert_array_almost_equal(res.info.obj_val, resGUROBI.objval)

    def test_update_l(self):
        # Update lower bound
        l_new = -100 * np.ones(self.m)
        self.model.update(l=l_new)
        res = self.model.solve()

        # solve problem with gurobi
        qp_prob = mpbpy.QuadprogProblem(self.P, self.q,
                                        self.A, l_new, self.uA)
        resGUROBI = qp_prob.solve(solver=mpbpy.GUROBI, verbose=False)

        # Assert close
        nptest.assert_array_almost_equal(res.x, resGUROBI.x)
        nptest.assert_array_almost_equal(res.y, resGUROBI.y)
        nptest.assert_array_almost_equal(res.info.obj_val, resGUROBI.objval)

    def test_update_u(self):
        # Update lower bound
        u_new = 1000 * np.ones(self.m)
        self.model.update(u=u_new)
        res = self.model.solve()

        # solve problem with gurobi
        qp_prob = mpbpy.QuadprogProblem(self.P, self.q,
                                        self.A, self.lA, u_new)
        resGUROBI = qp_prob.solve(solver=mpbpy.GUROBI, verbose=False)

        # Assert close
        nptest.assert_array_almost_equal(res.x, resGUROBI.x)
        nptest.assert_array_almost_equal(res.y, resGUROBI.y)
        nptest.assert_array_almost_equal(res.info.obj_val, resGUROBI.objval)

    def test_update_bounds(self):
        # Update lower bound
        l_new = -100 * np.ones(self.m)
        # Update lower bound
        u_new = 1000 * np.ones(self.m)
        self.model.update(u=u_new, l=l_new)
        res = self.model.solve()

        # solve problem with gurobi
        qp_prob = mpbpy.QuadprogProblem(self.P, self.q,
                                        self.A, l_new, u_new)
        resGUROBI = qp_prob.solve(solver=mpbpy.GUROBI, verbose=False)

        # Assert close
        nptest.assert_array_almost_equal(res.x, resGUROBI.x)
        nptest.assert_array_almost_equal(res.y, resGUROBI.y)
        nptest.assert_array_almost_equal(res.info.obj_val, resGUROBI.objval)

    def test_update_max_iter(self):
        self.model.update_settings(max_iter=10)
        res = self.model.solve()

        # Assert max iter reached
        self.assertEqual(res.info.status_val,
                         self.model.constant('OSQP_MAX_ITER_REACHED'))
