# Test osqp python module
import osqp
# import osqppurepy as osqp
import numpy as np
from scipy import sparse

# Unit Test
import unittest
import numpy.testing as nptest

class basic_tests(unittest.TestCase):

    def setUp(self):

        # Simple QP problem
        self.P = sparse.csc_matrix([[11., 0.], [0., 0.]])
        self.q = np.array([3, 4])
        self.A = sparse.csc_matrix([[-1, 0], [0, -1], [-1, -3], [2, 5], [3, 4]])
        self.u = np.array([0., 0., -15, 100, 80])
        self.l = -np.inf * np.ones(len(self.u))
        self.n = self.P.shape[0]
        self.m = self.A.shape[0]
        self.opts = {'verbose': False,
                     'eps_abs': 1e-09,
                     'eps_rel': 1e-09,
                     'max_iter': 2500,
                     'rho': 0.1,
                     'adaptive_rho': False,
                     'polish': False,
                     'check_termination': 1,
                     'warm_start': True}
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **self.opts)

    def test_basic_QP(self):
        # Solve problem
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x, np.array([0., 5.]))
        nptest.assert_array_almost_equal(res.y, np.array([1.66666667, 0.,
                                                          1.33333333, 0., 0.]))
        nptest.assert_array_almost_equal(res.info.obj_val, 20.)

    def test_update_q(self):
        # Update linear cost
        q_new = np.array([10, 20])
        self.model.update(q=q_new)
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x, np.array([0., 5.]))
        nptest.assert_array_almost_equal(res.y, np.array([3.33333334, 0.,
                                                          6.66666667, 0., 0.]))
        nptest.assert_array_almost_equal(res.info.obj_val, 100.)

    def test_update_l(self):
        # Update lower bound
        l_new = -100 * np.ones(self.m)
        self.model.update(l=l_new)
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x, np.array([0., 5.]))
        nptest.assert_array_almost_equal(res.y, np.array([1.66666667, 0.,
                                                          1.33333333, 0., 0.]))
        nptest.assert_array_almost_equal(res.info.obj_val, 20.)

    def test_update_u(self):
        # Update lower bound
        u_new = 1000 * np.ones(self.m)
        self.model.update(u=u_new)
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x, np.array([-1.51515152e-01,
                                                          -3.33282828e+02]))
        nptest.assert_array_almost_equal(res.y, np.array([0., 0., 1.33333333,
                                                          0., 0.]))
        nptest.assert_array_almost_equal(res.info.obj_val, -1333.4595959614962)

    def test_update_bounds(self):
        # Update lower bound
        l_new = -100 * np.ones(self.m)
        # Update lower bound
        u_new = 1000 * np.ones(self.m)
        self.model.update(u=u_new, l=l_new)
        res = self.model.solve()

        # Assert close
        nptest.assert_array_almost_equal(res.x, np.array([-0.12727273,
                                                          -19.94909091]))
        nptest.assert_array_almost_equal(res.y, np.array([0., 0., 0.,
                                                          -0.8, 0.]))
        nptest.assert_array_almost_equal(res.info.obj_val, -80.0890909023583)

    def test_update_max_iter(self):
        self.model.update_settings(max_iter=80)
        res = self.model.solve()

        # Assert max iter reached
        self.assertEqual(res.info.status_val,
                         self.model.constant('OSQP_MAX_ITER_REACHED'))

    def test_update_check_termination(self):
        self.model.update_settings(check_termination=0)
        res = self.model.solve()

        # Assert max iter reached
        self.assertEqual(res.info.iter, self.opts['max_iter'])

    def test_update_rho(self):
        res_default = self.model.solve()

        # Setup with different rho and update
        default_opts = self.opts.copy()
        default_opts['rho'] = 0.7
        self.model = osqp.OSQP()
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u,
                         **default_opts)
        self.model.update_settings(rho=self.opts['rho'])
        res_updated_rho = self.model.solve()

        # Assert same number of iterations
        self.assertEqual(res_default.info.iter, res_updated_rho.info.iter)

    def test_update_time_limit(self):
        res = self.model.solve()
        self.assertEqual(res.info.status_val,
                         self.model.constant('OSQP_SOLVED'))

        # Ensure the solver will time out
        self.model.update_settings(time_limit=1e-6, max_iter=2000000000,
                                   check_termination=0)

        res = self.model.solve()
        self.assertEqual(res.info.status_val,
                         self.model.constant('OSQP_TIME_LIMIT_REACHED'))

    def test_upper_triangular_P(self):
        res_default = self.model.solve()

        # Get upper triangular P
        P_triu = sparse.triu(self.P, format='csc')

        # Setup and solve with upper triangular part only
        m = osqp.OSQP()
        m.setup(P=P_triu, q=self.q, A=self.A, l=self.l, u=self.u,
                **self.opts)
        res_triu = m.solve()

        # Assert equal
        nptest.assert_array_almost_equal(res_default.x, res_triu.x)
        nptest.assert_array_almost_equal(res_default.y, res_triu.y)
        nptest.assert_array_almost_equal(res_default.info.obj_val, 
                                         res_triu.info.obj_val)
