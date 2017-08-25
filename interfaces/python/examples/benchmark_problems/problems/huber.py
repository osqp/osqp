import numpy as np
import scipy.sparse as spa
import cvxpy


class HuberExample(object):
    '''
    Huber QP example
    '''
    def __init__(self, n, seed=1):
        '''
        Generate problem in QP format and CVXPY format
        '''
        # Set random seed
        np.random.seed(seed)

        self.n = n               # Number of features
        self.m = self.n * 100    # Number of data-points

        self.Ad = spa.random(self.m, self.n, density=0.5,
                             data_rvs=np.random.randn)
        self.x_true = np.random.randn(n) / np.sqrt(n)
        ind95 = (np.random.rand(self.m) < 0.95).astype(float)
        self.bd = self.Ad.dot(self.x_true) + \
            np.multiply(0.5*np.random.randn(self.m), ind95) \
            + np.multiply(10.*np.random.rand(self.m), 1. - ind95)

        self.qp_problem = self._generate_qp_problem()
        self.cvxpy_problem, self.cvxpy_variables = \
            self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'Huber'

    def _generate_qp_problem(self):
        '''
        Generate QP problem
        '''
        # Construct the problem
        #       minimize	1/2 u.T * u + np.ones(m).T * v
        #       subject to  -u - v <= Ax - b <= u + v
        #                   0 <= u <= 1
        #                   v >= 0
        Im = spa.eye(self.m)
        P = spa.block_diag((spa.csc_matrix((self.n, self.n)), Im,
                            spa.csc_matrix((self.m, self.m))), format='csc')
        q = np.append(np.zeros(self.m + self.n), np.ones(self.m))
        A = spa.vstack([
                spa.hstack([self.Ad, Im, Im]),
                spa.hstack([self.Ad, -Im, -Im]),
                spa.hstack([spa.csc_matrix((2 * self.m, self.n)),
                            spa.eye(2 * self.m)])
                ]).tocsc()
        l = np.hstack([self.bd,
                       -np.inf*np.ones(self.m),
                       np.zeros(2 * self.m)])
        u = np.hstack([np.inf*np.ones(self.m),
                       self.bd,
                       np.ones(self.m),
                       np.inf*np.ones(self.m)])

        # Constraints without bounds
        A_nobounds = spa.vstack([
                spa.hstack([self.Ad, Im, Im]),
                spa.hstack([self.Ad, -Im, -Im])
                ]).tocsc()
        l_nobounds = np.hstack([self.bd, -np.inf*np.ones(self.m)])
        u_nobounds = np.hstack([np.inf*np.ones(self.m), self.bd])

        # Bounds
        lx = np.hstack([-np.inf * np.ones(self.n),
                        np.zeros(2 * self.m)])
        ux = np.hstack([np.inf * np.ones(self.n),
                        np.ones(self.m),
                        np.inf*np.ones(self.m)])
        bounds_idx = np.arange(self.n, self.n + 2 * self.m)

        problem = {}
        problem['P'] = P
        problem['q'] = q
        problem['A'] = A
        problem['l'] = l
        problem['u'] = u
        problem['m'] = A.shape[0]
        problem['n'] = A.shape[1]
        problem['A_nobounds'] = A_nobounds
        problem['l_nobounds'] = l_nobounds
        problem['u_nobounds'] = u_nobounds
        problem['bounds_idx'] = bounds_idx
        problem['lx'] = lx
        problem['ux'] = ux

        return problem

    def _generate_cvxpy_problem(self):
        '''
        Generate QP problem
        '''
        # Model with CVXPY
        #       minimize	1/2 u.T * u + np.ones(m).T * v
        #       subject to  -u - v <= Ax - b <= u + v
        #                   0 <= u <= 1
        #                   v >= 0
        x = cvxpy.Variable(self.n)
        u = cvxpy.Variable(self.m)
        v = cvxpy.Variable(self.m)

        objective = cvxpy.Minimize(.5 * cvxpy.quad_form(u, spa.eye(self.m))
                                   + np.ones(self.m) * v)
        constraints = [-u - v <= self.Ad * x - self.bd,
                       self.Ad * x - self.bd <= u + v,
                       0. <= u, u <= 1.,
                       v >= 0.]
        problem = cvxpy.Problem(objective, constraints)

        return problem, (x, u, v)

    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''

        (x_cvx, u_cvx, v_cvx) = self.cvxpy_variables
        constraints = self.cvxpy_problem.constraints

        # primal solution
        x = np.concatenate((x_cvx.value.A1,
                            u_cvx.value.A1,
                            v_cvx.value.A1))

        # dual solution
        y = np.concatenate((-constraints[0].dual_value.A1,
                            constraints[1].dual_value.A1,
                            constraints[3].dual_value.A1 -
                            constraints[2].dual_value.A1,
                            -constraints[4].dual_value.A1))

        return x, y
