import numpy as np
import scipy.sparse as spa
import cvxpy


class SVMExample(object):
    '''
    SVM QP example
    '''
    def __init__(self, n, seed=1):
        '''
        Generate problem in QP format and CVXPY format
        '''
        # Set random seed
        np.random.seed(seed)

        self.n = n               # Number of features
        self.m = self.n * 100    # Number of data-points

        # Generate data
        self.N = int(self.m / 2)
        self.gamma = 1.0
        self.b_svm = np.append(np.ones(self.N), -np.ones(self.N))
        A_upp = spa.random(self.N, self.n, density=.5,
                           data_rvs=np.random.randn)
        A_low = spa.random(self.N, self.n, density=.5,
                           data_rvs=np.random.randn)
        self.A_svm = spa.vstack([
            A_upp / np.sqrt(self.n) + (A_upp != 0.).astype(float) / self.n,
            A_low / np.sqrt(self.n) - (A_low != 0.).astype(float) / self.n
            ]).tocsc()

        self.qp_problem = self._generate_qp_problem()
        self.cvxpy_problem, self.cvxpy_variables = \
            self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'SVM'

    def _generate_qp_problem(self):
        '''
        Generate QP problem
        '''

        # Construct the problem
        #       minimize	 x.T * x + gamma 1.T * t
        #       subject to  t >= diag(b) A x + 1
        #                   t >= 0

        P = spa.block_diag((spa.eye(self.n),
                            spa.csc_matrix((self.m, self.m))), format='csc')
        q = np.append(np.zeros(self.n), (self.gamma/2) * np.ones(self.m))
        A = spa.vstack([spa.hstack([spa.diags(self.b_svm).dot(self.A_svm),
                                    -spa.eye(self.m)]),
                        spa.hstack([spa.csc_matrix((self.m, self.n)),
                                    spa.eye(self.m)])
                        ]).tocsc()
        l = np.hstack([-np.inf*np.ones(self.m), np.zeros(self.m)])
        u = np.hstack([-np.ones(self.m), np.inf*np.ones(self.m)])

        # Constraints without bounds
        A_nobounds = spa.hstack([spa.diags(self.b_svm).dot(self.A_svm),
                                 -spa.eye(self.m)]).tocsc()
        l_nobounds = -np.inf*np.ones(self.m)
        u_nobounds = -np.ones(self.m)
        bounds_idx = np.arange(self.n, self.n + self.m)

        # Separate bounds
        lx = np.append(-np.inf*np.ones(self.n), np.zeros(self.m))
        ux = np.append(np.inf*np.ones(self.n), np.inf*np.ones(self.m))

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

        n = self.n
        m = self.m
        x = cvxpy.Variable(n)
        t = cvxpy.Variable(m)

        objective = cvxpy.Minimize(.5 * cvxpy.quad_form(x, spa.eye(n))
                                   + .5 * self.gamma * np.ones(m) * t)
        constraints = [t >= spa.diags(self.b_svm).dot(self.A_svm) * x + 1,
                       t >= 0]

        problem = cvxpy.Problem(objective, constraints)

        return problem, (x, t)

    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''

        (x_cvx, t_cvx) = self.cvxpy_variables
        constraints = self.cvxpy_problem.constraints

        # primal solution
        x = np.concatenate((x_cvx.value.A1,
                            t_cvx.value.A1))

        # dual solution
        y = np.concatenate((constraints[0].dual_value.A1,
                            -constraints[1].dual_value.A1))

        return x, y
