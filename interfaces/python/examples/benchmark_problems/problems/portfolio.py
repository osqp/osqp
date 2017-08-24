import numpy as np
import scipy.sparse as spa
import cvxpy


class PortfolioExample(object):
    '''
    Portfolio QP example
    '''
    def __init__(self, k, seed=1):
        '''
        Generate problem in QP format and CVXPY format
        '''
        # Set random seed
        np.random.seed(seed)

        self.k = k               # Number of factors
        self.n = k * 100         # Number of assets

        # Generate data
        self.F = spa.random(self.n, self.k, density=0.5,
                            data_rvs=np.random.randn, format='csc')
        self.D = spa.diags(np.random.rand(self.n) *
                           np.sqrt(self.k), format='csc')
        self.mu = np.random.randn(self.n)
        self.gamma = 1.0

        self.qp_problem = self._generate_qp_problem()
        self.cvxpy_problem = self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'Portfolio'

    def _generate_qp_problem(self):
        '''
        Generate QP problem
        '''

        # Construct the problem
        #       minimize	x' D x + y' I y - (1/gamma) * mu' x
        #       subject to  1' x = 1
        #                   F' x = y
        #                   0 <= x <= 1
        P = spa.block_diag((2 * self.D, 2 * spa.eye(self.k)), format='csc')
        q = np.append(- self.mu / self.gamma, np.zeros(self.k))
        A = spa.vstack([
                spa.hstack([spa.csc_matrix(np.ones((1, self.n))),
                           spa.csc_matrix((1, self.k))]),
                spa.hstack([self.F.T, -spa.eye(self.k)]),
                spa.hstack((spa.eye(self.n), spa.csc_matrix((self.n, self.k))))
            ]).tocsc()
        l = np.hstack([1., np.zeros(self.k), np.zeros(self.n)])
        u = np.hstack([1., np.zeros(self.k), np.ones(self.n)])

        # Constraints without bounds
        A_nobounds = spa.vstack([
                spa.hstack([spa.csc_matrix(np.ones((1, self.n))),
                            spa.csc_matrix((1, self.k))]),
                spa.hstack([self.F.T, -spa.eye(self.k)]),
                ]).tocsc()
        l_nobounds = np.hstack([1., np.zeros(self.k)])
        u_nobounds = np.hstack([1., np.zeros(self.k)])

        # Separate bounds
        lx = np.hstack([np.zeros(self.n), -np.inf * np.ones(self.k)])
        ux = np.hstack([np.ones(self.n), np.inf * np.ones(self.k)])

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
        problem['lx'] = lx
        problem['ux'] = ux

        return problem

    def _generate_cvxpy_problem(self):
        '''
        Generate QP problem
        '''

        n_var = self.F.shape[0]
        m_var = self.F.shape[1]
        x = cvxpy.Variable(n_var)
        y = cvxpy.Variable(m_var)

        objective = cvxpy.Minimize(cvxpy.quad_form(x, self.D) +
                                   cvxpy.quad_form(y, spa.eye(m_var)) +
                                   - 1 / self.gamma * (self.mu * x))
        constraints = [np.ones(n_var) * x == 1,
                       self.F.T * x == y,
                       0 <= x, x <= 1]
        problem = cvxpy.Problem(objective, constraints)

        return problem

    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''

        variables = self.cvxpy_problem.variables()
        constraints = self.cvxpy_problem.constraints

        # primal solution
        x = np.concatenate((variables[0].value.A1,
                            variables[1].value.A1))

        # dual solution
        y = np.concatenate(([constraints[0].dual_value],
                            constraints[1].dual_value.A1,
                            constraints[3].dual_value.A1 -
                            constraints[2].dual_value.A1))

        return x, y
