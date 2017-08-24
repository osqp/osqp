import numpy as np
import scipy.sparse as spa
import cvxpy


class RandomQPExample(object):
    '''
    Random QP example
    '''
    def __init__(self, n, seed=1):
        '''
        Generate problem in QP format and CVXPY format
        '''
        # Set random seed
        np.random.seed(seed)

        m = int(n/2)

        # Generate problem data
        self.n = n
        self.m = m
        P = spa.random(n, n, density=0.5,
                       data_rvs=np.random.randn,
                       format='csc')
        self.P = P.dot(P.T).tocsc()
        self.q = np.random.randn(n)
        self.A = spa.random(m, n, density=0.5,
                            data_rvs=np.random.randn,
                            format='csc')
        self.l = np.random.randn(m)
        self.u = np.copy(self.l)

        self.qp_problem = self._generate_qp_problem()
        self.cvxpy_problem = self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'Random QP'

    def _generate_qp_problem(self):
        '''
        Generate QP problem
        '''
        problem = {}
        problem['P'] = self.P
        problem['q'] = self.q
        problem['A'] = self.A
        problem['l'] = self.l
        problem['u'] = self.u
        problem['m'] = self.A.shape[0]
        problem['n'] = self.A.shape[1]

        return problem

    def _generate_cvxpy_problem(self):
        '''
        Generate QP problem
        '''
        x_var = cvxpy.Variable(self.n)
        objective = .5 * cvxpy.quad_form(x_var, self.P) + self.q * x_var
        constraints = [self.A * x_var <= self.u, self.A * x_var >= self.l]
        problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

        return problem

    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''

        variables = self.cvxpy_problem.variables()
        constraints = self.cvxpy_problem.constraints

        # primal solution
        x = variables[0].value.A1

        # dual solution
        y = constraints[0].dual_value.A1 - constraints[1].dual_value.A1

        return x, y
