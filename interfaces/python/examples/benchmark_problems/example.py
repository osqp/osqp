import numpy as np
import numpy.linalg as la
import os
import errno
from multiprocessing import Pool, cpu_count
from itertools import repeat

from solvers.solvers import SOLVER_MAP
from benchmark_problems.random_qp import RandomQPExample

examples = [RandomQPExample]

EXAMPLES_MAP = {example.name: example for example in examples}


# Function to create directories
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        # Catch exception if directory created in between
        if exception.errno != errno.EEXIST:
            raise


class Example(object):
    '''
    Examples runner
    '''
    def __init__(self, name,
                 dims,
                 solvers,
                 solver_settings,
                 n_instances=10):
        self.name = name
        self.dims = dims
        self.n_instances = n_instances
        self.solvers = solvers
        self.solver_settings = solver_settings

    def solve(self, parallel=True):
        '''
        Solve problems
        '''

        print("Solving %s" % self.name)
        print("-----------------")

        for n in self.dims:

            for i in range(self.n_instances):

                # Create example with dimension n and seed i
                # N.B. i = instance number
                example_instance = EXAMPLES_MAP[self.name](n, i)

                if parallel:
                    pool = Pool(processes=cpu_count())
                    results = \
                        pool.starmap(self.solve_single_example,
                                     zip(repeat(example_instance),
                                         repeat(n),
                                         repeat(i),
                                         self.solvers,
                                         self.settings, ))
                else:
                    results = []
                    for solver in self.solvers:
                        results.append(
                            self.solve_single_example(example_instance,
                                                      n,
                                                      i,
                                                      solver,
                                                      self.settings[solver]))

    def solve_single_example(self,
                             example_instance,
                             dimension, instance_number,
                             solver, settings):
        '''
        Solve 'example' with 'solver' and store results in a csv file

        The results are stored as:
            ./results/dimension/solver/Ninstance_number.csv

        Args:
            example: example instance
            dimension: problem leading dimension
            instance_number: number of the instance
            solver: solver name
            settings: settings dictionary for the solver

        '''

        # Solution directory
        path = os.path.join('.', 'results',
                            dimension,
                            solver)

        # Create directory for the results
        make_sure_path_exists(path)

        # Check if solution already exists
        file_name = os.path.join(path, 'N%i.csv' % instance_number)

        if not os.path.isfile(file_name):

            # Solve problem
            s = SOLVER_MAP[solver](settings)
            results = s.solve(example_instance)

            # Create solution as pandas table
            
            # Store solution

    def is_qp_solution_optimal(self, x, y,
                               eps_abs=1e-03, eps_rel=1e-03):
        '''
        Check optimality condition of the QP given the
        primal-dual solution (x, y) and the tolerance eps
        '''

        # Get problem matrices
        qp_problem = self.qp_problem
        P = qp_problem.P
        q = qp_problem.q
        A = qp_problem.A
        l = qp_problem.l
        u = qp_problem.u

        # Check primal feasibility
        Ax = A.dot(x)
        eps_pri = eps_abs + eps_rel * la.norm(Ax, np.inf)
        pri_res = np.minimum(Ax - l, 0) + np.maximum(Ax - u, 0)

        if la.norm(pri_res, np.inf) > eps_pri:
            print("Error in primal residual: %.4e > %.4e" %
                  (la.norm(pri_res, np.inf), eps_pri), end='')
            return False

        # Check dual feasibility
        Px = P.dot(x)
        Aty = A.T.dot(y)
        eps_dua = eps_abs + eps_rel * np.max([la.norm(Px, np.inf),
                                              la.norm(q, np.inf),
                                              la.norm(Aty, np.inf)])
        dua_res = Px + q + Aty

        if la.norm(dua_res, np.inf) > eps_dua:
            print("Error in dual residual: %.4e > %.4e" %
                  (la.norm(dua_res, np.inf), eps_dua), end='')
            return False

        # Check complementary slackness
        y_plus = np.maximum(y, 0)
        y_minus = np.minimum(y, 0)

        eps_comp = eps_abs + eps_rel * np.max([la.norm(Ax, np.inf)])

        comp_res_u = np.minimum(y_plus, np.abs(u - Ax))
        comp_res_l = np.minimum(-y_minus, np.abs(Ax - l))

        if la.norm(comp_res_l, np.inf) > eps_comp:
            print("Error in complementary slackness residual l: %.4e > %.4e" %
                  (la.norm(comp_res_l, np.inf), eps_comp))
            return False

        if la.norm(comp_res_u, np.inf) > eps_comp:
            print("Error in complementary slackness residual u: %.4e > %.4e" %
                  (la.norm(comp_res_u, np.inf), eps_comp))
            return False
        # If we arrived until here, the solution is optimal
        return True
