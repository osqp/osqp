import os
from multiprocessing import Pool, cpu_count
from itertools import repeat
import pandas as pd

from solvers.solvers import SOLVER_MAP
from benchmark_problems.problems.random_qp import RandomQPExample
from benchmark_problems.problems.eq_qp import EqQPExample
from benchmark_problems.problems.portfolio import PortfolioExample
from benchmark_problems.problems.lasso import LassoExample
from benchmark_problems.utils import make_sure_path_exists

examples = [RandomQPExample, EqQPExample, PortfolioExample, LassoExample]

EXAMPLES_MAP = {example.name(): example for example in examples}


class Example(object):
    '''
    Examples runner
    '''
    def __init__(self, name,
                 dims,
                 solvers,
                 settings,
                 n_instances=10):
        self.name = name
        self.dims = dims
        self.n_instances = n_instances
        self.solvers = solvers
        self.settings = settings

    def solve(self, parallel=False):
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
                    settings_list = \
                        [self.settings[solver] for solver in self.solvers]
                    pool = Pool(processes=cpu_count())
                    pool.starmap(self.solve_single_example,
                                 zip(repeat(example_instance),
                                     repeat(n),
                                     repeat(i),
                                     self.solvers,
                                     settings_list))
                else:
                    for solver in self.solvers:
                        self.solve_single_example(example_instance,
                                                  n,
                                                  i,
                                                  solver,
                                                  self.settings[solver])

    def solve_single_example(self,
                             example_instance,
                             dimension, instance_number,
                             solver, settings):
        '''
        Solve 'example' with 'solver' and store results in a csv file

        The results are stored as

            ./results/benchmark_problems/{solver}/{class}/n{dimension}/i{instance_number}.csv

        using a pandas table with fields
            - 'class': example class
            - 'solver': solver name
            - 'status': solver status
            - 'run_time': execution time
            - 'iter': number of iterations
            - 'obj_val': objective value
            - 'n': leading dimension
            - 'N': nnz dimension (nnz(P) + nnz(A))

        Args:
            example: example instance
            dimension: problem leading dimension
            instance_number: number of the instance
            solver: solver name
            settings: settings dictionary for the solver

        '''
        # Solution directory
        path = os.path.join('.', 'results', 'benchmark_problems',
                            solver,
                            self.name,
                            'n%i' % dimension
                            )

        # Create directory for the results
        make_sure_path_exists(path)

        # Check if solution already exists
        file_name = os.path.join(path, 'i%i.csv' % instance_number)

        if not os.path.isfile(file_name):

            print(" - Solving %s with n=%i, instance=%i with solver %s" %
                  (self.name, dimension, instance_number, solver))

            # Solve problem
            s = SOLVER_MAP[solver](settings)
            results = s.solve(example_instance)

            # Create solution as pandas table
            P = example_instance.qp_problem['P']
            A = example_instance.qp_problem['A']
            N = P.nnz + A.nnz
            solution_dict = {'class': [self.name],
                             'solver': [solver],
                             'status': [results.status],
                             'run_time': [results.run_time],
                             'iter': [results.niter],
                             'obj_val': [results.obj_val],
                             'n': [dimension],
                             'N': [N]}

            # Add status polish if OSQP
            if solver[:4] == 'OSQP':
                solution_dict['status_polish'] = results.status_polish

            # Store solution
            df = pd.DataFrame(solution_dict)
            df.to_csv(file_name, index=False)
