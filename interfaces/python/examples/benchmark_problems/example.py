import os
from multiprocessing import Pool, cpu_count
from itertools import repeat
import pandas as pd

from solvers.solvers import SOLVER_MAP
from benchmark_problems.problems.random_qp import RandomQPExample
from benchmark_problems.problems.eq_qp import EqQPExample
from benchmark_problems.problems.portfolio import PortfolioExample
from benchmark_problems.problems.lasso import LassoExample
from benchmark_problems.problems.svm import SVMExample
from benchmark_problems.problems.huber import HuberExample
from benchmark_problems.problems.control import ControlExample
from benchmark_problems.utils import make_sure_path_exists

examples = [RandomQPExample,
            EqQPExample,
            PortfolioExample,
            LassoExample,
            SVMExample,
            HuberExample,
            ControlExample]

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

    def solve(self, parallel=True):
        '''
        Solve problems of type example

        The results are stored as

            ./results/benchmark_problems/{solver}/{class}/n{dimension}.csv

        using a pandas table with fields
            - 'class': example class
            - 'solver': solver name
            - 'status': solver status
            - 'run_time': execution time
            - 'iter': number of iterations
            - 'obj_val': objective value
            - 'n': leading dimension
            - 'N': nnz dimension (nnz(P) + nnz(A))
        '''

        print("Solving %s" % self.name)
        print("-----------------")

        if parallel:
            pool = Pool(processes=min(self.n_instances, cpu_count()))

        # Iterate over all solvers
        for solver in self.solvers:
            settings = self.settings[solver]

            # Initialize solver results
            results_solver = []

            # Solution directory
            path = os.path.join('.', 'results', 'benchmark_problems',
                                solver,
                                self.name
                                )

            # Create directory for the results
            make_sure_path_exists(path)

            # Get solver file name
            solver_file_name = os.path.join(path, 'full.csv')

            for n in self.dims:

                # Check if solution already exists
                n_file_name = os.path.join(path, 'n%i.csv' % n)

                if not os.path.isfile(n_file_name):

                    if parallel:
                        instances_list = list(range(self.n_instances))
                        n_results = pool.starmap(self.solve_single_example,
                                                 zip(repeat(n),
                                                     instances_list,
                                                     repeat(solver),
                                                     repeat(settings)))
                    else:
                        n_results = []
                        for instance in range(self.n_instances):
                            n_results.append(
                                self.solve_single_example(n,
                                                          instance,
                                                          solver,
                                                          settings)
                                )

                    # Combine n_results
                    df = pd.concat(n_results)

                    # Store n_results
                    df.to_csv(n_file_name, index=False)

                else:
                    # Load from file
                    df = pd.read_csv(n_file_name)

                # Combine list of dataframes
                results_solver.append(df)

            # Create total dataframe for the solver from list
            df_solver = pd.concat(results_solver)

            # Store dataframe
            df_solver.to_csv(solver_file_name, index=False)

        if parallel:
            pool.close()  # Not accepting any more jobs on this pool
            pool.join()   # Wait for all processes to finish

    def solve_single_example(self,
                             dimension, instance_number,
                             solver, settings):
        '''
        Solve 'example' with 'solver'

        Args:
            dimension: problem leading dimension
            instance_number: number of the instance
            solver: solver name
            settings: settings dictionary for the solver

        '''

        # Create example instance
        example_instance = EXAMPLES_MAP[self.name](dimension,
                                                   instance_number)

        print(" - Solving %s with n = %i, instance = %i with solver %s" %
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

        # Return solution
        return pd.DataFrame(solution_dict)
