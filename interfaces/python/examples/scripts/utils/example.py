from future.utils import with_metaclass
import abc
from collections import OrderedDict
from scripts import utils
import numpy as np


class Example(with_metaclass(abc.ABCMeta, object)):
    """
    Example meta-class
    """
    @abc.abstractmethod
    def gen_qp_matrices(self, *args):
        pass

    @abc.abstractmethod
    def gen_cvxpy_problem(self, *args):
        pass

    @abc.abstractmethod
    def solve_problem(self, solver='osqp', osqp_settings=None):
        pass

    def run(self, osqp_settings=None):
        print("%s  example" % self.name.title(), end='')
        if hasattr(self, 'problem_name'):
            print("  %s" % self.problem_name)
        else:
            print("")
        print("---------------------------")

        # Reset random seed for repetibility
        np.random.seed(1)

        # Define statistics
        statistics = {}
        for solver in self.solvers:
            statistics[solver] = {}
            statistics[solver]['timing'] = []
            statistics[solver]['iter'] = []

        # Problem dimensions
        dimensions = {'n': [],
                      'n_var': [],
                      'm_constraints': [],
                      'nnzA': [],
                      'nnzP': []}

        for i in range(len(self.n_vec)):
            # Generate QP
            qp_matrices = self.gen_qp_matrices(self.n_vec[i])

            # Get dimensions
            if hasattr(qp_matrices, 'k'):
                if "k" not in dimensions:
                    dimensions["k"] = []
                dimensions['k'].append(qp_matrices.k)
            dimensions['n'].append(qp_matrices.n)
            dimensions['n_var'].append(qp_matrices.P.shape[0])
            dimensions['m_constraints'].append(qp_matrices.A.shape[0])
            dimensions['nnzA'].append(qp_matrices.A.nnz)
            dimensions['nnzP'].append(qp_matrices.P.nnz)

            # Solve for all the solvers
            for solver in self.solvers:
                timing, niter = self.solve_problem(qp_matrices,
                                                   solver,
                                                   osqp_settings)
                statistics[solver]['timing'].append(timing)
                statistics[solver]['iter'].append(niter)

        # Create ordered timings dictionary
        solver_timings = self.create_timings_dict(statistics)

        # Store data and plots
        self.store_data_and_plots(solver_timings, dimensions)

    def create_timings_dict(self, statistics):
        timings_dict = OrderedDict()

        for solver in self.solvers:

            # Convert solver name
            if solver == 'osqp':
                solver_name = 'OSQP (warm start)'
            elif solver == 'osqp_coldstart':
                solver_name = 'OSQP (cold start)'
            elif solver == 'osqp_no_caching':
                solver_name = 'OSQP (no caching)'
            elif solver == 'gurobi':
                solver_name = 'GUROBI'
            elif solver == 'mosek':
                solver_name = 'MOSEK'
            elif solver == 'ecos':
                solver_name = 'ECOS'
            elif solver == 'qpoases':
                solver_name = 'qpOASES'

            # Store statistics
            timings_dict[solver_name] = statistics[solver]['timing']

        return timings_dict

    def store_data_and_plots(self, solver_timings, dimensions):
        '''
        Store dimensions and timings
        '''
        utils.store_timings(self.name, solver_timings, self.n_vec, 'mean')
        utils.store_timings(self.name, solver_timings, self.n_vec, 'max')
        utils.store_dimensions(self.name, dimensions)

        '''
        Store plots
        '''
        fig_size = None  # Adapt for talk plots
        if hasattr(self, 'problem_name'):
            plot_name = self.problem_name
        else:
            plot_name = None
        utils.generate_plot(self.name, 'time', 'median', self.n_vec,
                            solver_timings,
                            fig_size=fig_size,
                            plot_name=plot_name)
        utils.generate_plot(self.name, 'time', 'total', self.n_vec,
                            solver_timings,
                            fig_size=fig_size,
                            plot_name=plot_name)
        utils.generate_plot(self.name, 'time', 'mean', self.n_vec,
                            solver_timings,
                            fig_size=fig_size,
                            plot_name=plot_name)
