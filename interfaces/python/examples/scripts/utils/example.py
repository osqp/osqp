from future.utils import with_metaclass
import abc
from collections import OrderedDict
from scripts import utils

class Example(with_metaclass(abc.ABCMeta, object)):
    """
    Example class
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

    @abc.abstractmethod
    def run(self, osqp_settings=None):
        pass

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
        cols_dims = ['n', 'm', 'nnzA', 'nnzP']
        utils.store_timings(self.name, solver_timings, self.n_vec, 'mean')
        utils.store_timings(self.name, solver_timings, self.n_vec, 'max')
        utils.store_dimensions(self.name, dimensions, cols_dims)

        '''
        Store plots
        '''
        fig_size = None  # Adapt for talk plots

        utils.generate_plot(self.name, 'time', 'median', self.n_vec,
                            solver_timings,
                            fig_size=fig_size)
        utils.generate_plot(self.name, 'time', 'total', self.n_vec,
                            solver_timings,
                            fig_size=fig_size)
        utils.generate_plot(self.name, 'time', 'mean', self.n_vec,
                            solver_timings,
                            fig_size=fig_size)
