import cvxpy.settings as stgs
import cvxpy
from . import statuses as s
from .results import Results


class ECOSSolver(object):

    STATUS_MAP = {stgs.OPTIMAL: s.OPTIMAL,
                  stgs.OPTIMAL_INACCURATE: s.OPTIMAL_INACCURATE,
                  stgs.INFEASIBLE: s.PRIMAL_INFEASIBLE,
                  stgs.INFEASIBLE_INACCURATE: s.PRIMAL_INFEASIBLE_INACCURATE,
                  stgs.UNBOUNDED_INACCURATE: s.DUAL_INFEASIBLE_INACCURATE}

    def __init__(self, settings={}):
        '''
        Initialize solver object by setting require settings
        '''
        self._settings = settings

    def name(self):
        return 'ECOS'

    @property
    def settings(self):
        """Solver settings"""
        return self._settings

    def solve(self, example):
        '''
        Solve problem

        Args:
            example: example object

        Returns:
            Results structure
        '''
        problem = example.cvxpy_problem

        if 'verbose' in self._settings:
            verbose = self._settings["verbose"]

        try:
            problem.solve(solver=cvxpy.ECOS, verbose=verbose)
        except:
            if self._settings['verbose']:
                print("Error in ECOS solution\n")
            return Results(s.SOLVER_ERROR, None, None, None,
                           None, None)

        status = self.STATUS_MAP.get(problem.status, s.SOLVER_ERROR)

        # Obtain time and number of iterations
        run_time = problem.solver_stats.setup_time + \
            problem.solver_stats.solve_time

        niter = problem.solver_stats.num_iters

        obj_val = problem.objective.value

        # Get primal, dual solution
        x, y = example.revert_cvxpy_solution()

        # Validate status
        if not example.is_qp_solution_optimal(x, y):
            status = s.SOLVER_ERROR

        return Results(status,
                       obj_val,
                       x,
                       y,
                       run_time,
                       niter)
