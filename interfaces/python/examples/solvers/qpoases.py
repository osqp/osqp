import qpoases
import numpy as np
from . import statuses as s
from .results import Results


class qpOASESSolver(object):

    # Get return value class
    _PyReturnValue = qpoases.PyReturnValue()
    STATUS_MAP = {_PyReturnValue.SUCCESSFUL_RETURN: s.OPTIMAL,
                  _PyReturnValue.INIT_FAILED_INFEASIBILITY:
                  s.PRIMAL_INFEASIBLE,
                  _PyReturnValue.INIT_FAILED_UNBOUNDEDNESS: s.DUAL_INFEASIBLE,
                  _PyReturnValue.MAX_NWSR_REACHED: s.MAX_ITER_REACHED,
                  _PyReturnValue.INIT_FAILED: s.SOLVER_ERROR
                  }

    def __init__(self, settings={}):
        '''
        Initialize solver object by setting require settings
        '''
        self._settings = settings

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
        p = example.problem
        n, m = p.P.shape[0], p.A.shape[0]

        if p.P is not None:
            P = np.ascontiguousarray(p.P.todense())

        if p.A is not None:
            A = np.ascontiguousarray(p.A.todense())

        # Define contiguous array vectors
        q = np.ascontiguousarray(p.q)
        l = np.ascontiguousarray(p.l)
        u = np.ascontiguousarray(p.u)

        # Create infinite arrays of bounds
        lx = np.ascontiguousarray(-np.inf * np.ones(p.n))
        ux = np.ascontiguousarray(np.inf * np.ones(p.n))

        # Solve with qpOASES
        qpoases_m = qpoases.PyQProblem(n, m)
        options = qpoases.PyOptions()

        if not self._settings['verbose']:
            options.printLevel = qpoases.PyPrintLevel.NONE

        for param, value in self._settings.items():
            if param == 'verbose':
                if value is False:
                    options.printLevel = qpoases.PyPrintLevel.NONE
            elif param == 'cputime':
                qpoases_cpu_time = np.array([value])
            elif param == 'nWSR':
                qpoases_nWSR = np.array([value])
            else:
                exec("options.%s = %s" % (param, value))

        qpoases_m.setOptions(options)

        if 'cputime' not in self._settings:
            # Set default to max 10 seconds in runtime
            qpoases_cpu_time = np.array([10.])

        if 'nWSR' not in self._settings:
            # Set default to max 1000 working set recalculations
            qpoases_nWSR = np.array([1000])

        # Set number of working set recalculations
        status = qpoases_m.init(P, q, A, lx, ux, l, u,
                                qpoases_nWSR, qpoases_cpu_time)

        # Check status
        status = self.STATUS_MAP.get(status, s.SOLVER_ERROR)

        # run_time
        run_time = qpoases_cpu_time[0]

        # number of iterations
        niter = qpoases_nWSR[0]

        if status in s.SOLUTION_PRESENT:
            x = np.zeros(p.n)
            y_temp = np.zeros(p.n + p.m)
            obj_val = qpoases_m.getObjVal()
            qpoases_m.getPrimalSolution(x)
            qpoases_m.getDualSolution(y_temp)

            # Change sign and take only last part of y
            # N.B. No bounds on x in our formulation
            y = -y_temp[p.n:]

            if not example.is_qp_solution_optimal(p, x, y):
                status = s.SOLVER_ERROR

            return Results(status, obj_val,
                           x, y,
                           run_time, niter)
        else:
            return Results(status, None, None, None,
                           run_time, niter)
