import mosek
import numpy as np
import scipy.sparse as spa
from . import statuses as s
from .results import Results


class MOSEKSolver(object):

    # Map of Mosek status to mathprogbasepy status.
    STATUS_MAP = {mosek.solsta.optimal: s.OPTIMAL,
                  mosek.solsta.integer_optimal: s.OPTIMAL,
                  mosek.solsta.prim_infeas_cer: s.PRIMAL_INFEASIBLE,
                  mosek.solsta.dual_infeas_cer: s.DUAL_INFEASIBLE,
                  mosek.solsta.near_optimal: s.OPTIMAL_INACCURATE,
                  mosek.solsta.near_prim_infeas_cer:
                  s.PRIMAL_INFEASIBLE_INACCURATE,
                  mosek.solsta.near_dual_infeas_cer:
                  s.DUAL_INFEASIBLE_INACCURATE,
                  mosek.solsta.unknown: s.SOLVER_ERROR}

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
        p = example.qp_problem

        # Get problem dimensions
        n = p.P.shape[0]
        m = p.A.shape[0]

        '''
        Load problem
        '''
        # Create environment
        env = mosek.Env()

        # Create optimization task
        task = env.Task()

        if self._settings['verbose']:
            # Define a stream printer to grab output from MOSEK
            def streamprinter(text):
                import sys
                sys.stdout.write(text)
                sys.stdout.flush()
            env.set_Stream(mosek.streamtype.log, streamprinter)
            task.set_Stream(mosek.streamtype.log, streamprinter)

        # Load problem into task object

        # Append 'm' empty constraints.
        # The constraints will initially have no bounds.
        task.appendcons(m)

        # Append 'n' variables.
        # The variables will initially be fixed at zero (x=0).
        task.appendvars(n)

        # Add linear cost by iterating over all variables
        for j in range(n):
            task.putcj(j, p.q[j])
            task.putvarbound(j, mosek.boundkey.fr, -np.inf, np.inf)

        # Add constraints
        if p.A is not None:
            row_A, col_A, el_A = spa.find(p.A)
            task.putaijlist(row_A, col_A, el_A)

            for j in range(m):
                # Get bounds and keys
                u_temp = p.u[j] if p.u[j] < 1e20 else np.inf
                l_temp = p.l[j] if p.l[j] > -1e20 else -np.inf

                # Divide 5 cases
                if (np.abs(l_temp - u_temp) < 1e-08):
                    bound_key = mosek.boundkey.fx
                elif l_temp == -np.inf and u_temp == np.inf:
                    bound_key = mosek.boundkey.fr
                elif l_temp != -np.inf and u_temp == np.inf:
                    bound_key = mosek.boundkey.lo
                elif l_temp != -np.inf and u_temp != np.inf:
                    bound_key = mosek.boundkey.ra
                elif l_temp == -np.inf and u_temp != np.inf:
                    bound_key = mosek.boundkey.up

                # Add bound
                task.putconbound(j, bound_key, l_temp, u_temp)

        # Add quadratic cost
        if p.P.count_nonzero():  # If there are any nonzero elms in P
            P = spa.tril(p.P, format='coo')
            task.putqobj(P.row, P.col, P.data)

        # Set problem minimization
        task.putobjsense(mosek.objsense.minimize)

        '''
        Set parameters
        '''
        for param, value in self._settings.items():
            if param == 'verbose':
                if value is False:
                    self._handle_str_param(task, 'MSK_IPAR_LOG'.strip(), 0)
            else:
                if isinstance(param, str):
                    self._handle_str_param(task, param.strip(), value)
                else:
                    self._handle_enum_param(task, param, value)

        '''
        Solve problem
        '''
        try:
            # Optimization and check termination code
            task.optimize()
        except:
            if self._settings['verbose']:
                print("Error in MOSEK solution\n")
            return Results(s.SOLVER_ERROR, None, None, None,
                           None, None)

        if self._settings['verbose']:
            task.solutionsummary(mosek.streamtype.msg)

        '''
        Extract results
        '''

        # Get solution type and status
        soltype, solsta = self.choose_solution(task)

        # Map status using statusmap
        status = self.STATUS_MAP.get(solsta, s.SOLVER_ERROR)

        # Get statistics
        cputime = task.getdouinf(mosek.dinfitem.optimizer_time) + \
            task.getdouinf(mosek.dinfitem.presolve_time)
        total_iter = task.getintinf(mosek.iinfitem.intpnt_iter)

        if status in s.SOLUTION_PRESENT:
            # get primal variables values
            x = np.zeros(task.getnumvar())
            task.getxx(soltype, x)
            # get obj value
            objval = task.getprimalobj(soltype)
            # get dual
            y = np.zeros(task.getnumcon())
            task.gety(soltype, y)
            # it appears signs are inverted
            y = -y

            if not example.is_qp_solution_optimal(p, x, y):
                status = s.SOLVER_ERROR

            return Results(status, objval, x, y,
                           cputime, total_iter)
        else:
            return Results(status, None, None, None,
                           cputime, None)

    def choose_solution(self, task):
        """Chooses between the basic, interior point solution or integer solution
        Parameters
        N.B. From CVXPY
        ----------
        task : mosek.Task
            The solver status interface.
        Returns
        -------
        soltype
            The preferred solution (mosek.soltype.*)
        solsta
            The status of the preferred solution (mosek.solsta.*)
        """
        import mosek

        def rank(status):
            # Rank solutions
            # optimal > near_optimal > anything else > None
            if status == mosek.solsta.optimal:
                return 3
            elif status == mosek.solsta.near_optimal:
                return 2
            elif status is not None:
                return 1
            else:
                return 0

        solsta_bas, solsta_itr = None, None

        # Integer solution
        if task.solutiondef(mosek.soltype.itg):
            solsta_itg = task.getsolsta(mosek.soltype.itg)
            return mosek.soltype.itg, solsta_itg

        # Continuous solution
        if task.solutiondef(mosek.soltype.bas):
            solsta_bas = task.getsolsta(mosek.soltype.bas)

        if task.solutiondef(mosek.soltype.itr):
            solsta_itr = task.getsolsta(mosek.soltype.itr)

        # As long as interior solution is not worse, take it
        # (for backward compatibility)
        if rank(solsta_itr) >= rank(solsta_bas):
            return mosek.soltype.itr, solsta_itr
        else:
            return mosek.soltype.bas, solsta_bas

    @staticmethod
    def _handle_str_param(task, param, value):
        if param.startswith("MSK_DPAR_"):
            task.putnadouparam(param, value)
        elif param.startswith("MSK_IPAR_"):
            task.putnaintparam(param, value)
        elif param.startswith("MSK_SPAR_"):
            task.putnastrparam(param, value)
        else:
            raise ValueError("Invalid MOSEK parameter '%s'." % param)

    @staticmethod
    def _handle_enum_param(task, param, value):
        if isinstance(param, mosek.dparam):
            task.putdouparam(param, value)
        elif isinstance(param, mosek.iparam):
            task.putintparam(param, value)
        elif isinstance(param, mosek.sparam):
            task.putstrparam(param, value)
        else:
            raise ValueError("Invalid MOSEK parameter '%s'." % param)
