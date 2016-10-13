# Interface to various QP solvers
import numpy as np
import scipy.sparse as spspa
import solvers.solvers as s

# Solver Constants
OPTIMAL = "optimal"
OPTIMAL_INACCURATE = "optimal_inaccurate"
INFEASIBLE = "infeasible"
INFEASIBLE_INACCURATE = "infeasible_inaccurate"
UNBOUNDED = "unbounded"
UNBOUNDED_INACCURATE = "unbounded_inaccurate"
SOLVER_ERROR = "solver_error"
# Statuses that indicate a solution was found.
SOLUTION_PRESENT = [OPTIMAL, OPTIMAL_INACCURATE]
# Statuses that indicate the problem is infeasible or unbounded.
INF_OR_UNB = [INFEASIBLE, INFEASIBLE_INACCURATE,
              UNBOUNDED, UNBOUNDED_INACCURATE]


class quadprogProblem(object):
    """
    Defines QP problem of the form
        minimize	1/2 x' Q x + c'x
        subject to	Aeq x == beq
                    Aineq x <= bineq
                    lb <= x <= ub

    Attributes
    ----------
    Q, c
    Aeq, beq
    Aineq, bineq
    lb, ub
    """

    def __init__(self, Q, c, Aeq, beq, Aineq, bineq, lb=None, ub=None):
        self.Q = Q
        self.c = c
        self.Aeq = Aeq
        self.beq = beq
        self.Aineq = Aineq
        self.bineq = bineq
        self.lb = lb if lb is not None else -np.inf*np.ones(c.size)
        self.ub = ub if ub is not None else np.inf*np.ones(c.size)

    def solve(self, solver=s.GUROBI, **kwargs):
        """
        Solve Quadratic Program with desired solver
        Supported solvers: CPLEX, GUROBI, OSQP
        """

        # Set solver
        if solver == s.GUROBI:
            from solvers.gurobi_qpif import GUROBI
            solver = GUROBI(**kwargs)  # Initialize solver
        elif solver == s.CPLEX:
                from solvers.cplex_qpif import CPLEX
                solver = CPLEX(**kwargs)  # Initialize solver
        elif solver == s.OSQP:
                from solvers.osqp_qpif import OSQP
                solver = OSQP(**kwargs)  # Initialize solver

        # Solve problem
        results = solver.solve(self)  # Solve problem


        return results
