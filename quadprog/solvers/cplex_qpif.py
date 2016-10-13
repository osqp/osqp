# CPLEX interface to solve QP problems
import numpy as np
import quadprog.problem as qp
from quadprog.results import quadprogResults
import cplex as cpx
import ipdb


class CPLEX(object):
    """
    An interface for the CPLEX QP solver.
    """

    # Map of CPLEX status to CVXPY status. #TODO: add more!
    STATUS_MAP = {1: qp.OPTIMAL,
                  3: qp.INFEASIBLE,
                  2: qp.UNBOUNDED,
                  6: qp.OPTIMAL_INACCURATE}

    def __init__(self, **kwargs):
        self.options = kwargs

    def solve(self, p):

        # Convert Matrices in CSR format
        p.Aeq = p.Aeq.tocsr()
        p.Aineq = p.Aineq.tocsr()
        p.Q = p.Q.tocsr()

        # Get problem dimensions
        nx = p.Q.shape[0]
        neq = p.Aeq.shape[0]
        nineq = p.Aineq.shape[0]

        # Adjust infinity values in bounds
        for i in range(nx):
            if p.ub[i] == -np.inf:
                p.ub[i] = -cpx.infinity
            if p.ub[i] == np.inf:
                p.ub[i] = cpx.infinity
            if p.lb[i] == -np.inf:
                p.lb[i] = -cpx.infinity
            if p.lb[i] == np.inf:
                p.lb[i] = cpx.infinity

        # Adjust infinity values in rhs
        for i in range(neq):
            if p.beq[i] == np.inf:
                p.beq[i] = cpx.infinity
            elif p.beq[i] == -np.inf:
                p.beq[i] = -cpx.infinity

        for i in range(nineq):
            if p.bineq[i] == np.inf:
                p.bineq[i] = cpx.infinity
            elif p.bineq[i] == -np.inf:
                p.bineq[i] = -cpx.infinity

        # Define CPLEX problem
        m = cpx.Cplex()

        # Minimize problem
        m.objective.set_sense(m.objective.sense.minimize)

        # Add variables
        m.variables.add(obj=p.c,           # Linear objective part
                        ub=p.ub, lb=p.lb)    # Lower and upper bounds

        # Add constraints
        sense = ["E"]*neq + ["L"]*nineq  # Constraints sense: == and <=
        rows = []
        for i in range(neq):  # Add equalities
            start = p.Aeq.indptr[i]
            end = p.Aeq.indptr[i+1]
            rows.append([p.Aeq.indices[start:end].tolist(),
                         p.Aeq.data[start:end].tolist()])
        for i in range(nineq):  # Add inequalities
            start = p.Aineq.indptr[i]
            end = p.Aineq.indptr[i+1]
            rows.append([p.Aineq.indices[start:end].tolist(),
                         p.Aineq.data[start:end].tolist()])
        m.linear_constraints.add(lin_expr=rows,
                                 senses=sense,
                                 rhs=np.hstack([p.beq, p.bineq]).tolist())

        # Set quadratic Cost
        qmat = []
        for i in range(nx):
            start = p.Q.indptr[i]
            end = p.Q.indptr[i+1]
            qmat.append([p.Q.indices[start:end].tolist(),
                        p.Q.data[start:end].tolist()])
        m.objective.set_quadratic(qmat)

        # Set parameters
        for param, value in self.options.iteritems():
            if param == "verbose":
                if value == 0:
                    m.set_results_stream(None)
                    m.set_log_stream(None)
                    m.set_error_stream(None)
                    m.set_warning_stream(None)
            else:
                exec("m.parameters.%s.set(%d)" % (param, value))

        # Solve problem
        start = m.get_time()
        m.solve()
        end = m.get_time()

        # Return results

        # Get status
        status = self.STATUS_MAP.get(m.solution.get_status(), qp.SOLVER_ERROR)

        # Get computation time
        cputime = end-start

        if (status != qp.SOLVER_ERROR) & (status != qp.INFEASIBLE):
            # Get objective value
            objval = m.solution.get_objective_value()

            # Get solution
            sol = np.array(m.solution.get_values())

            # Get dual values
            duals = m.solution.get_dual_values()
            sol_dual_eq = -np.array(duals[:neq])    # Cplex uses swapped signs (-1)
            sol_dual_ineq = -np.array(duals[neq:])  # Cplex uses swapped signs (-1)

            # Bounds
            sol_dual_ub = np.zeros(nx)
            sol_dual_lb = np.zeros(nx)

            RCx = m.solution.get_reduced_costs()  # Get reduced costs
            for i in range(nx):
                if RCx[i] >= 1e-07:
                    sol_dual_lb[i] = RCx[i]
                else:
                    sol_dual_ub[i] = -RCx[i]

            # Get computation time
            cputime = end-start

            # Get total number of iterations
            total_iter = int(m.solution.progress.get_num_barrier_iterations())

            return quadprogResults(status, objval, sol, sol_dual_eq,
                                   sol_dual_ineq, sol_dual_lb, sol_dual_ub,
                                   cputime, total_iter)
        else:
            return quadprogResults(status, None, None, None,
                                   None, None, None,
                                   cputime, None)
