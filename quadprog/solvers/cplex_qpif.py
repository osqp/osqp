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
        p.A = p.A.tocsr()
        p.P = p.P.tocsr()

        # Get problem dimensions
        n = p.P.shape[0]
        m = p.A.shape[0]

        # Adjust infinity values in bounds
        for i in range(m):
            if p.uA[i] == -np.inf:
                p.uA[i] = -cpx.infinity
            if p.uA[i] == np.inf:
                p.uA[i] = cpx.infinity
            if p.lA[i] == -np.inf:
                p.lA[i] = -cpx.infinity
            if p.lA[i] == np.inf:
                p.lA[i] = cpx.infinity

        # Define CPLEX problem
        model = cpx.Cplex()

        # Minimize problem
        model.objective.set_sense(model.objective.sense.minimize)

        # Add variables
        model.variables.add(obj=p.q,
                            lb=-cpx.infinity*np.ones(n),
                            ub=cpx.infinity*np.ones(n))  # Linear objective part

        # Add constraints
        sense = ["R"] * m  # Constraints sense: (range between lA and uA)
        rows = []
        for i in range(m):  # Add inequalities
            start = p.A.indptr[i]
            end = p.A.indptr[i+1]
            rows.append([p.A.indices[start:end].tolist(),
                         p.A.data[start:end].tolist()])
        model.linear_constraints.add(lin_expr=rows,
                                     senses=sense,
                                     range_values=(p.lA - p.uA).tolist(),
                                     rhs=p.uA.tolist())

        # Set quadratic Cost
        if p.P.count_nonzero():  # Only if quadratic form is not null
            qmat = []
            for i in range(n):
                start = p.P.indptr[i]
                end = p.P.indptr[i+1]
                qmat.append([p.P.indices[start:end].tolist(),
                            p.P.data[start:end].tolist()])
            model.objective.set_quadratic(qmat)

        # Set parameters
        for param, value in self.options.iteritems():
            if param == "verbose":
                if value == 0:
                    model.set_results_stream(None)
                    model.set_log_stream(None)
                    model.set_error_stream(None)
                    model.set_warning_stream(None)
            else:
                exec("model.parameters.%s.set(%d)" % (param, value))

        # Solve problem
        try:
            start = model.get_time()
            model.solve()
            end = model.get_time()
        except:  # Error in the solution
            print "Error in CPLEX solution\n"
            return quadprogResults(qp.SOLVER_ERROR, None, None, None,
                                   np.inf, None)

        # Return results

        # Get status
        status = self.STATUS_MAP.get(model.solution.get_status(),
                                     qp.SOLVER_ERROR)

        # Get computation time
        cputime = end-start

        if (status != qp.SOLVER_ERROR) & (status != qp.INFEASIBLE):
            # Get objective value
            objval = model.solution.get_objective_value()

            # Get solution
            sol = np.array(model.solution.get_values())

            # Get dual values
            dual = -np.array(model.solution.get_dual_values())
            # dual_eq = -np.array(duals[:neq])    # Cplex uses swapped signs (-1)
            # dual_ineq = -np.array(duals[neq:])  # Cplex uses swapped signs (-1)

            # Bounds
            # dual_ub = np.zeros(n)
            # dual_lb = np.zeros(n)

            # RCx = m.solution.get_reduced_costs()  # Get reduced costs
            # for i in range(n):
            #     if RCx[i] >= 1e-07:
            #         dual_lb[i] = RCx[i]
            #     else:
            #         dual_ub[i] = -RCx[i]

            # Get computation time
            cputime = end-start

            # Get total number of iterations
            total_iter = int(model.solution.progress.get_num_barrier_iterations())

            return quadprogResults(status, objval, sol, dual,
                                   cputime, total_iter)
        else:
            return quadprogResults(status, None, None, None,
                                   cputime, None)
