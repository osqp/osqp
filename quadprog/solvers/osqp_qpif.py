# OSQP interface to solve QP problems
import osqp.osqp as osqp


class OSQP(object):
    """
    An interface for the OSQP QP solver.
    """

    def __init__(self, **kwargs):
        self.options = kwargs

    def solve(self, p):

        # Convert Matrices in CSR format
        p.Aeq = p.Aeq.tocsr()
        p.Aineq = p.Aineq.tocsr()
        p.Q = p.Q.tocsr()

        # Define OSQP object
        m = osqp.OSQP(**self.options)

        # Define QP problem
        m.problem(p.Q, p.c, p.Aeq, p.beq, p.Aineq, p.bineq, p.lb, p.ub)

        # Solve QP with OSQP
        res = m.solve()

        # Return quadprogResults
        return res
