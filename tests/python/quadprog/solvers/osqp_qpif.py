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
        p.A = p.A.tocsr()
        p.P = p.P.tocsr()

        # Define OSQP object
        model = osqp.OSQP(**self.options)

        # Define QP problem
        model.problem(p.P, p.q, p.A, p.lA, p.uA)

        # Solve QP with OSQP
        res = model.solve()

        # Return quadprogResults
        return res
