# CPLEX interface to solve QP problems
import numpy as np
import quadprog.problem as qp
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

        # Solve QP problem with OSQP
        results = osqp.solve(p.Q, p.c, p.Aeq, p.beq,
                             p.Aineq, p.bineq, p.lb, p.ub, **self.options)

        return results
