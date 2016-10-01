# Quadprog results class
class quadprogResults(object):

    """
    Stores results of a QP solver
    """

    def __init__(self, status, objval, x, sol_dual_eq,
                 sol_dual_ineq, sol_dual_lb, sol_dual_ub, cputime):
        self.status = status
        self.objval = objval
        self.x = x
        self.sol_dual_eq = sol_dual_eq
        self.sol_dual_ineq = sol_dual_ineq
        self.sol_dual_lb = sol_dual_lb
        self.sol_dual_ub = sol_dual_ub
        self.cputime = cputime
