# Quadprog results class
class quadprogResults(object):

    """
    Stores results of a QP solver
    """

    def __init__(self, status, objval, x, dual, cputime, total_iter):
        self.status = status
        self.objval = objval
        self.x = x
        self.dual = dual
        self.cputime = cputime
        self.total_iter = total_iter
