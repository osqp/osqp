import numpy as np
import numpy.linalg as la
import errno
import os

import sys
from contextlib import contextmanager


@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    # assert that Python and C stdio write using the same file descriptor
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
            _redirect_stdout(to=old_stdout)


# Function to create directories
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        # Catch exception if directory created in between
        if exception.errno != errno.EEXIST:
            raise


def is_qp_solution_optimal(qp_problem, x, y,
                           eps_abs=1e-03, eps_rel=1e-03):
    '''
    Check optimality condition of the QP given the
    primal-dual solution (x, y) and the tolerance eps
    '''

    # Get problem matrices
    P = qp_problem['P']
    q = qp_problem['q']
    A = qp_problem['A']
    l = qp_problem['l']
    u = qp_problem['u']

    # Check primal feasibility
    Ax = A.dot(x)
    eps_pri = eps_abs + eps_rel * la.norm(Ax, np.inf)
    pri_res = np.minimum(Ax - l, 0) + np.maximum(Ax - u, 0)

    if la.norm(pri_res, np.inf) > eps_pri:
        print("Error in primal residual: %.4e > %.4e" %
              (la.norm(pri_res, np.inf), eps_pri))
        return False

    # Check dual feasibility
    Px = P.dot(x)
    Aty = A.T.dot(y)
    eps_dua = eps_abs + eps_rel * np.max([la.norm(Px, np.inf),
                                          la.norm(q, np.inf),
                                          la.norm(Aty, np.inf)])
    dua_res = Px + q + Aty

    if la.norm(dua_res, np.inf) > eps_dua:
        print("Error in dual residual: %.4e > %.4e" %
              (la.norm(dua_res, np.inf), eps_dua))
        return False

    # Check complementary slackness
    y_plus = np.maximum(y, 0)
    y_minus = np.minimum(y, 0)

    eps_comp = eps_abs + eps_rel * np.max([la.norm(Ax, np.inf)])

    comp_res_u = np.minimum(y_plus, np.abs(u - Ax))
    comp_res_l = np.minimum(-y_minus, np.abs(Ax - l))

    if la.norm(comp_res_l, np.inf) > eps_comp:
        print("Error in complementary slackness residual l: %.4e > %.4e" %
              (la.norm(comp_res_l, np.inf), eps_comp))
        return False

    if la.norm(comp_res_u, np.inf) > eps_comp:
        print("Error in complementary slackness residual u: %.4e > %.4e" %
              (la.norm(comp_res_u, np.inf), eps_comp))
        return False
    # If we arrived until here, the solution is optimal
    return True
