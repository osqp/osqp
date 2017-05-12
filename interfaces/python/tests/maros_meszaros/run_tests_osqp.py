from __future__ import print_function

import numpy as np
import scipy.sparse as spa
import sys
import os


from utils.utils import load_maros_meszaros_problem
import mathprogbasepy as mpbpy

import osqp


def constrain_scaling(s, min_val, max_val):
    s = np.minimum(np.maximum(s, min_val), max_val)
    return s


# def main():
# Directory of problems
prob_dir = './mat'
lst_probs = os.listdir(prob_dir)

# Count number of problems
n_prob = len([name for name in lst_probs
             if os.path.isfile(prob_dir + "/" + name)])

# Problems index
p = 0

# Number unsolved problems
n_unsolved = 0

# Solve all Maroz Meszaros problems
for f in lst_probs:

    # if f[:-4] == 'CVXQP1_M':
    # if f[:-4] == 'AUG2DCQP':
    # if f[:-4] == 'BOYD1':
    # if f[:-4] == 'AUG2D':
    # if f[:-4] == 'AUG2DC':
    # if f[:-4] == 'CONT-101':
    # if f[:-4] == 'CONT-300':
    # if True:


        m = load_maros_meszaros_problem(prob_dir + "/" + f)  # Load problem


        print("%i) %s\t" % (p, f[:-4]), end='')

        # Solve problem
        # res = m.solve(solver=mpbpy.OSQP, verbose=True)  # No verbosity



        # Normalize cost (TODO: remove)
        norm_q = np.linalg.norm(m.q)
        # cost_scal = 1.
        cost_scal = constrain_scaling(norm_q, 1e-03, 1e03)
        if norm_q < 1e-06:   #  q is null!
            cost_scal = 1.

        # P = m.P / cost_scal
        # q = m.q / cost_scal

        cost_scal = 1


        q = m.q / cost_scal
        P = m.P / cost_scal
        A = m.A
        l = m.l
        u = m.u


        # # Normalize constraints (TODO: remove)
        # m_constraints = len(m.l)
        # E = np.zeros(m_constraints)
        # for i in range(m_constraints):
        #     abs_l = np.abs(m.l[i])
        #     if np.isinf(abs_l) or abs_l > 1e10 or abs_l < 1e-06:
        #         abs_l = 1.
        #     else:
        #         abs_l = constrain_scaling(abs_l,
        #                                   1e-03, 1e03)
        #
        #     abs_u = np.abs(m.u[i])
        #     if np.isinf(abs_u) or abs_u > 1e10 or abs_l < 1e-06:
        #         abs_u = 1.
        #     else:
        #         abs_u = constrain_scaling(abs_u,
        #                                   1e-03, 1e03)
        #
        #     # # Scale using maximum bound
        #     # max_abs_bnds = np.minimum(abs_l, abs_u)
        #     # E[i] = 1./max_abs_bnds
        #
        #     # Scale using both bounds
        #     # E[i] = 1. / (abs_l * abs_u)
        #
        #     # Exponentially scale bounds
        #     log_l = np.log(abs_l)
        #     log_u = np.log(abs_u)
        #     E[i] = np.exp((log_l + log_u)/2)
        #
        #     # import ipdb; ipdb.set_trace()
        #
        #
        # # Select scaling
        # # E = spa.diags(E)
        # E = spa.diags(np.ones(m_constraints))
        #
        # # import ipdb; ipdb.set_trace()
        #
        #
        # l = E.dot(m.l)
        # u = E.dot(m.u)
        # A = E.dot(m.A).tocsc()

        s = osqp.OSQP()
        s.setup(P, q, A, l, u,
                rho=0.1,
                auto_rho=True,
                verbose=False,
                scaling_iter=100,
                max_iter=10000,
                # early_terminate_interval=1,
                scaling=True)
        res = s.solve()

        # import ipdb; ipdb.set_trace()

        # Check if pure python implementation gives same results
        # import osqppurepy
        # s = osqppurepy.OSQP()
        # s.setup(P, q, A, l, u,
        #         rho=0.1,
        #         auto_rho=False,
        #         verbose=True,
        #         max_iter=2500,
        #         scaling=True)
        # res = s.solve()

        # for rho in np.logspace(-4, 2, 10):
        #     s = osqp.OSQP()
        #     s.setup(P, q, A, l, u,
        #             rho=rho,
        #             auto_rho=False,
        #             verbose=True,
        #             max_iter=2500,
        #             scaling=True)
        #     res = s.solve()

        p += 1

        if res.info.status_val == \
            s.constant('OSQP_MAX_ITER_REACHED'):
            n_unsolved += 1
            # import ipdb; ipdb.set_trace()

        print(res.info.status)


print('Number of solved problems %i/%i' % (n_prob - n_unsolved,
                                           n_prob))





# # Parsing optional command line arguments
# if __name__ == '__main__':
#     if len(sys.argv) == 1:
#         main()
#     else:
#         main(sys.argv[1:])
