#!/usr/bin/env python
from __future__ import absolute_import
import scipy.sparse as spspa
import numpy as np
from mathprogbasepy import QuadprogProblem
from .qp_example import QPExample

# import utils.data_struct as ds
# from quadprog.solvers.solvers import GUROBI, CPLEX, OSQP
# import quadprog.solvers.osqp.osqp as osqp
#
# import matplotlib.pyplot as plt
# import seaborn


class huber_fit(QPExample):

    def name(self):
        return "huber_fit"

    @staticmethod
    def run_tests(n_vec, m_vec, rho_vec, sigma_vec, alpha_vec, nm_num_prob, **options):
        prob = huber_fit(n_vec, m_vec, rho_vec, sigma_vec,
                         alpha_vec, nm_num_prob, dens_lvl=0.4)
        prob.perform_tests(**options)
        return prob.df, prob.full_df

    def create_dims(self, n_vec, m_vec):
        """Reduce n_vec and m_vec choosing the dimensions that make the related
           problem feasible
        """
        dims_mat = np.array([], dtype=np.int64).reshape(2, 0)
        for m in m_vec[m_vec > 5]:
            for n in n_vec[n_vec > m/2]:
                dims_mat = np.hstack((dims_mat,
                                      np.array([[n - int(m/2)], [int(m/4)]])))
        return dims_mat

    def gen_problem(self, m, n, dens_lvl=1.0):
        """
        Huber fitting problem is defined as
                minimize	sum( huber(ai'x - bi) ),

        where huber() is the Huber penalty function defined as
                            | 1/2 x^2       |x| <= 1
                huber(x) = <
                            | |x| - 1/2     |x| > 1

        Arguments
        ---------
        m, n        - Dimensions of matrix A        <int>
        osqp_opts   - Parameters of OSQP solver
        dens_lvl    - Density level of matrix A     <float>
        """
        # Generate data
        A = spspa.random(m, n, density=dens_lvl, format='csc')
        x_true = np.random.randn(n) / np.sqrt(n)
        ind95 = (np.random.rand(m) < 0.95).astype(float)
        b = A.dot(x_true) + np.multiply(0.5*np.random.randn(m), ind95) \
                          + np.multiply(10.*np.random.rand(m), 1. - ind95)

        # Construct the problem
        #       minimize	1/2 u.T * u + np.ones(m).T * v
        #       subject to  -u - v <= Ax - b <= u + v
        #                   0 <= u <= 1
        #                   v >= 0
        Im = spspa.eye(m)
        P = spspa.block_diag((spspa.csc_matrix((n, n)), Im,
                              spspa.csc_matrix((m, m))), format='csc')
        q = np.append(np.zeros(m + n), np.ones(m))
        A = spspa.vstack([
                spspa.hstack([A, Im, Im]),
                spspa.hstack([A, -Im, -Im]),
                spspa.hstack([spspa.csc_matrix((m, n)), Im,
                              spspa.csc_matrix((m, m))]),
                spspa.hstack([spspa.csc_matrix((m, n + m)), Im])]).tocsc()
        lA = np.hstack([b, -np.inf*np.ones(m), np.zeros(2*m)])
        uA = np.hstack([np.inf*np.ones(m), b,
                       np.ones(m), np.inf*np.ones(m)])

        # Create a quadprog_problem and store it in a private variable
        return QuadprogProblem(P, q, A, lA, uA)


#
# # Generate and solve a Huber fitting problem
# class huber_fit(object):
#     """
#     Huber fitting problem is defined as
#             minimize	sum( huber(ai'x - bi) ),
#
#     where huber() is the Huber penalty function defined as
#                         | 1/2 x^2       |x| <= 1
#             huber(x) = <
#                         | |x| - 1/2     |x| > 1
#
#     Arguments
#     ---------
#     m, n        - Dimensions of matrix A        <int>
#     osqp_opts   - Parameters of OSQP solver
#     dens_lvl    - Density level of matrix A     <float>
#     """
#
#     def __init__(self, m, n, dens_lvl=1.0, osqp_opts={}):
#         # Generate data
#         A = spspa.random(m, n, density=dens_lvl, format='csc')
#         x_true = np.random.randn(n) / np.sqrt(n)
#         ind95 = (np.random.rand(m) < 0.95).astype(float)
#         b = A.dot(x_true) + np.multiply(0.5*np.random.randn(m), ind95) \
#                           + np.multiply(10.*np.random.rand(m), 1. - ind95)
#
#         # Construct the problem
#         #       minimize	1/2 u.T * u + np.ones(m).T * v
#         #       subject to  -u - v <= Ax - b <= u + v
#         #                   0 <= u <= 1
#         #                   v >= 0
#         Im = spspa.eye(m)
#         P = spspa.block_diag((spspa.csc_matrix((n, n)), Im,
#                               spspa.csc_matrix((m, m))), format='csc')
#         q = np.append(np.zeros(m + n), np.ones(m))
#         A = spspa.vstack([
#                 spspa.hstack([A, Im, Im]),
#                 spspa.hstack([A, -Im, -Im]),
#                 spspa.hstack([spspa.csc_matrix((m, n)), Im,
#                               spspa.csc_matrix((m, m))]),
#                 spspa.hstack([spspa.csc_matrix((m, n + m)), Im])]).tocsc()
#         lA = np.hstack([b, -np.inf*np.ones(m), np.zeros(2*m)])
#         uA = np.hstack([np.inf*np.ones(m), b, np.ones(m), np.inf*np.ones(m)])
#
#         # Create a quadprog_problem and store it in a private variable
#         self._prob = qp.quadprog_problem(P, q, A, lA, uA)
#         # Create an OSQP object and store it in a private variable
#         self._osqp = osqp.OSQP(**osqp_opts)
#         self._osqp.problem(P, q, A, lA, uA)

    # def solve(self, solver=OSQP):
    #     """
    #     Solve the problem with a specificed solver.
    #     """
    #     if solver == OSQP:
    #         results = self._osqp.solve()
    #     elif solver == CPLEX:
    #         results = self._prob.solve(solver=CPLEX, verbose=1)
    #     elif solver == GUROBI:
    #         results = self._prob.solve(solver=GUROBI, OutputFlag=0)
    #     else:
    #         assert False, "Unhandled solver"
    #     return results


# # ==================================================================
# # ==    Solve a small example
# # ==================================================================
#
# # Set the random number generator
# sp.random.seed(1)
#
# # Set problem
# n = 20
# m = 10*n
#
# # Set options of the OSQP solver
# options = {'eps_abs':       1e-4,
#            'eps_rel':       1e-4,
#            'alpha':         1.6,
#            'scale_problem': True,
#            'scale_steps':   4,
#            'polish':        False}
#
# # Create a lasso object
# huber_fit_obj = huber_fit(m, n, dens_lvl=0.50, osqp_opts=options)
#
# # Solve with different solvers
# resultsCPLEX = huber_fit_obj.solve(solver=CPLEX)
# resultsGUROBI = huber_fit_obj.solve(solver=GUROBI)
# resultsOSQP = huber_fit_obj.solve(solver=OSQP)
#
# # Print objective values
# print "CPLEX  Objective Value: %.3f" % resultsCPLEX.objval
# print "GUROBI Objective Value: %.3f" % resultsGUROBI.objval
# print "OSQP   Objective Value: %.3f" % resultsOSQP.objval
# print "\n"
#
# # Print timings
# print "CPLEX  CPU time: %.3f" % resultsCPLEX.cputime
# print "GUROBI CPU time: %.3f" % resultsGUROBI.cputime
# print "OSQP   CPU time: %.3f" % resultsOSQP.cputime
#
# # ipdb.set_trace()
#
# # Recover A, x and b from the problem
# A = huber_fit_obj._osqp.problem.A[:m, :n]
# x = resultsOSQP.x[:n]
# Ax = A.dot(x)
# b = huber_fit_obj._osqp.problem.lA[:m]
#
# # Plot Ax against b
# plt.rc('font', family='serif')
# plt.figure(figsize=(10, 10))
# plt.plot(Ax, b, 'b.')
# plt.plot([np.min(Ax), np.max(Ax)], [np.min(Ax), np.max(Ax)], 'r')
# plt.xlabel('$a_i^T x$', fontsize=18)
# plt.ylabel('$b_i$', fontsize=18)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# axes = plt.gca()
# dAx = np.max(Ax) - np.min(Ax)
# db = np.max(b) - np.min(b)
# axes.set_xlim([np.min(Ax)-0.1*(dAx), np.max(Ax)+0.1*(dAx)])
# axes.set_ylim([np.min(b)-0.1*(db), np.max(b)+0.1*(db)])
# plt.show()
