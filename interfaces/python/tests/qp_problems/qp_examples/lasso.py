#!/usr/bin/env python

from __future__ import absolute_import
import scipy.sparse as spspa
import scipy as sp
import numpy as np
from mathprogbasepy import QuadprogProblem
from .qp_example import QPExample

# from quadprog.solvers.solvers import GUROBI, CPLEX, OSQP
# import quadprog.solvers.osqp.osqp as osqp

# import matplotlib.pyplot as plt
# import seaborn


class lasso(QPExample):

    def name(self):
        return "lasso"

    @staticmethod
    def run_tests(n_vec, m_vec, rho_vec, sigma_vec, alpha_vec, nm_num_prob, **options):
        prob = lasso(n_vec, m_vec, rho_vec, sigma_vec,
                     alpha_vec, nm_num_prob, dens_lvl=0.4)
        prob.perform_tests(**options)
        return prob.df, prob.full_df

    def create_dims(self, n_vec, m_vec):
        """Reduce n_vec and m_vec choosing the dimensions that make the related
           problem feasible
        """
        dims_mat = np.array([], dtype=np.int64).reshape(2, 0)
        for n in n_vec:
            for m in m_vec[m_vec > n]:
                dims_mat = np.hstack((dims_mat,
                                      np.array([[n], [m]])))
        return dims_mat

    def gen_problem(self, m, n, dens_lvl=0.5, version='sparse'):
        """
        Lasso problem is defined as
                minimize	|| Ax - b ||^2 + gamma * || x ||_1

        Arguments
        ---------
        m, n        - Dimensions of matrix A        <int>
        osqp_opts   - Parameters of OSQP solver
        dens_lvl    - Density level of matrix A     <float>
        version     - QP reformulation              ['dense', 'sparse']
        """
        # Generate data
        Ad = spspa.random(m, n, density=dens_lvl)
        x_true = np.multiply((np.random.rand(n) > 0.5).astype(float),
                             np.random.randn(n)) / np.sqrt(n)
        bd = Ad.dot(x_true) + .5*np.random.randn(m)
        gamma = sp.rand()
        # gamma_max = 0.2 * np.linalg.norm(Ad.T.dot(bd), np.inf)
        # self._gammas = np.exp(np.linspace(np.log(gamma_max),
        #                       np.log(gamma_max * 1e-2), inst))
        # self._iter_cnt = 0

        # Construct the problem
        if version == 'dense':
            #       minimize	|| Ax - b ||^2 + gamma * np.ones(n).T * t
            #       subject to  -t <= x <= t
            P = spspa.block_diag((2*Ad.T.dot(Ad), spspa.csc_matrix((n, n))),
                                 format='csc')
            q = np.append(-2*Ad.T.dot(bd), gamma*np.ones(n))
            In = spspa.eye(n)
            A = spspa.vstack([spspa.hstack([In, -In]),
                              spspa.hstack([In, In])]).tocsc()
            lA = np.append(-np.inf * np.ones(n), np.zeros(n))
            uA = np.append(np.zeros(n), np.inf * np.ones(n))
        elif version == 'sparse':
            #       minimize	y.T * y + gamma * np.ones(n).T * t
            #       subject to  y = Ax
            #                   -t <= x <= t
            P = spspa.block_diag((spspa.csc_matrix((n, n)), 2*spspa.eye(m),
                                  spspa.csc_matrix((n, n))), format='csc')
            q = np.append(np.zeros(m + n), gamma*np.ones(n))
            In = spspa.eye(n)
            Onm = spspa.csc_matrix((n, m))
            A = spspa.vstack([spspa.hstack([Ad, -spspa.eye(m),
                                            spspa.csc_matrix((m, n))]),
                             spspa.hstack([In, Onm, -In]),
                             spspa.hstack([In, Onm, In])]).tocsc()
            lA = np.hstack([bd, -np.inf * np.ones(n), np.zeros(n)])
            uA = np.hstack([bd, np.zeros(n), np.inf * np.ones(n)])
        else:
            assert False, "Unhandled version"

        # Create a quadprog_problem and return in
        return QuadprogProblem(P, q, A, lA, uA)








#
#
#
#
# # Generate and solve a (sequence of) Lasso problem(s)
# class lasso(object):
#     """
#     Lasso problem is defined as
#             minimize	|| Ax - b ||^2 + gamma * || x ||_1
#
#     Arguments
#     ---------
#     m, n        - Dimensions of matrix A        <int>
#     osqp_opts   - Parameters of OSQP solver
#     dens_lvl    - Density level of matrix A     <float>
#     version     - QP reformulation              ['dense', 'sparse']
#     """
#
#     def __init__(self, m, n, dens_lvl=1.0, inst=1, version='dense',
#                  osqp_opts={}):
#         # Generate data
#         Ad = spspa.random(m, n, density=dens_lvl)
#         x_true = np.multiply((np.random.rand(n) > 0.5).astype(float),
#                              np.random.randn(n)) / np.sqrt(n)
#         bd = Ad.dot(x_true) + .5*np.random.randn(m)
#         gamma_max = 0.2 * np.linalg.norm(Ad.T.dot(bd), np.inf)
#         self._gammas = np.exp(np.linspace(np.log(gamma_max),
#                               np.log(gamma_max * 1e-2), inst))
#         self._iter_cnt = 0
#
#         # Construct the problem
#         if version == 'dense':
#             #       minimize	|| Ax - b ||^2 + gamma * np.ones(n).T * t
#             #       subject to  -t <= x <= t
#             P = spspa.block_diag((2*Ad.T.dot(Ad), spspa.csc_matrix((n, n))),
#                                  format='csc')
#             q = np.append(-2*Ad.T.dot(bd), self._gammas[0]*np.ones(n))
#             In = spspa.eye(n)
#             A = spspa.vstack([spspa.hstack([In, -In]),
#                               spspa.hstack([In, In])]).tocsc()
#             lA = np.append(-np.inf * np.ones(n), np.zeros(n))
#             uA = np.append(np.zeros(n), np.inf * np.ones(n))
#         elif version == 'sparse':
#             #       minimize	y.T * y + gamma * np.ones(n).T * t
#             #       subject to  y = Ax
#             #                   -t <= x <= t
#             P = spspa.block_diag((spspa.csc_matrix((n, n)), 2*spspa.eye(m),
#                                   spspa.csc_matrix((n, n))), format='csc')
#             q = np.append(np.zeros(m + n), self._gammas[0]*np.ones(n))
#             In = spspa.eye(n)
#             Onm = spspa.csc_matrix((n, m))
#             A = spspa.vstack([spspa.hstack([Ad, -spspa.eye(m),
#                                             spspa.csc_matrix((m, n))]),
#                              spspa.hstack([In, Onm, -In]),
#                              spspa.hstack([In, Onm, In])]).tocsc()
#             lA = np.hstack([bd, -np.inf * np.ones(n), np.zeros(n)])
#             uA = np.hstack([bd, np.zeros(n), np.inf * np.ones(n)])
#         else:
#             assert False, "Unhandled version"
#
#         # Create a quadprog_problem and store it in a private variable
#         self._prob = qp.quadprog_problem(P, q, A, lA, uA)
#         # Create an OSQP object and store it in a private variable
#         self._osqp = osqp.OSQP(**osqp_opts)
#         self._osqp.problem(P, q, A, lA, uA)
#
#     def solve(self, solver=OSQP):
#         """
#         Solve the problem with a specificed solver.
#         """
#         if solver == OSQP:
#             results = self._osqp.solve()
#         elif solver == CPLEX:
#             results = self._prob.solve(solver=CPLEX, verbose=0)
#         elif solver == GUROBI:
#             results = self._prob.solve(solver=GUROBI, OutputFlag=0)
#         else:
#             assert False, "Unhandled solver"
#         return results
#
#     def update_gamma(self, n):
#         """
#         Update parameter gamma in the problem and in
#         Return:
#         0   if the update is successful
#         1   if iter_cnt  > len(gammas)-1
#         """
#         self._iter_cnt = self._iter_cnt + 1
#         try:
#             gamma = self._gammas[self._iter_cnt]
#         except IndexError:
#             return 1
#         # Update current q in the problem
#         q = self._prob.q
#         q[-n:] = gamma*np.ones(n)
#         self._prob.q = q
#         # Update gmma in the OSQP object
#         self._osqp.set_problem_data(q=q)
#         # Update successfull
#         return 0
#
#
# # ==================================================================
# # ==    Solve a small example
# # ==================================================================
#
# # Set the random number generator
# sp.random.seed(1)
#
# # Set problem
# m = 20
# n = 10*m
# numofinst = 5
#
# # Set options of the OSQP solver
# options = {'eps_abs':       1e-4,
#            'eps_rel':       1e-4,
#            'alpha':         1.6,
#            'scale_problem': True,
#            'scale_steps':   4,
#            'polish':        False,
#            'warm_start':    True}
#
# # Create a lasso object
# lasso_obj = lasso(m, n, inst=numofinst, version='sparse', osqp_opts=options)
# for i in range(numofinst):
#     # Solve with different solvers
#     resultsCPLEX = lasso_obj.solve(solver=CPLEX)
#     resultsGUROBI = lasso_obj.solve(solver=GUROBI)
#     resultsOSQP = lasso_obj.solve(solver=OSQP)
#
#     # Print objective values
#     print "CPLEX  Objective Value: %.3f" % resultsCPLEX.objval
#     print "GUROBI Objective Value: %.3f" % resultsGUROBI.objval
#     print "OSQP   Objective Value: %.3f" % resultsOSQP.objval
#     print "\n"
#
#     # Print timings
#     print "CPLEX  CPU time: %.3f" % resultsCPLEX.cputime
#     print "GUROBI CPU time: %.3f" % resultsGUROBI.cputime
#     print "OSQP   CPU time: %.3f" % resultsOSQP.cputime
#     if numofinst > 1:
#         lasso_obj.update_gamma(n)
