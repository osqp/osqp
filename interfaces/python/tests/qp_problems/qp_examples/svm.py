#!/usr/bin/env python
from __future__ import absolute_import
import scipy.sparse as spspa
import numpy as np
from .qp_example import QPExample
from utils.qp_problem import QPProblem


class svm(QPExample):

    def name(self):
        return "svm"

    @staticmethod
    def run_tests(n_vec, m_vec, rho_vec, sigma_vec, alpha_vec, nm_num_prob, **options):
        prob = svm(n_vec, m_vec, rho_vec, sigma_vec,
                   alpha_vec, nm_num_prob, dens_lvl=0.4)
        prob.perform_tests(**options)
        return prob.df

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

    def gen_problem(self, m, n, dens_lvl=.8):
        """
        Support vector machine problem is defined as
                minimize	|| x ||^2 + gamma * 1.T * max(0, diag(b) A x + 1)

        Arguments
        ---------
        m, n        - Dimensions of matrix A        <int>
        osqp_opts   - Parameters of OSQP solver
        dens_lvl    - Density level of matrix A     <float>
        """

        # Generate data
        if m % 2 == 1:
            m = m + 1
        N = int(m / 2)
        gamma = 1.0
        b = np.append(np.ones(N), -np.ones(N))
        random_scaling = spspa.diags(np.power(10, 2 * np.random.randn(N)))
        A_upp = random_scaling.dot(spspa.random(N, n, density=dens_lvl))
        A_low = random_scaling.dot(spspa.random(N, n, density=dens_lvl))
        A = spspa.vstack([
                A_upp / np.sqrt(n) + (A_upp != 0.).astype(float) / n,
                A_low / np.sqrt(n) - (A_low != 0.).astype(float) / n]).tocsc()

        # Construct the problem
        #       minimize	 x.T * x + gamma 1.T * t
        #       subject to  t >= diag(b) A x + 1
        #                   t >= 0

        P = spspa.block_diag((2*spspa.eye(n), spspa.csc_matrix((m, m))),
                             format='csc')
        q = np.append(np.zeros(n), gamma*np.ones(m))
        A = spspa.vstack([
                spspa.hstack([spspa.diags(b).dot(A), -spspa.eye(m)]),
                spspa.hstack([spspa.csc_matrix((m, n)), spspa.eye(m)]),
            ]).tocsc()
        lA = np.append(-np.inf * np.ones(m), np.zeros(m))
        uA = np.append(-np.ones(m), np.inf * np.ones(m))

        # Create a quadprog_problem and store it in a private variable
        return QPProblem(P, q, A, lA, uA)

# # Generate and solve an SVM problem
# class svm(object):
#     """
#     Support vector machine problem is defined as
#             minimize	|| x ||^2 + gamma * 1.T * max(0, diag(b) A x + 1)
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
#         if m % 2 == 1:
#             m = m + 1
#         N = m / 2
#         gamma = 1.0
#         b = np.append(np.ones(N), -np.ones(N))
#         A_upp = spspa.random(N, n, density=dens_lvl)
#         A_low = spspa.random(N, n, density=dens_lvl)
#         A = spspa.vstack([
#                 A_upp / np.sqrt(n) + (A_upp != 0.).astype(float) / n,
#                 A_low / np.sqrt(n) - (A_low != 0.).astype(float) / n]).tocsc()
#
#         # Construct the problem
#         #       minimize	 x.T * x + gamma 1.T * t
#         #       subject to  t >= diag(b) A x + 1
#         #                   t >= 0
#
#         P = spspa.block_diag((2*spspa.eye(n), spspa.csc_matrix((m, m))),
#                              format='csc')
#         q = np.append(np.zeros(n), gamma*np.ones(m))
#         A = spspa.vstack([
#                 spspa.hstack([spspa.diags(b).dot(A), -spspa.eye(m)]),
#                 spspa.hstack([spspa.csc_matrix((m, n)), spspa.eye(m)]),
#             ]).tocsc()
#         lA = np.append(-np.inf * np.ones(m), np.zeros(m))
#         uA = np.append(-np.ones(m), np.inf * np.ones(m))
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
#
# # ==================================================================
# # ==    Solve a small example
# # ==================================================================
#
# # Set the random number generator
# sp.random.seed(1)
#
# # Set problem
# n = 30
# m = 2*n
#
# # Set options of the OSQP solver
# options = {'eps_abs':       1e-5,
#            'eps_rel':       1e-5,
#            'alpha':         1.6,
#            'scale_problem': True,
#            'scale_steps':   4,
#            'polish':        False}
#
# # Create an svm object
# svm_obj = svm(m, n, dens_lvl=0.3, osqp_opts=options)
#
# # Solve with different solvers
# resultsCPLEX = svm_obj.solve(solver=CPLEX)
# resultsGUROBI = svm_obj.solve(solver=GUROBI)
# resultsOSQP = svm_obj.solve(solver=OSQP)
#
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
