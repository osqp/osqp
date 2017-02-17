#!/usr/bin/env python

from __future__ import absolute_import
import scipy.sparse as spspa
import numpy as np
from mathprogbasepy import QuadprogProblem
from .qp_example import QPExample


class nonneg_l2(QPExample):

    def name(self):
        return "nonneg_l2"

    @staticmethod
    def run_tests(n_vec, m_vec, rho_vec, sigma_vec, alpha_vec, **options):
        prob = nonneg_l2(n_vec, m_vec, rho_vec, sigma_vec,
                         alpha_vec, dens_lvl=0.4)
        prob.perform_tests(**options)
        return prob.df

    def create_dims(self, n_vec, m_vec):
        """Reduce n_vec and m_vec choosing the dimensions that make the related
           problem feasible
        """
        self.dims_mat = np.array([], dtype=np.int64).reshape(2, 0)
        for n in n_vec:
            for m in m_vec[m_vec > n]:
                self.dims_mat = np.hstack((self.dims_mat,
                                          np.array([[n], [m]])))

    def gen_problem(self, m, n, dens_lvl=0.5, version='dense'):
        """
        Nonnegative least-squares problem is defined as
                minimize	|| Ax - b ||^2
                subjec to   x >= 0

        Arguments
        ---------
        m, n        - Dimensions of matrix A        <int>
        osqp_opts   - Parameters of OSQP solver
        dens_lvl    - Density level of matrix A     <float>
        version     - QP reformulation              ['dense', 'sparse']
        """
        # Generate data
        Ad = spspa.random(m, n, density=dens_lvl, format='csc')
        x_true = np.ones(n) / n + np.random.randn(n) / np.sqrt(n)
        bd = Ad.dot(x_true) + 0.5*np.random.randn(m)

        # Construct the problem
        if version == 'dense':
            #       minimize	1/2 x.T (A.T * A) x - (A.T * b).T * x
            #       subject to  x >= 0
            P = Ad.T.dot(Ad)
            q = -Ad.T.dot(bd)
            A = spspa.eye(n)
            lA = np.zeros(n)
            uA = np.inf * np.ones(n)
        elif version == 'sparse':
            #       minimize	1/2 y.T*y
            #       subject to  y = Ax - b
            #                   x >= 0
            Im = spspa.eye(m)
            P = spspa.block_diag((spspa.csc_matrix((n, n)), Im), format='csc')
            q = np.zeros(n + m)
            A = spspa.vstack([
                    spspa.hstack([Ad, -Im]),
                    spspa.hstack([spspa.eye(n), spspa.csc_matrix((n, m))]),
                ]).tocsc()
            lA = np.append(bd, np.zeros(n))
            uA = np.append(bd, np.inf * np.ones(n))
        else:
            assert False, "Unhandled version"

        # Create a quadprog_problem and return it
        return QuadprogProblem(P, q, A, lA, uA)


# # Generate and solve a nonnegative least-squares problem
# class nonneg_l2(object):
#     """
#     Nonnegative least-squares problem is defined as
#             minimize	|| Ax - b ||^2
#             subjec to   x >= 0
#
#     Arguments
#     ---------
#     m, n        - Dimensions of matrix A        <int>
#     osqp_opts   - Parameters of OSQP solver
#     dens_lvl    - Density level of matrix A     <float>
#     version     - QP reformulation              ['dense', 'sparse']
#     """
#
#     def __init__(self, m, n, dens_lvl=1.0, version='dense',
#                  osqp_opts={}):
#         # Generate data
#         Ad = spspa.random(m, n, density=dens_lvl, format='csc')
#         x_true = np.ones(n) / n + np.random.randn(n) / np.sqrt(n)
#         bd = Ad.dot(x_true) + 0.5*np.random.randn(m)
#
#         # Construct the problem
#         if version == 'dense':
#             #       minimize	1/2 x.T (A.T * A) x - (A.T * b).T * x
#             #       subject to  x >= 0
#             P = Ad.T.dot(Ad)
#             q = -Ad.T.dot(bd)
#             A = spspa.eye(n)
#             lA = np.zeros(n)
#             uA = np.inf * np.ones(n)
#         elif version == 'sparse':
#             #       minimize	1/2 y.T*y
#             #       subject to  y = Ax - b
#             #                   x >= 0
#             Im = spspa.eye(m)
#             P = spspa.block_diag((spspa.csc_matrix((n, n)), Im), format='csc')
#             q = np.zeros(n + m)
#             A = spspa.vstack([
#                     spspa.hstack([Ad, -Im]),
#                     spspa.hstack([spspa.eye(n), spspa.csc_matrix((n, m))]),
#                 ]).tocsc()
#             lA = np.append(bd, np.zeros(n))
#             uA = np.append(bd, np.inf * np.ones(n))
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
#             results = self._prob.solve(solver=CPLEX, verbose=1)
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
# m = 3
# n = 2*m
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
# nonneg_l2_obj = nonneg_l2(m, n, dens_lvl=0.5, version='sparse',
#                           osqp_opts=options)
#
# # Solve with different solvers
# resultsCPLEX = nonneg_l2_obj.solve(solver=CPLEX)
# resultsGUROBI = nonneg_l2_obj.solve(solver=GUROBI)
# resultsOSQP = nonneg_l2_obj.solve(solver=OSQP)
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
