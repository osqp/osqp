#!/usr/bin/env python

from __future__ import absolute_import
import scipy.sparse as spspa
import numpy as np
from mathprogbasepy import QuadprogProblem
from .qp_example import QPExample


class portfolio(QPExample):

    def name(self):
        return "portfolio"

    @staticmethod
    def run_tests(n_vec, m_vec, rho_vec, sigma_vec, alpha_vec, **options):
        prob = portfolio(n_vec, m_vec, rho_vec, sigma_vec,
                         alpha_vec, dens_lvl=0.4)
        prob.perform_tests(**options)
        return prob.df

    def create_dims(self, n_vec, m_vec):
        """Reduce n_vec and m_vec choosing the dimensions that make the related
           problem feasible
        """
        self.dims_mat = np.array([], dtype=np.int64).reshape(2, 0)
        for n in n_vec:
            for m in m_vec[m_vec < 0.1*n]:
                self.dims_mat = np.hstack((self.dims_mat,
                                          np.array([[n], [m]])))

    def gen_problem(self, n, k, dens_lvl=0.5, version='dense'):
        """
        Portfolio optimization problem is defined as
                maximize	mu.T * x - gamma x.T (F * F.T + D) x
                subjec to   1.T x = 1
                            x >= 0

        Arguments
        ---------
        k, n        - Dimensions of matrix F        <int>
        osqp_opts   - Parameters of OSQP solver
        dens_lvl    - Density level of matrix A     <float>
        version     - QP reformulation              ['dense', 'sparse']
        """
        # Generate data
        F = spspa.random(n, k, density=dens_lvl, format='csc')
        D = spspa.diags(np.random.rand(n) * np.sqrt(k), format='csc')
        mu = np.random.randn(n)
        gamma = 1

        # Construct the problem
        if version == 'dense':
            #       minimize	x.T (F * F.T + D) x - mu.T / gamma * x
            #       subject to  1.T x = 1
            #                   0 <= x <= 1
            P = 2 * (F.dot(F.T) + D)
            q = -mu / gamma
            A = spspa.vstack([np.ones((1, n)), spspa.eye(n)]).tocsc()
            lA = np.append([1.], np.zeros(n))
            uA = np.append([1.], np.ones(n))

        elif version == 'sparse':
            #       minimize	x.T*D*x + y.T*y - mu.T / gamma * x
            #       subject to  1.T x = 1
            #                   F.T x = y
            #                   0 <= x <= 1
            P = spspa.block_diag((2*D, 2*spspa.eye(k)), format='csc')
            q = np.append(-mu / gamma, np.zeros(k))
            A = spspa.vstack([
                    spspa.hstack([spspa.csc_matrix(np.ones((1, n))),
                                  spspa.csc_matrix((1, k))]),
                    spspa.hstack([F.T, -spspa.eye(k)]),
                    spspa.hstack([spspa.eye(n), spspa.csc_matrix((n, k))])
                ]).tocsc()
            lA = np.hstack([1., np.zeros(k), np.zeros(n)])
            uA = np.hstack([1., np.zeros(k), np.ones(n)])

        else:
            assert False, "Unhandled version"
        # Create a quadprog_problem and return it
        return QuadprogProblem(P, q, A, lA, uA)

# # Generate and solve a Portfolio optimization problem
# class portfolio(object):
#     """
#     Portfolio optimization problem is defined as
#             maximize	mu.T * x - gamma x.T (F * F.T + D) x
#             subjec to   1.T x = 1
#                         x >= 0
#
#     Arguments
#     ---------
#     k, n        - Dimensions of matrix F        <int>
#     osqp_opts   - Parameters of OSQP solver
#     dens_lvl    - Density level of matrix A     <float>
#     version     - QP reformulation              ['dense', 'sparse']
#     """
#
#     def __init__(self, k, n, dens_lvl=1.0, version='dense',
#                  osqp_opts={}):
#         # Generate data
#         F = spspa.random(n, k, density=dens_lvl, format='csc')
#         D = spspa.diags(np.random.rand(n) * np.sqrt(k), format='csc')
#         mu = np.random.randn(n)
#         gamma = 1
#
#         # Construct the problem
#         if version == 'dense':
#             #       minimize	x.T (F * F.T + D) x - mu.T / gamma * x
#             #       subject to  1.T x = 1
#             #                   0 <= x <= 1
#             P = 2 * (F.dot(F.T) + D)
#             q = -mu / gamma
#             A = spspa.vstack([np.ones((1, n)), spspa.eye(n)]).tocsc()
#             lA = np.append([1.], np.zeros(n))
#             uA = np.append([1.], np.ones(n))
#
#         elif version == 'sparse':
#             #       minimize	x.T*D*x + y.T*y - mu.T / gamma * x
#             #       subject to  1.T x = 1
#             #                   F.T x = y
#             #                   0 <= x <= 1
#             P = spspa.block_diag((2*D, 2*spspa.eye(k)), format='csc')
#             q = np.append(-mu / gamma, np.zeros(k))
#             A = spspa.vstack([
#                     spspa.hstack([spspa.csc_matrix(np.ones((1, n))),
#                                   spspa.csc_matrix((1, k))]),
#                     spspa.hstack([F.T, -spspa.eye(k)]),
#                     spspa.hstack([spspa.eye(n), spspa.csc_matrix((n, k))])
#                 ]).tocsc()
#             lA = np.hstack([1., np.zeros(k), np.zeros(n)])
#             uA = np.hstack([1., np.zeros(k), np.ones(n)])
#
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
#
# # ==================================================================
# # ==    Solve a small example
# # ==================================================================
#
# # Set the random number generator
# sp.random.seed(1)
#
# # Set problem
# k = 20
# n = 10*k
#
# # Set options of the OSQP solver
# options = {'eps_abs':       1e-5,
#            'eps_rel':       1e-5,
#            'alpha':         1.6,
#            'scale_problem': True,
#            'scale_steps':   20,
#            'polish':        False}
#
# # Create a lasso object
# portfolio_obj = portfolio(k, n, dens_lvl=0.3, version='sparse',
#                           osqp_opts=options)
#
# # Solve with different solvers
# resultsCPLEX = portfolio_obj.solve(solver=CPLEX)
# resultsGUROBI = portfolio_obj.solve(solver=GUROBI)
# resultsOSQP = portfolio_obj.solve(solver=OSQP)
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
