#!/usr/bin/env python

from __future__ import absolute_import
import scipy.sparse as spspa
import numpy as np
from .qp_example import QPExample
from utils.qp_problem import QPProblem


class lp(QPExample):

    def name(self):
        return "lp"

    @staticmethod
    def run_tests(n_vec, m_vec, rho_vec, sigma_vec, alpha_vec, nm_num_prob, **options):
        prob = lp(n_vec, m_vec, rho_vec, sigma_vec,
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
    def gen_problem(self, m, n, dens_lvl=0.5):
        """
        Linear program in the inequality from is defined as
                minimize	c.T * x
                subjec to   Ax <= b

        Arguments
        ---------
        m, n        - Dimensions of matrix A        <int>
        osqp_opts   - Parameters of OSQP solver
        dens_lvl    - Density level of matrix A     <float>
        """

        # Generate data
        x_true = np.random.randn(n) / np.sqrt(n)
        random_scaling = spspa.diags(np.power(10, 2 * np.random.randn(m)))
        A = random_scaling.dot(spspa.random(m, n, density=dens_lvl,
                               format='csc'))
        uA = A.dot(x_true) + 0.1*np.random.rand(m)
        lA = -np.inf * np.ones(m)
        q = -A.T.dot(np.random.rand(m))
        P = spspa.csc_matrix((n, n))

        # Create a quadprog_problem and return it
        return QPProblem(P, q, A, lA, uA)
#
#
# # Generate and solve a linear program
# class lp(object):
#     """
#     Linear program in the inequality from is defined as
#             minimize	c.T * x
#             subjec to   Ax <= b
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
#         x_true = np.random.randn(n) / np.sqrt(n)
#         A = spspa.random(m, n, density=dens_lvl, format='csc')
#         uA = A.dot(x_true) + 0.1*np.random.rand(m)
#         lA = -np.inf * np.ones(m)
#         q = -A.T.dot(np.random.rand(m))
#         P = spspa.csc_matrix((n, n))
#
#         # Create a quadprog_problem and store it in a private variable
#         self._prob = qp.quadprog_problem(P, q, A, lA, uA)
#         # Create an OSQP object and store it in a private variable
#         self._osqp = osqp.OSQP(**osqp_opts)
#         self._osqp.problem(P, q, A, lA, uA)

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
# n = 20
# m = 100
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
# lp_obj = lp(m, n, dens_lvl=0.3, osqp_opts=options)
#
# # Solve with different solvers
# resultsCPLEX = lp_obj.solve(solver=CPLEX)
# resultsGUROBI = lp_obj.solve(solver=GUROBI)
# resultsOSQP = lp_obj.solve(solver=OSQP)
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
