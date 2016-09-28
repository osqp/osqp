#!/usr/bin/env python

import sys
import numpy as np
from numpy import linalg as npla
from scipy import linalg as spla
import cvxpy as cvx
from cvxopt import cholmod

import scipy as sp
import matplotlib.pyplot as plt
import ipdb		# ipdb.set_trace()

# Solver Constants
OPTIMAL = "optimal"
UNSOLVED = "optimal_inaccurate"
INFEASIBLE = "infeasible"
UNBOUNDED = "unbounded"



class quadProgSolution:
	"""
	stores QP solution
	"""
	def __init__(self, status, objval, sol_prim, sol_dual_eq, sol_dual_ineq, sol_dual_lb, sol_dual_ub):
		self.status = status
		self.objval = objval
		self.sol_prim = sol_prim
		self.sol_dual_eq = sol_dual_eq
		self.sol_dual_ineq = sol_dual_ineq
		self.sol_dual_lb = sol_dual_lb
		self.sol_dual_ub = sol_dual_ub


def project(xbar, lb, ub):
	nx = np.size(lb)
	
	# Round x part to [l, u] interval
	xbar[:nx] = np.minimum(np.maximum(xbar[:nx], lb), ub)
	
	# Round slack variables to positive ortant
	xbar[nx:] = np.maximum(xbar[nx:], 0)
	
	return xbar


# Base QP Solver
def OSQP(Q, c, Aeq, beq, Aineq, bineq, lb, ub, x0=0, max_iter=500, rho=1.6, printiter=25, scaling=False):
	"""
	Operator splitting solver for a QP problem given
	in the following form
		minimize	1/2 x' Q x + c'x
		subject to	Aeq x == beq
					Aineq x <= bineq
					lb <= x <= ub
	"""
	
	# Get dimensions
	nx = c.shape[0]
	neq = beq.shape[0]
	nineq = bineq.shape[0]
	nvar = nx + nineq  # Number of variables in standard form: x and s variables
	
	# Form compact (c) matrices for standard QP from
	Qc = spla.block_diag(Q, np.zeros((nineq, nineq)))
	cc = np.append(c, np.zeros(nineq))
	Ac = np.vstack([np.hstack([Aeq, np.zeros((neq, nineq))]), np.hstack([Aineq, np.eye(nineq)])])
	bc = np.append(beq, bineq)
	
	# Scaling (s): Normalize rows of Ac
	if scaling:
		scaler = np.sqrt(np.square(Ac).sum(1))	# norm of each row of Ac
		As = Ac / scaler[:,None]
		bs = bc / scaler
	else:
		As = Ac
		bs = bc
	
	# Factorize KKT matrix using LU decomposition (for now)
	KKT = np.vstack([np.hstack([Qc + rho*np.eye(nvar), As.T]), np.hstack([As, -1./rho*np.eye(nineq + neq)])])
	LU, piv = spla.lu_factor(KKT)
	
	print "Splitting QP Solver"
	print "-------------------\n"
	print "iter |\t   cost\n"
	
	# Run ADMM
	z = np.zeros(nvar)
	u = np.zeros(neq + nineq + nvar)  # Number of dual variables in ADMM
	
	for i in range(max_iter):
	
		# Update RHS of KKT system
		qtemp = -cc + rho*(z + np.dot(As.T, bs) - np.dot(As.T, u[:neq + nineq]) - u[neq + nineq:])
		qbar = np.append(qtemp, np.zeros(nineq+neq))
		
		# x update
		x = spla.lu_solve((LU, piv), qbar)[:nvar]	# Select only first nvar elements
		# z update
		z = project(x + u[neq+nineq:], lb, ub)
		# u update
		u = u + np.append(np.dot(As, x), x) - np.append(np.zeros(neq + nineq), z) - np.append(bs, np.zeros(nvar))
		
		#todo: Stopping criterion
		
		# Print cost function every printiter iterations
		if (i+1 == 1) | (np.mod(i+1, printiter) == 0):
			xtemp = z[:nx]
			f = .5*np.dot(np.dot(xtemp.T, Q), xtemp) + np.dot(c.T, xtemp)
			print "%4s | \t%7.2f" % (i+1, f)
	print "Optimization Done\n"
	
	# Rescale (r) dual variables
	if scaling:
		dual_vars = rho*u[:neq+nineq] / scaler
	else:
		dual_vars = rho*u[:neq+nineq]
	
	
	#todo: What is a status of the obtained solution?
	
	#todo: Solution polishing
	
	# Retrieve solution
	sol_prim = z[:nx]
	objval = .5*np.dot(np.dot(sol_prim, Q), sol_prim) + np.dot(c, sol_prim)
	
	# Retrieve dual variables
	sol_dual_eq = dual_vars[:neq]
	sol_dual_ineq = dual_vars[neq:]	
	stat_cond = np.dot(Q, sol_prim) + c + np.dot(Aeq.T, sol_dual_eq) + np.dot(Aineq.T, sol_dual_ineq)
	sol_dual_lb = -np.minimum(stat_cond, 0)
	sol_dual_ub = np.maximum(stat_cond, 0)
	
	# Return solution as a quadProgSolution object
	return quadProgSolution(OPTIMAL, objval, sol_prim, sol_dual_eq, sol_dual_ineq, sol_dual_lb, sol_dual_ub)




def isPSD(A, tol=1e-8):
	eigs, _ = sp.linalg.eigh(A)
	return np.all(eigs > -tol)



def main():
	
	# Select an example QP
	expl = 'random' # {'small', 'random'}
	if expl == 'random':
		nx = 50
		neq = 10
		nineq = 20
		# Generate random Matrices
		Qt = sp.randn(nx, nx)
		Q = np.dot(Qt.T, Qt)
		c = sp.randn(nx)
		Aeq = sp.randn(neq, nx)
		beq = sp.randn(neq)
		Aineq = sp.randn(nineq, nx)
		bineq = 100*sp.rand(nineq)
		lb =-3.*np.ones(nx)
		ub = 5.*np.ones(nx)
	elif expl == 'small':
		nx = 2
		neq = 1
		nineq = 0
		# Generate a small example
		Q = np.array([ [4., 1.], [1., 2.] ])
		c = np.ones(nx)
		Aeq = np.ones((1,nx))
		beq = np.array([1.0])
		Aineq = np.zeros((0,nx))
		bineq = np.zeros(0)
		lb = np.zeros(nx)
		ub = np.Inf*np.ones(nx)
	else:
		assert False, "Unknown example"
	
	#todo: Generate QPs coming from MPC, finance, SVM, etc.
	
	# Solve QP via ADMM
	solADMM = OSQP(Q, c, Aeq, beq, Aineq, bineq, lb, ub)
	
	# Solve QP via cvxpy+ECOS
	x = cvx.Variable(nx)
	dict_constr = {}	# dict_constr has keys of all given constraints
	constraints = []	# list of constraints
	if beq.size:  # if beq is nonempty
		constraints = constraints + [Aeq*x == beq]
		dict_constr['eq'] = len(constraints)-1	# value assigned to each key is an index of the constraint
	if bineq.size:
		constraints = constraints + [Aineq*x <= bineq]
		dict_constr['ineq'] = len(constraints)-1
	if lb.size:
		constraints = constraints + [x >= lb]
		dict_constr['lb'] = len(constraints)-1
	if ub.size:
		constraints = constraints + [x >= lb]
		dict_constr['ub'] = len(constraints)-1
	objective = cvx.Minimize(.5*cvx.quad_form(x, Q) + c.T*x)
	problem = cvx.Problem(objective, constraints)
	problem.solve(solver=cvx.ECOS, verbose=False)
	# Get dual variables
	dual_eq   = np.asarray(constraints[dict_constr['eq']].dual_value).flatten()   if 'eq'   in dict_constr.keys() else np.zeros(0)
	dual_ineq = np.asarray(constraints[dict_constr['ineq']].dual_value).flatten() if 'ineq' in dict_constr.keys() else np.zeros(0)
	dual_lb   = np.asarray(constraints[dict_constr['lb']].dual_value).flatten()   if 'lb'   in dict_constr.keys() else np.zeros(0)
	dual_ub   = np.asarray(constraints[dict_constr['ub']].dual_value).flatten()   if 'ub'   in dict_constr.keys() else np.zeros(0)
	
	# Compare solutions
	print "ADMM Objective Value = %.3f\n" % solADMM.objval,
	print "ECOS Objective Value = %.3f" % objective.value
	
	# Compare dual variables of ADMM and ECOS solutions
	if 'eq' in dict_constr.keys():
		print "dual_eq DIFF   = %.4f" % npla.norm(dual_eq - solADMM.sol_dual_eq)
	if 'ineq' in dict_constr.keys():
		print "dual_ineq DIFF = %.4f" % npla.norm(dual_ineq - solADMM.sol_dual_ineq)
	if 'lb' in dict_constr.keys():
		print "dual_lb DIFF   = %.4f" % npla.norm(dual_lb - solADMM.sol_dual_lb)
	if 'ub' in dict_constr.keys():
		print "dual_ub DIFF   = %.4f" % npla.norm(dual_ub - solADMM.sol_dual_ub)
	

	
	
# Parsing optional command line arguments
if __name__ == '__main__':
	if len(sys.argv) == 1:
		main()
	else:
		main(sys.argv[1:])

	