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
	def __init__(self, status, objval, sol):
		self.status = status
		self.objval = objval
		self.sol = sol


def project(xbar, lb, ub):
	nx = np.size(lb)
	
	# Round x part to [l, u] interval
	xbar[:nx] = np.minimum(np.maximum(xbar[:nx], lb), ub)
	
	# Round slack variables to positive ortant
	xbar[nx:] = np.maximum(xbar[nx:], 0)
	
	return xbar


# Base QP Solver
def SQPSSolve(Q, c, Aeq, beq, Aineq, bineq, lb, ub, x0=0, max_iter=500, rho=1.6):
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
	
	# Reformat missing data
	
	
	# Form complete (c) matrices for inequality constraints
	Ac = np.vstack([np.hstack([Aeq, np.zeros((neq, nineq))]), np.hstack([Aineq, np.eye(nineq)])])
	bc = np.append(beq, bineq)
	Qc = spla.block_diag(Q, np.zeros((nineq, nineq)))
	cc = np.append(c, np.zeros(nineq))
	
	# Factorize Matrices (Later)
	M = np.vstack([np.hstack([Qc + rho*np.eye(nvar), Ac.T]), np.hstack([Ac, -1./rho*np.eye(nineq + neq)])])
	
	# Factorize matrix M using LU decomposition
	LU, piv = spla.lu_factor(M)
	
	print "Splitting QP Solver"
	print "-------------------\n"
	print "iter |\t %6s\n" % 'cost'
	
	# Run ADMM
	z = np.zeros(nvar)
	u = np.zeros(neq + nineq + nvar)  # Number of dual variables in ADMM
	
	for i in range(max_iter):
		qtemp = -cc + rho*(z + np.dot(Ac.T, bc) - np.dot(Ac.T, u[:neq + nineq]) - u[neq + nineq:])
		
		qbar = np.append(qtemp, np.zeros(nineq+neq))
		
		# x update
		x = spla.lu_solve((LU, piv), qbar)[:nvar]	# Select only first nvar elements
		# z update
		z = project(x + u[neq+nineq:], lb, ub)
		# u update
		u = u + np.append(np.dot(Ac, x), x) - np.append(np.zeros(neq + nineq), z) - np.append(bc, np.zeros(nvar))
		
		# Compute cost function
		xtemp = x[:nx]
		f = .5*np.dot(np.dot(xtemp.T, Q), xtemp) + np.dot(c.T, xtemp)
		
		if (i+1 == 1) | (np.mod(i+1, 25) == 0):
			print "%4s | \t%7.2f" % (i+1, f)
	
	print "Optimization Done\n"
	
	sol = x[:nx]
	objval = .5*np.dot(np.dot(sol, Q), sol) + np.dot(c, sol)
	return quadProgSolution(OPTIMAL, objval, sol)


def isPSD(A, tol=1e-8):
	E, _ = sp.linalg.eigh(A)
	return np.all(E > -tol)





def main(argv=[]):
    """
    Handle command line arguments from the operating system
    """
	
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
		# Generate a small example
		Q = np.array([ [4., 1.], [1., 2.] ])
		c = np.ones(2)
		Aeq = np.ones((1,2))
		beq = np.array([1.0])
		Aineq = np.zeros((0,nx))
		bineq = np.zeros(0)
		lb = np.zeros(2)
		ub = np.Inf*np.ones(2)
	else:
		assert False, "Unknown example"
	
	# Solve QP ADMM
	results = SQPSSolve(Q, c, Aeq, beq, Aineq, bineq, lb, ub)
	
	# Solve QP with cvxpy
	x = cvx.Variable(nx)
	constraints = []
	if beq.size:  # if beq is nonempty
		constraints = constraints + [Aeq*x == beq]
	if bineq.size:
		constraints = constraints + [Aineq*x <= bineq]
	if lb.size:
		constraints = constraints + [x >= lb]
	if ub.size:
		constraints = constraints + [x >= lb]
	objective = cvx.Minimize(.5*cvx.quad_form(x, Q) + c.T*x)
	problem = cvx.Problem(objective, constraints)
	resultECOS = problem.solve(solver=cvx.ECOS, verbose=False)
	
	# Compare solutions
	print "ADMM Objective Value = %.3f\n" % results.objval,
	print "ECOS Objective Value = %.3f" % objective.value
	
	#ipdb.set_trace()



	
# Parsing optional command line arguments
if __name__ == '__main__':
	if len(sys.argv) == 1:
		main()
	else:
		main(sys.argv[1:])

	