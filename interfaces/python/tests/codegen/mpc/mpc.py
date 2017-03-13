#from __future__ import print_function
import numpy as np
import scipy.sparse as sp
import scipy as scp
import osqp
from numpy import eye, diag, ones, kron, vstack, hstack, zeros, diagonal
from scipy.linalg import block_diag
import shutil
import os


import cvxpy as cvx

vs = vstack
hs = hstack

np.random.seed(1)

m = 3
n = 6
T = 20
dt = .1

Ad = np.eye(n) + dt * np.random.randn(n,n)
Bd = dt * np.random.randn(n,m)
Q = np.random.randn(10*n,n)/10
Q = Q.T.dot(Q)
R = np.random.randn(10*m,m)/100
R = R.T.dot(R)
x0 = np.random.randn(n,1) / dt
#x0 = copy(xmax)
xmin = -200
xmax =  200
umax =  20

# Dynamics.
A = kron(eye(T,T+1), Ad)
A += kron(hs([diag(ones(T-1), 1), vs([zeros((T-1,1)),1])]), -eye(n,n))
A = hs([A, kron(eye(T,T), Bd)])
l = zeros((n*T,1))
u = zeros((n*T,1))

# Initial condition.
A = vs([ A, hs([ eye(n), zeros((n,T*(n+m))) ]) ])
l = vs([ l, x0 ])
u = vs([ u, x0 ])

# State bounds.
A = vs([ A, hs([ zeros((n*T, n)), eye(n*T), zeros((n*T, T*m)) ]) ])
l = vs([ l, hs([ kron(ones((n*T,1)), xmin) ]) ])
u = vs([ u, hs([ kron(ones((n*T,1)), xmax) ]) ])

# State bounds.
A = vs([ A, hs([ zeros((m*T, (T+1)*n)), eye(m*T) ]) ])
l = vs([ l, hs([ kron(ones((m*T,1)), -umax) ]) ])
u = vs([ u, hs([ kron(ones((m*T,1)), umax) ]) ])


P = block_diag(zeros((n,n)), kron(eye(T), Q), kron(eye(T), R))
q = zeros(((T+1)*n + T*m))


def solve_mpc_cvx(Ad, Bd, Q, R, T, x_0, umax):
    x = cvx.Variable(n, T+1)
    u = cvx.Variable(m, T)

    states = []
    for t in range(T):
        cost = cvx.quad_form(x[:,t+1],Q) + cvx.quad_form(u[:,t], R)
        constr = [x[:,t+1] == Ad*x[:,t] + Bd*u[:,t]]
        constr += [cvx.norm(u[:,t], 'inf') <= umax]
        states.append(cvx.Problem(cvx.Minimize(cost), constr) )
    prob = sum(states)
    prob.constraints += [x[:,0] == x_0]
    prob.solve(verbose=True)
    return x.value, u.value, prob.value

def solve_qp_cvx(P, q, A, l, u):
    n = P.shape[0]
    x = cvx.Variable(n)

    cost = cvx.quad_form(x, P) + q.T*x
    constr = [l <= A*x, A*x <= u]
    prob = cvx.Problem(cvx.Minimize(cost), constr)
    prob.solve()
    return x.value, prob.value

 
xstar, ustar, pstar = solve_mpc_cvx(Ad, Bd, Q, R, T, x0, umax)
#xstar2, pstar2 = solve_qp_cvx(P, q, A, l, u)


## Pass the data to OSQP
#m = osqp.OSQP()
#m.setup(sp.csc_matrix(P), q, sp.csc_matrix(A), l, u, rho=0.1)
#
## Generate the code
#w = m.codegen("code", 'Unix Makefiles', early_terminate=1, embedded_flag=1)
#
#shutil.copy('example.c', os.path.join('code', 'src', 'example.c'))
