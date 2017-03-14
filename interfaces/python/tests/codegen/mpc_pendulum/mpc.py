from __future__ import print_function
import numpy as np
import scipy.sparse as sp
import osqp
from builtins import range


def b(x, n, N):
    b = np.zeros((N+1)*n)
    b[:n] = -x
    return b


# Model of the system
dyn_A = sp.csc_matrix([[1., 0.02, 0., 0.], [0., 1., 0., 0.],
                       [0., 0., 1.0049, 0.02], [0., 0., 0.4913, 1.0049]])
dyn_B = sp.csc_matrix([[0.0002], [0.02], [-0.0005], [-0.0501]])
[n, m] = dyn_B.shape

# Objective function
obj_Q = sp.diags([1., 0.3, 0.3, 0.1])
obj_R = sp.diags([0.1])
# QN = dare(A,B,Q,R)
obj_QN = sp.csc_matrix([[69.4161, 39.9917, 111.6822, 22.3337],
                        [39.9917, 43.8824, 131.5453, 26.2255],
                        [111.6822, 131.5453, 618.2991, 121.9162],
                        [22.3337, 26.2255, 121.9162, 24.6494]])

# Input constraints
constr_D = sp.diags([1.0])
constr_dl = -5.0
constr_du = 5.0

# State constraints
constr_E = sp.eye(n)
constr_el = np.array([-0.5, -1.0, -0.2, -0.5])
constr_eu = np.array([0.5, 1.0, 0.2, 0.5])

# Terminal state constraints
constr_F = constr_E
constr_fl = constr_el
constr_fu = constr_eu

# Prediction horizon
N = 10

'''
MPC Matrices
'''
# Objective
Hx = sp.kron(sp.eye(N), obj_Q)
Hu = sp.kron(sp.eye(N), obj_R)
H = sp.block_diag([Hx, obj_QN, Hu])

# Dynamics
Mx = sp.kron(sp.eye(N+1), -sp.eye(n)) + sp.kron(sp.eye(N+1, k=-1), dyn_A)
Mu = sp.kron(sp.vstack([sp.csc_matrix((1, N)), sp.eye(N)]), dyn_B)
M = sp.hstack([Mx, Mu])

# Constraints
Gx = sp.kron(sp.eye(N), constr_E)
Gu = sp.kron(sp.eye(N), constr_D)
G = sp.block_diag([Gx, constr_F, Gu])
gl = np.hstack([np.tile(constr_el, N), constr_fl, np.tile(constr_dl, N)])
gu = np.hstack([np.tile(constr_eu, N), constr_fu, np.tile(constr_du, N)])


# Pass the data to OSQP
x = np.array([0., 0., 0.15, 0.])  # initial state: [p, p_dot, theta, theta_dot]
P = H
q = np.zeros((N+1)*n + N*m)
A = sp.vstack([M, G])
l = np.hstack([b(x, n, N), gl])
u = np.hstack([b(x, n, N), gu])

m = osqp.OSQP()
m.setup(P, q, A, l, u, rho=1e-1, sigma=1e-4)

# Generate the code
try:
    import emosqp
except:
    m.codegen("code", 'MinGW Makefiles', early_terminate=1, embedded_flag=1)
    import emosqp

'''
Apply MPC in closed loop
'''


np.random.seed(1)

# Apply MPC to the system
sim_steps = 20

B = dyn_B.T.toarray()[0]
for i in range(sim_steps):

    # Solve
    sol = emosqp.solve()
    u = sol[0][(N+1)*n]
    status_val = sol[2]
    numofiter = sol[3]

    print(numofiter)

    # Compute details
    if status_val != 1:
        print('Problem unsolved')
        break

    # Apply first control input to the plant
    x = dyn_A.dot(x) + B.dot(u)

    # Update initial state
    l_new = np.hstack([b(x, n, N), gl])
    u_new = np.hstack([b(x, n, N), gu])
    emosqp.update_bounds(l_new, u_new)
