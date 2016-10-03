#!/usr/bin/env python

# Test QP solver against Maros Mezaros Benchmark suite
import sys
import scipy.io as spio
import scipy.sparse as spspa
import numpy as np
import ipdb
import osqp
import cplex as cpx     # Using CPLEX solver to compare results
import gurobipy as grb  # Use Gurobi to compare results

reload(osqp)


def solveCPLEX(Q, c, Aeq, beq, Aineq, bineq, lb, ub):
    # Convert Matrices in CSR format
    Aeq = Aeq.tocsr()
    Aineq = Aineq.tocsr()
    Q = 2.*Q.tocsr()

    # Convert Q matrix to COO format
    # Q = Q.tocoo()

    # Get problem dimensions
    nx = Q.shape[0]
    neq = Aeq.shape[0]
    nineq = Aineq.shape[0]

    # Adjust infinity values in bounds
    for i in range(nx):
        if ub[i] == -np.inf:
            ub[i] = -cpx.infinity
        if ub[i] == np.inf:
            ub[i] = cpx.infinity
        if lb[i] == -np.inf:
            lb[i] = -cpx.infinity
        if lb[i] == np.inf:
            lb[i] = cpx.infinity

    # Define CPLEX problem
    p = cpx.Cplex()

    # Minimize problem
    p.objective.set_sense(p.objective.sense.minimize)

    # Add variables
    p.variables.add(obj=c,           # Linear objective part
                    ub=ub, lb=lb)    # Lower and upper bounds

    # Add constraints
    sense = ["E"]*neq + ["L"]*nineq  # Constraints sense: == and <=
    rows = []
    for i in range(neq):  # Add equalities
        start = Aeq.indptr[i]
        end = Aeq.indptr[i+1]
        rows.append([Aeq.indices[start:end].tolist(),
                     Aeq.data[start:end].tolist()])
    for i in range(nineq):  # Add inequalities
        start = Aineq.indptr[i]
        end = Aineq.indptr[i+1]
        rows.append([Aineq.indices[start:end].tolist(),
                     Aineq.data[start:end].tolist()])
    p.linear_constraints.add(lin_expr=rows,
                             senses=sense,
                             rhs=np.hstack([beq, bineq]).tolist())

    # Set quadratic Cost
    qmat = []
    for i in range(nx):
        start = Q.indptr[i]
        end = Q.indptr[i+1]
        qmat.append([Q.indices[start:end].tolist(),
                    Q.data[start:end].tolist()])
    p.objective.set_quadratic(qmat)

    # Solve problem
    p.solve()

    # Return solution
    return p


def solveGUROBI(Q, c, Aeq, beq, Aineq, bineq, lb, ub):

    # Convert Matrices in CSR format
    Aeq = Aeq.tocsr()
    Aineq = Aineq.tocsr()

    # Convert Q matrix to COO format
    Q = Q.tocoo()

    # Get problem dimensions
    nx = Q.shape[0]
    neq = Aeq.shape[0]
    nineq = Aineq.shape[0]

    # Create a new model
    m = grb.Model("qp")

    # Add variables
    for i in range(nx):
        m.addVar(lb=lb[i], ub=ub[i], obj=c[i])
    m.update()
    x = m.getVars()

    # Add equality constraints: iterate over the rows of Aeq
    # adding each row into the model
    for i in range(neq):
        start = Aeq.indptr[i]
        end = Aeq.indptr[i+1]
        variables = [x[j] for j in Aeq.indices[start:end]]  # Get nnz
        coeff = Aeq.data[start:end]
        expr = grb.LinExpr(coeff, variables)
        m.addConstr(lhs=expr, sense=grb.GRB.EQUAL, rhs=beq[i])

    # Add inequality constraints: iterate over the rows of Aeq
    # adding each row into the model
    for i in range(nineq):
        start = Aineq.indptr[i]
        end = Aineq.indptr[i+1]
        variables = [x[j] for j in Aineq.indices[start:end]]  # Get nnz
        coeff = Aineq.data[start:end]
        expr = grb.LinExpr(coeff, variables)
        m.addConstr(lhs=expr, sense=grb.GRB.LESS_EQUAL, rhs=bineq[i])

    # Set quadratic cost
    obj = grb.QuadExpr()
    for i in range(Q.nnz):
        obj += Q.data[i]*x[Q.row[i]]*x[Q.col[i]]
    m.setObjective(obj)

    # Update model
    m.update()

    # Solve
    m.optimize()

    # Return model
    return m, m.getConstrs(), x


def main():
    # for file in os.listdir('tests/maros_meszaros'):
    # Do all the tests
    p = spio.loadmat('tests/maros_meszaros/CVXQP2_S.mat')
    # p = spio.loadmat('tests/maros_meszaros/AUG2D.mat')
    Q = p['Q'].astype(float)  # Convert to dense matrix (To remove)
    c = p['c'].T.flatten().astype(float)
    Aeq = p['A'].astype(float)  # Convert to dense matrix (To remove)
    beq = p['ru'].T.flatten().astype(float)
    lb = p['lb'].T.flatten().astype(float)
    ub = p['ub'].T.flatten().astype(float)
    nx = Q.shape[0]
    Aineq = spspa.csc_matrix(np.zeros((1, nx)))
    bineq = np.array([0.0])

    # Solve problem with OSQP, CPLEX and GUROBi
    solOSQP = osqp.OSQP(Q, c, Aeq, beq, Aineq, bineq, lb, ub,
                        max_iter=5000, print_level=2)

    solCPLEX = solveCPLEX(Q, c, Aeq, beq, Aineq, bineq, lb, ub)
    solGUROBI, cGUROBI, xGUROBI = solveGUROBI(Q, c, Aeq, beq, Aineq, bineq, lb, ub)

    print "Objective Gurobi = %.2f" % solGUROBI.getObjective().getValue()
    print "Objective CPLEX = %.2f" % solCPLEX.solution.get_objective_value()

    print "Norm of objective value difference %.4f" % \
        np.linalg.norm(solOSQP.objval*2. - solCPLEX.solution.get_objective_value())
    print "Norm of solution difference %.4f" % \
        np.linalg.norm(solOSQP.sol_prim - solCPLEX.solution.get_values())

    ipdb.set_trace()



# Parsing optional command line arguments
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1:])
