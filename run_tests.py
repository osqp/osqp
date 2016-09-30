#!/usr/bin/env python

# Test QP solver against Maros Mezaros Benchmark suite
import sys
import scipy.io as spio
import numpy as np
import ipdb
import osqp
import cplex  # Using CPLEX solver to compare results

reload(osqp)


def solveCPLEX(Q, c, Aeq, beq, Aineq, bineq, lb, ub):  # Define CPLEX data

    # Get problem dimensions
    nx = Q.shape[0]
    neq = Aeq.shape[0]
    nineq = Aineq.shape[0]

    # Define CPLEX problem
    p = cplex.Cplex()

    # Minimize problem
    p.objective.set_sense(p.objective.sense.minimize)

    # Add variables
    p.variables.add(obj=c,           # Linear objective part
                    ub=ub, lb=lb)    # Lower and upper bounds

    # Add constraints
    sense = ["E"]*neq + ["L"]*nineq  # Constraints sense: == and <=
    rows = []
    for i in range(neq):  # Add equalities
        rows.append([range(nx), Aeq[i, :].tolist()])
    for i in range(nineq):  # Add inequalities
        rows.append([range(nx), Aineq[i, :].tolist()])
    p.linear_constraints.add(lin_expr=rows,
                             senses=sense,
                             rhs=np.hstack([beq, bineq]).tolist())
                            #  names=["c_"+str(i) for i in range(neq + nineq)])

    # Set quadratic Cost
    qmat = []
    for i in range(nx):
        qmat.append([range(nx), Q[i, :].tolist()])
    p.objective.set_quadratic(qmat)

    # Solve problem
    p.solve()

    # Return solution
    return p


def main():
    # for file in os.listdir('tests/maros_meszaros'):
    # Do all the tests
    p = spio.loadmat('tests/maros_meszaros/CVXQP2_S.mat')
    Q = np.asarray(p['Q'].todense()).astype(float)  # Convert to dense matrix (To remove)
    c = p['c'].T.flatten().astype(float)
    Aeq = np.asarray(p['A'].todense()).astype(float)  # Convert to dense matrix (To remove)
    beq = p['ru'].T.flatten().astype(float)
    lb = p['lb'].T.flatten().astype(float)
    ub = p['ub'].T.flatten().astype(float)
    nx = Q.shape[0]
    Aineq = np.zeros((1, nx))
    bineq = np.array([0.0])

    # Solve problem with OSQP and CPLEX
    solOSQP = osqp.OSQP(Q, c, Aeq, beq, Aineq, bineq, lb, ub,
                        max_iter=5000, print_level=2)
    solCPLEX = solveCPLEX(Q, c, Aeq, beq, Aineq, bineq, lb, ub)

    print "Norm of objective value difference %.4f" % \
        np.linalg.norm(solOSQP.objval - solCPLEX.solution.get_objective_value())
    print "Norm of solution difference %.4f" % \
        np.linalg.norm(solOSQP.sol_prim - solCPLEX.solution.get_values())







# Parsing optional command line arguments
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1:])
