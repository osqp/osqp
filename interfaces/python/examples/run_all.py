'''
Run all benchmarks for the OSQP paper

This code tests the solvers:
    - OSQP
    - GUROBI
    - MOSEK
    - ECOS
    - qpOASES

'''
from benchmark_example import Example
from solvers.solvers \
    import OSQP, ECOS, GUROBI, MOSEK, qpOASES

# Define solvers
solvers = [OSQP,
           GUROBI,
           MOSEK,
           ECOS,
           qpOASES]

settings = {OSQP: {},
            GUROBI: {},
            MOSEK: {},
            ECOS: {},
            qpOASES: {}}

# Run benchmark problems
random_qp = Example('Random QP',
                    [10, 20, 30],
                    solvers,
                    settings)
random_qp.solve()
