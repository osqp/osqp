'''
Run all benchmarks for the OSQP paper

This code tests the solvers:
    - OSQP
    - GUROBI
    - MOSEK
    - ECOS
    - qpOASES

'''
from benchmark_problems.example import Example
import solvers.solvers as s

# Define solvers
solvers = [s.OSQP,
           s.OSQP_polish,
           s.GUROBI,
           s.MOSEK,
           s.ECOS,
           s.qpOASES]

settings = {s.OSQP: {'polish': False},
            s.OSQP_polish: {'polish': True},
            s.GUROBI: {},
            s.MOSEK: {},
            s.ECOS: {},
            s.qpOASES: {}}

# Shut up solvers
for key in settings:
    settings[key]['verbose'] = False

# Run benchmark problems

# Random QP
random_qp = Example('Random QP',
                    [10, 20, 30],
                    solvers,
                    settings)
# random_qp.solve()

# Equality constrained QP
eq_qp = Example('Eq QP',
                [10, 20, 30],
                solvers,
                settings)
# eq_qp.solve()

# Portfolio
portfolio = Example('Portfolio',
                    [200, 200, 400],
                    solvers,
                    settings)
portfolio.solve()
