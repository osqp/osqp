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
solvers = [
           s.OSQP,
           #  s.OSQP_polish,
           s.GUROBI,
           s.MOSEK,
           s.ECOS,
           s.qpOASES
           ]

settings = {
            s.OSQP: {'polish': False},
            #  s.OSQP_polish: {'polish': True},
            s.GUROBI: {},
            s.MOSEK: {},
            s.ECOS: {},
            s.qpOASES: {'nWSR': 1e6,    # Number of working set recalculations
                        'cputime': 120  # Seconds
                        }
            }

# Number of instances per different dimension
n_instances = 10

# Shut up solvers
for key in settings:
    settings[key]['verbose'] = False

# Run benchmark problems

# Random QP
random_qp = Example('Random QP',
                    [10, 20, 30],
                    solvers,
                    settings,
                    n_instances)
# random_qp.solve()

# Equality constrained QP
eq_qp = Example('Eq QP',
                [10, 20, 30],
                solvers,
                settings,
                n_instances)
# eq_qp.solve()

# Portfolio
portfolio = Example('Portfolio',
                    [2, 3, 4],
                    solvers,
                    settings,
                    n_instances)
portfolio.solve()


# Lasso
lasso = Example('Lasso',
                [2],
                solvers,
                settings,
                n_instances)
# lasso.solve()


# SVM
svm = Example('SVM',
              [2, 3, 4],
              solvers,
              settings,
              n_instances)
svm.solve()

# Huber
# Control
