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
from benchmark_problems.utils import gen_int_log_space

# Define solvers
solvers = [
            s.OSQP,
            # s.OSQP_polish,
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
             s.qpOASES: {'nWSR': 1000000,    # Number of working set recalcs
                         'cputime': 900.     # Seconds (N.B. Must be float!)
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
                    gen_int_log_space(10, 10000, 20),
                    solvers,
                    settings,
                    n_instances)
random_qp.solve()

# Equality constrained QP
eq_qp = Example('Eq QP',
                gen_int_log_space(10, 10000, 20),
                solvers,
                settings,
                n_instances)
eq_qp.solve()

# Portfolio
portfolio = Example('Portfolio',
                    gen_int_log_space(5, 150, 20),
                    solvers,
                    settings,
                    n_instances)
portfolio.solve()


# Lasso
lasso = Example('Lasso',
                gen_int_log_space(10, 1000, 20),
                solvers,
                settings,
                n_instances)
lasso.solve()


# SVM
svm = Example('SVM',
              gen_int_log_space(10, 1000, 20),
              solvers,
              settings,
              n_instances)
svm.solve()


# Huber
huber = Example('Huber',
                gen_int_log_space(10, 1000, 20),
                solvers,
                settings,
                n_instances)
huber.solve()


# Control
control = Example('Control',
                  gen_int_log_space(4, 100, 20),
                  solvers,
                  settings,
                  n_instances)
control.solve()
