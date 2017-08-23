from solvers.ecos import ECOSSolver
from solvers.gurobi import GUROBISolver
from solvers.mosek import MOSEKSolver
from solvers.osqp import OSQPSolver

ECOS = 'ECOS'
GUROBI = 'GUROBI'
OSQP = 'OSQP'
MOSEK = 'MOSEK'
qpOASES = 'qpOASES'

solvers = [ECOSSolver, GUROBISolver, MOSEKSolver, OSQPSolver]

SOLVER_MAP = {solver.name(): solver for solver in solvers}
