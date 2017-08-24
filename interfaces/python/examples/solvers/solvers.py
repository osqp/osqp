from solvers.ecos import ECOSSolver
from solvers.gurobi import GUROBISolver
from solvers.mosek import MOSEKSolver
from solvers.osqp import OSQPSolver
from solvers.qpoases import qpOASESSolver

ECOS = 'ECOS'
GUROBI = 'GUROBI'
OSQP = 'OSQP'
OSQP_polish = OSQP + '_polish'
MOSEK = 'MOSEK'
qpOASES = 'qpOASES'

# solvers = [ECOSSolver, GUROBISolver, MOSEKSolver, OSQPSolver]
# SOLVER_MAP = {solver.name(): solver for solver in solvers}


SOLVER_MAP = {OSQP: OSQPSolver,
              OSQP_polish: OSQPSolver,
              GUROBI: GUROBISolver,
              MOSEK: MOSEKSolver,
              ECOS: ECOSSolver,
              qpOASES: qpOASESSolver}
