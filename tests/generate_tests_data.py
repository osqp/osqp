# Code to generate the unittests for OSQP C code
import numpy as np

# Set numpy seed for reproducibility
np.random.seed(2)

import basic_qp.generate_problem
import basic_qp2.generate_problem
import lin_alg.generate_problem
import solve_linsys.generate_problem
import infeasibility.generate_problem
import update_matrices.generate_problem
