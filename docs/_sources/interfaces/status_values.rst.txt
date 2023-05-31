.. _status_values :



Status values and errors
========================

Status values
-------------

These are the exit statuses, their respective constants and values returned by the solver as defined in `constants.h <https://github.com/osqp/osqp/blob/master/include/constants.h>`_.
The *inaccurate* statuses define when the optimality, primal infeasibility or dual infeasibility conditions are satisfied with tolerances 10 times larger than the ones set.

+------------------------------+-----------------------------------+-------+
| Status                       | Constant                          | Value |
+==============================+===================================+=======+
| solved                       | OSQP_SOLVED                       | 1     |
+------------------------------+-----------------------------------+-------+
| solved inaccurate            | OSQP_SOLVED_INACCURATE            | 2     |
+------------------------------+-----------------------------------+-------+
| primal infeasible            | OSQP_PRIMAL_INFEASIBLE            | 3     |
+------------------------------+-----------------------------------+-------+
| primal infeasible inaccurate | OSQP_PRIMAL_INFEASIBLE_INACCURATE | 4     |
+------------------------------+-----------------------------------+-------+
| dual infeasible              | OSQP_DUAL_INFEASIBLE              | 5     |
+------------------------------+-----------------------------------+-------+
| dual infeasible inaccurate   | OSQP_DUAL_INFEASIBLE_INACCURATE   | 6     |
+------------------------------+-----------------------------------+-------+
| maximum iterations reached   | OSQP_MAX_ITER_REACHED             | 7     |
+------------------------------+-----------------------------------+-------+
| run time limit reached       | OSQP_TIME_LIMIT_REACHED           | 8     |
+------------------------------+-----------------------------------+-------+
| problem non convex           | OSQP_NON_CVX                      | 9     |
+------------------------------+-----------------------------------+-------+
| interrupted by user          | OSQP_SIGINT                       | 10    |
+------------------------------+-----------------------------------+-------+
| unsolved                     | OSQP_UNSOLVED                     | 11    |
+------------------------------+-----------------------------------+-------+

.. note::

   We recommend the user to **check the convexity of their problem before
   passing it to OSQP**! If the user passes a non-convex problem we do not
   assure the solver will be able to detect it.

   OSQP will try to detect **non-convex** problems by checking if the residuals
   diverge or if there are any issues in the initial factorization (if a direct
   method is used). It will detect non-convex problems when one or more of the
   eigenvalues of :code:`P` are "clearly" negative, i.e., when :code:`P + sigma
   * I` is not positive semidefinite. However, it might fail to detect
   non-convexity when :code:`P` has slightly negative eigenvalues, i.e., when
   :code:`P + sigma * I` is positive semidefinite and :code:`P` is not.



	 
Solver Errors
-------------

OSQP can return errors during the setup and solve steps. Here is a table of the meaning
and their constant values.


+------------------------------------------------+-----------------------------------+-------+
| Errors                                         | Constant                          | Value |
+================================================+===================================+=======+
| No error                                       | OSQP_NO_ERROR                     | 0     |
+------------------------------------------------+-----------------------------------+-------+
| Data validation failed                         | OSQP_DATA_VALIDATION_ERROR        | 1     |
+------------------------------------------------+-----------------------------------+-------+
| Settings validation failed                     | OSQP_SETTINGS_VALIDATION_ERROR    | 2     |
+------------------------------------------------+-----------------------------------+-------+
| Linear system solver initialization failed     | OSQP_LINSYS_SOLVER_INIT_ERROR     | 3     |
+------------------------------------------------+-----------------------------------+-------+
| Non convex problem detected                    | OSQP_NONCVX_ERROR                 | 4     |
+------------------------------------------------+-----------------------------------+-------+
| Memory allocation error                        | OSQP_MEM_ALLOC_ERROR              | 5     |
+------------------------------------------------+-----------------------------------+-------+
| Workspace not initialized                      | OSQP_WORKSPACE_NOT_INIT           | 6     |
+------------------------------------------------+-----------------------------------+-------+
| Error loading algebra library                  | OSQP_ALGEBRA_LOAD_ERROR           | 7     |
+------------------------------------------------+-----------------------------------+-------+
| Error opening file for writing                 | OSQP_FOPEN_ERROR                  | 8     |
+------------------------------------------------+-----------------------------------+-------+
| Error validating given code generation defines | OSQP_CODEGEN_DEFINES_ERROR        | 9     |
+------------------------------------------------+-----------------------------------+-------+
| Solver data not initialized                    | OSQP_DATA_NOT_INITIALIZED         | 10    |
+------------------------------------------------+-----------------------------------+-------+
| Function not implemented in current algebra    | OSQP_FUNC_NOT_IMPLEMENTED         | 11    |
+------------------------------------------------+-----------------------------------+-------+
