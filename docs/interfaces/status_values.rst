.. _status_values :

Status values
==============

These are the exit statuses, their respective constants and values returned by the solver as defined in `constants.h <https://github.com/oxfordcontrol/osqp/blob/master/include/constants.h>`_

+-----------------------------+------------------------+----------+
| Status                      | Constant               | Value    |
+=============================+========================+==========+
| Solved                      | OSQP_SOLVED            | 1        |
+-----------------------------+------------------------+----------+
| Maximum iterations reached  | OSQP_MAX_ITER_REACHED  | -2       |
+-----------------------------+------------------------+----------+
| Primal infeasible           | OSQP_PRIMAL_INFEASIBLE | -3       |
+-----------------------------+------------------------+----------+
| Dual infeasible             | OSQP_DUAL_INFEASIBLE   | -4       |
+-----------------------------+------------------------+----------+
| Interrupted by user         | OSQP_SIGINT            | -5       |
+-----------------------------+------------------------+----------+
| Unsolved                    | OSQP_UNSOLVED          | -10      |
+-----------------------------+------------------------+----------+
