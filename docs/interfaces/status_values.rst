.. _status_values :

Status values
==============

These are the exit statuses, their respective constants and values returned by the solver as defined in `constants.h <https://github.com/oxfordcontrol/osqp/blob/master/include/constants.h>`_.
The *inaccurate* statuses define when the optimality, primal infeasibility or dual infeasibility conditions are satisfied with tolerances 10 times larger than the ones set.

+---------------------------------+------------------------------------+----------+
| Status                          | Constant                           | Value    |
+=================================+====================================+==========+
| Solved                          | OSQP_SOLVED                        | 1        |
+---------------------------------+------------------------------------+----------+
| Solved inaccurate               | OSQP_SOLVED_INACCURATE             | 2        |
+---------------------------------+------------------------------------+----------+
| Maximum iterations reached      | OSQP_MAX_ITER_REACHED              | -2       |
+---------------------------------+------------------------------------+----------+
| Primal infeasible               | OSQP_PRIMAL_INFEASIBLE             | -3       |
+---------------------------------+------------------------------------+----------+
| Primal infeasible inaccurate    | OSQP_PRIMAL_INFEASIBLE_INACCURATE  | 3        |
+---------------------------------+------------------------------------+----------+
| Dual infeasible                 | OSQP_DUAL_INFEASIBLE               | -4       |
+---------------------------------+------------------------------------+----------+
| Dual infeasible inaccurate      | OSQP_DUAL_INFEASIBLE_INACCURATE    | 4        |
+---------------------------------+------------------------------------+----------+
| Interrupted by user             | OSQP_SIGINT                        | -5       |
+---------------------------------+------------------------------------+----------+
| Unsolved                        | OSQP_UNSOLVED                      | -10      |
+---------------------------------+------------------------------------+----------+
