.. _status_values :

Status values
==============

These are the exit statuses, their respective constants and values returned by the solver as defined in `constants.h <https://github.com/oxfordcontrol/osqp/blob/master/include/constants.h>`_.
The *inaccurate* statuses define when the optimality, primal infeasibility or dual infeasibility conditions are satisfied with tolerances 10 times larger than the ones set.

+---------------------------------+------------------------------------+----------+
| Status                          | Constant                           | Value    |
+=================================+====================================+==========+
| solved                          | OSQP_SOLVED                        | 1        |
+---------------------------------+------------------------------------+----------+
| solved inaccurate               | OSQP_SOLVED_INACCURATE             | 2        |
+---------------------------------+------------------------------------+----------+
| maximum iterations reached      | OSQP_MAX_ITER_REACHED              | -2       |
+---------------------------------+------------------------------------+----------+
| primal infeasible               | OSQP_PRIMAL_INFEASIBLE             | -3       |
+---------------------------------+------------------------------------+----------+
| primal infeasible inaccurate    | OSQP_PRIMAL_INFEASIBLE_INACCURATE  | 3        |
+---------------------------------+------------------------------------+----------+
| dual infeasible                 | OSQP_DUAL_INFEASIBLE               | -4       |
+---------------------------------+------------------------------------+----------+
| dual infeasible inaccurate      | OSQP_DUAL_INFEASIBLE_INACCURATE    | 4        |
+---------------------------------+------------------------------------+----------+
| interrupted by user             | OSQP_SIGINT                        | -5       |
+---------------------------------+------------------------------------+----------+
| unsolved                        | OSQP_UNSOLVED                      | -10      |
+---------------------------------+------------------------------------+----------+
