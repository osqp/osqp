Matlab
======

.. _matlab_setup:

Setup
-----
The solver is initialized by creating an OSQP object

.. code:: matlab

    m = osqp;

The problem is specified in the setup phase by running

.. code:: matlab

    m.setup(P, q, A, l, u, varargin)


The arguments :code:`q`, :code:`l` and :code:`u` are arrays. The elements of :code:`l` and :code:`u` can be :math:`\pm \infty` ( using :code:`Inf`). The arguments :code:`P` and :code:`A` are sparse matrices.

There is no need to specify all the problem data. They can be omitted by writing :code:`[]`.

The last argument :code:`varargin` specifies the solver options. You can pass the options in two ways. You can either set the individual parameters as field-value pairs, e.g.,

.. code:: matlab

    m.setup(P, q, A, l, u, 'eps_abs', 1e-04, 'eps_rel', 1e-04);


Alternatively, you can create a structure containing all the settings, change some of the fields and then pass it as the last argument

.. code:: matlab

    settings = m.default_settings();
    settings.eps_abs = 1e-04;
    settings.eps_rel = 1e-04;
    m.setup(P, q, A, l, u, settings);

The allowed settings are defined in :ref:`solver_settings`.


Solve
-----

The problem can be solved by

.. code:: matlab

   results = m.solve();

The :code:`results` structure contains the primal solution :code:`x`, the dual solution :code:`y`, certificate of primal infeasibility :code:`prim_inf_cert`, certificate of dual infeasibility :code:`dual_inf_cert` and the :code:`info` structure containing the solver statistics defined in the following table


+-----------------------+-------------------------------------------------+
| Member                | Description                                     |
+=======================+=================================================+
| :code:`iter`          | Number of iterations                            |
+-----------------------+-------------------------------------------------+
| :code:`status`        | Solver status                                   |
+-----------------------+-------------------------------------------------+
| :code:`status_val`    | Solver status value as in :ref:`status_values`  |
+-----------------------+-------------------------------------------------+
| :code:`status_polish` | Polishing status                                |
+-----------------------+-------------------------------------------------+
| :code:`obj_val`       | Objective value                                 |
+-----------------------+-------------------------------------------------+
| :code:`pri_res`       | Primal residual                                 |
+-----------------------+-------------------------------------------------+
| :code:`dua_res`       | Dual residual                                   |
+-----------------------+-------------------------------------------------+
| :code:`run_time`      | Total run time                                  |
+-----------------------+-------------------------------------------------+
| :code:`setup_time`    | Setup time                                      |
+-----------------------+-------------------------------------------------+
| :code:`polish_time`   | Polish time                                     |
+-----------------------+-------------------------------------------------+


Update
------
Part of problem data and settings can be updated without requiring a new problem setup.



Update problem data
^^^^^^^^^^^^^^^^^^^

Vectors :code:`q`, :code:`l` and :code:`u` can be updated with new values :code:`q_new`, :code:`l_new` and :code:`u_new` by just running

.. code:: python

    m.update('q', q_new, 'l', l_new, 'u', u_new);


The user does not have to specify all the arguments.


Update settings
^^^^^^^^^^^^^^^

Settings can be updated by running

.. code:: python

    m.update_settings(varargin);


where :code:`varargin` argument is described in :ref:`matlab_setup`. The allowed settings that can be updated are marked with an * in :ref:`solver_settings`.




Warm start
----------
OSQP automatically warm starts primal and dual variables from the previous QP solution. If you would like to warm start their values manually, you can use

.. code:: matlab

    m.warm_start('x', x0, 'y', y0)

where :code:`x0` and :code:`y0` are the new primal and dual variables. Please make sure :code`warm_start` solver option is active.
