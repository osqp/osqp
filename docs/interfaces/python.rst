Python
======

Import
------
The OSQP module is can be imported with

.. code:: python

    import osqp

The solver is initialized by creating an OSQP object

.. code:: python

    m = osqp.OSQP()


.. _python_setup:

Setup
-----
The problem is specified in the setup phase by running

.. code:: python

    m.setup(P=P, q=q, A=A, l=l, u=u, **settings)


The arguments :code:`q`, :code:`l` and :code:`u` are numpy arrays. The elements of :code:`l` and :code:`u` can be :math:`\pm \infty` ( using :code:`+numpy.inf`).

The arguments :code:`P` and :code:`A` are scipy sparse matrices in CSC format. If they are sparse matrices are in another format, the interface will attemp to convert them. There is no need to specify all the arguments.


The keyword arguments :code:`**settings` specify the solver settings as follows


+------------------------------------+-------------------------------------+----------------+
| Argument                           | Description                         | Default value  |
+====================================+=====================================+================+
| :code:`scaling`                    | Perform data scaling                |   True         |
+------------------------------------+-------------------------------------+----------------+
| :code:`rho`                        | ADMM rho step                       | Auto computed  |
+------------------------------------+-------------------------------------+----------------+
| :code:`auto_rho`                   | ADMM rho step automatic selection   |   True         |
+------------------------------------+-------------------------------------+----------------+
| :code:`sigma`                      | ADMM sigma step                     |   0.001        |
+------------------------------------+-------------------------------------+----------------+
| :code:`max_iter` *                 | Maximum number of iterations        |   2500         |
+------------------------------------+-------------------------------------+----------------+
| :code:`eps_abs`  *                 | Absolute tolerance                  |   1e-03        |
+------------------------------------+-------------------------------------+----------------+
| :code:`eps_rel`  *                 | Relative tolerance                  |   1e-03        |
+------------------------------------+-------------------------------------+----------------+
| :code:`eps_prim_inf`  *            | Primal infeasibility tolerance      |   1e-04        |
+------------------------------------+-------------------------------------+----------------+
| :code:`eps_dual_inf`  *            | Dual infeasibility tolerance        |   1e-04        |
+------------------------------------+-------------------------------------+----------------+
| :code:`alpha`    *                 | ADMM overrelaxation parameter       |   1.6          |
+------------------------------------+-------------------------------------+----------------+
| :code:`delta`    *                 | Polishing regularization parameter  |   1e-07        |
+------------------------------------+-------------------------------------+----------------+
| :code:`polish` *                   | Perform polishing                   |   True         |
+------------------------------------+-------------------------------------+----------------+
| :code:`verbose`  *                 | Print output                        |   True         |
+------------------------------------+-------------------------------------+----------------+
| :code:`early_terminate` *          | Evaluate termination criteria       |   True         |
+------------------------------------+-------------------------------------+----------------+
| :code:`early_terminate_interval` * | Interval for checking termination   |   25           |
+------------------------------------+-------------------------------------+----------------+
| :code:`warm_start` *               | Perform warm starting               |   True         |
+------------------------------------+-------------------------------------+----------------+
| :code:`scaling_norm`               | Scaling norm                        |   2            |
+------------------------------------+-------------------------------------+----------------+
| :code:`scaling_iter`               | Scaling iterations                  |   3            |
+------------------------------------+-------------------------------------+----------------+
| :code:`pol_refine_iter` *          | Refinement iterations in polish     |   5            |
+------------------------------------+-------------------------------------+----------------+

The settings marked with * can be changed without running the setup method again. See section :ref:`python_update_settings`.

Solve
-----

The problem can be solved by

.. code:: python

   results = m.solve()


The :code:`results` object contains the primal solution :code:`x`, the dual solution :code:`y` and the :code:`info` object containing the solver statistics defined in the following table


+-----------------------+----------------------+
| Member                | Description          |
+=======================+======================+
| :code:`iter`          | Number of iterations |
+-----------------------+----------------------+
| :code:`status`        | Solver status        |
+-----------------------+----------------------+
| :code:`status_val`    | Solver status code   |
+-----------------------+----------------------+
| :code:`status_polish` | Polishing status     |
+-----------------------+----------------------+
| :code:`obj_val`       | Objective value      |
+-----------------------+----------------------+
| :code:`pri_res`       | Primal residual      |
+-----------------------+----------------------+
| :code:`dua_res`       | Dual residual        |
+-----------------------+----------------------+
| :code:`run_time`      | Total run time       |
+-----------------------+----------------------+
| :code:`setup_time`    | Setup time           |
+-----------------------+----------------------+
| :code:`polish_time`   | Polish time          |
+-----------------------+----------------------+





Update
------
Problem vectors and part of the settings can be updated without requiring a new problem setup.

Update problem vectors
^^^^^^^^^^^^^^^^^^^^^^
Vectors :code:`q`, :code:`l` and :code:`u` can be updated with new values :code:`q_new`, :code:`l_new` and :code:`u_new` by just running

.. code:: python

    m.update(q=q_new, l=l_new, u=u_new)


The user does not have to specify all the keyword arguments.


.. _python_update_settings:

Update settings
^^^^^^^^^^^^^^^

Settings can be updated by running

.. code:: python

    m.update_settings(**kwargs)


where :code:`kwargs` are the allowed settings that can be updated marked with an * in :ref:`python_setup`.


Warm start
----------

Primal and dual variables can be warm-started with

.. code:: python

    m.warm_start(x=x0, y=y0)


where :code:`x0` and :code:`y0` are the new primal and dual variables.
