.. _solver_settings :

Solver settings
---------------

The solver settings are displayed in the following table. Not that the boolean values :code:`True/False` are defined as :code:`1/0` in the C and Matlab interfaces.


The settings marked with * can be changed without running the setup method again.


+------------------------------------+-------------------------------------+----------------+
| Argument                           | Description                         | Default value  |
+====================================+=====================================+================+
| :code:`scaling`                    | Perform data scaling                |   True         |
+------------------------------------+-------------------------------------+----------------+
| :code:`rho`                        | ADMM rho step                       | Auto computed  |
+------------------------------------+-------------------------------------+----------------+
| :code:`auto_rho`                   | ADMM rho step automatic selection   |   True         |
+------------------------------------+-------------------------------------+----------------+
| :code:`sigma`                      | ADMM sigma step                     |   1e-06        |
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
| :code:`delta`    *                 | Polishing regularization parameter  |   1e-06        |
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
| :code:`scaling_iter`               | Scaling iterations                  |   15            |
+------------------------------------+-------------------------------------+----------------+
| :code:`pol_refine_iter` *          | Refinement iterations in polish     |   5            |
+------------------------------------+-------------------------------------+----------------+
