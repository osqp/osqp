.. _c_interface:

C
=====


.. _C_main_API:

Main solver API
---------------

Main solver functions
^^^^^^^^^^^^^^^^^^^^^

The main C API is imported from the header :code:`osqp.h` and provides the following functions


.. doxygenfunction:: osqp_setup

.. doxygenfunction:: osqp_solve

.. doxygenfunction:: osqp_cleanup


Main solver data types
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: OSQPSolver
  :members:

.. doxygenstruct:: OSQPSolution
   :members:

.. doxygenstruct:: OSQPInfo
   :members:


Warm start
----------
OSQP automatically warm starts primal and dual variables from the previous QP solution. If you would like to warm start their values manually, you can use

.. doxygenfunction:: osqp_warm_start


.. _C_update_data :

Update problem data
-------------------
Problem data can be updated without executing the setup again using the following functions.

.. doxygenfunction:: osqp_update_data_vec

.. doxygenfunction:: osqp_update_data_mat


.. _C_settings :

Solver settings
---------------

Settings API
^^^^^^^^^^^^

.. doxygenfunction:: osqp_set_default_settings


Many solver settings can be updated without running setup again.

.. doxygenfunction:: osqp_update_settings

.. doxygenfunction:: osqp_update_rho


Settings structure
^^^^^^^^^^^^^^^^^^

The setting structure has the following fields.


.. doxygenstruct:: OSQPSettings
  :members:


.. _C_derivatives :

Compute solution derivatives
----------------------------
Adjoint derivatives of the QP problem can be computed at the current solution.

.. doxygenfunction:: osqp_adjoint_derivative_compute

.. doxygenfunction:: osqp_adjoint_derivative_get_mat

.. doxygenfunction:: osqp_adjoint_derivative_get_vec


.. _C_code_generation :

Code generation
---------------
The QP problem and all solver data can be written to a problem workspace for use by OSQP in embedded mode.

.. doxygenfunction:: osqp_set_default_codegen_defines

.. doxygenfunction:: osqp_codegen

.. doxygenstruct:: OSQPCodegenDefines
   :members:


.. _C_data_types :

Data types
----------

The most basic used datatypes are

* :code:`OSQPInt`: can be :code:`long` or :code:`int` if the compiler flag :code:`OSQP_USE_LONG` is set or not
* :code:`OSQPFloat`: can be a :code:`float` or a :code:`double` if the compiler flag :code:`OSQP_USE_FLOAT` is set or not.


The matrices are defined in `Compressed Sparse Column (CSC) format <https://people.sc.fsu.edu/~jburkardt/data/cc/cc.html>`_ using zero-based indexing.

.. doxygenstruct:: OSQPCscMatrix
   :members:



.. TODO: Add sublevel API
.. TODO: Add using your own linear system solver
