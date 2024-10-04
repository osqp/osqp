.. _c_interface:

C
=====

.. contents:: Table of Contents
   :depth: 3
   :local:


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


.. doxygenfunction:: OSQPSettings_new

.. doxygenfunction:: OSQPSettings_free

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

.. doxygenfunction:: osqp_codegen

.. doxygenfunction:: osqp_set_default_codegen_defines

.. doxygenfunction:: OSQPCodegenDefines_new

.. doxygenfunction:: OSQPCodegenDefines_free

.. doxygenstruct:: OSQPCodegenDefines
   :members:


.. _C_data_types :

Data types
----------

Primitive types
^^^^^^^^^^^^^^^

The most basic used datatypes are

* :code:`OSQPInt`: can be :code:`long` or :code:`int` if the compiler flag :code:`OSQP_USE_LONG` is set or not
* :code:`OSQPFloat`: can be a :code:`float` or a :code:`double` if the compiler flag :code:`OSQP_USE_FLOAT` is set or not.

Matrices
^^^^^^^^

The matrices are defined in `Compressed Sparse Column (CSC) format <https://people.sc.fsu.edu/~jburkardt/data/cc/cc.html>`_ using zero-based indexing, using the :c:struct:`OSQPCscMatrix` datatype.

.. doxygenstruct:: OSQPCscMatrix
   :members:

Data
""""

As a helper, OSQP provides the :c:func:`OSQPCscMatrix_set_data` function to assign existing data to an existing :c:struct:`OSQPCscMatrix`.

.. NOTE::
   When using the :c:func:`OSQPCscMatrix_set_data` function, the user is responsible for managing the memory used by the :c:var:`OSQPCscMatrix.x`, :c:var:`OSQPCscMatrix.p`
   :c:var:`OSQPCscMatrix.i` arrays.

.. doxygenfunction:: OSQPCscMatrix_set_data

Memory management
"""""""""""""""""

In non-embedded versions, the CSC matrix objects can be created with existing data, and free'd

.. doxygenfunction:: OSQPCscMatrix_new

.. doxygenfunction:: OSQPCscMatrix_free

Common matrices
"""""""""""""""

In non-embedded versions, several helper functions are provided to create common matrices in the CSC format, including a matrix of all structural zeros,
the square identity matrix, and diagonal matrices.

.. NOTE::
   The :c:var:`OSQPCscMatrix.x`, :c:var:`OSQPCscMatrix.p` and :c:var:`OSQPCscMatrix.i` arrays are managed by OSQP when using these functions,
   meaning they will automatically be freed inside :c:func:`OSQPCscMatrix_free`.

.. doxygenfunction:: OSQPCscMatrix_zeros

.. doxygenfunction:: OSQPCscMatrix_identity

The following diagonal matrix generation functions can generate tall, wide and square matrices.
They will always start in the upper-left corner and fill the main diagonal until the smallest dimension runs out,
leaving the remaining rows/columns as all 0.

.. doxygenfunction:: OSQPCscMatrix_diag_scalar

.. doxygenfunction:: OSQPCscMatrix_diag_vec



.. TODO: Add sublevel API
.. TODO: Add using your own linear system solver
