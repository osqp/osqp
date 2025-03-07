.. _c_interface:

C
=====

.. contents:: Table of Contents
   :depth: 3
   :local:


.. _c_main_api:

Main solver API
---------------

The main C API is imported from the header :code:`osqp.h`. It is divided into the following headers:
* Constants in osqp_api_constants.h
* Functions in osqp_api_functions.h
* Types in osqp_api_types.h


Main solver data types
^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: OSQPSolver
   :members:

.. doxygenstruct:: OSQPInfo
   :members:

.. doxygenstruct:: OSQPSolution
   :members:

.. doxygenstruct:: OSQPCodegenDefines
   :members:

Solver capabilities
"""""""""""""""""""

.. doxygenenum:: osqp_capabilities_type

Solver status
"""""""""""""

.. doxygenenum:: osqp_status_type

Polish status
"""""""""""""

.. doxygenenum:: osqp_polish_status_type

Linear system solvers
""""""""""""""""""""""

.. doxygenenum:: osqp_linsys_solver_type

Preconditioners for CG method
"""""""""""""""""""""""""""""

.. doxygenenum:: osqp_precond_type

Solver errors
"""""""""""""

.. doxygenenum:: osqp_error_type

Solver parameters and settings
""""""""""""""""""""""""""""""

.. doxygendefine:: OSQP_VERBOSE

.. doxygendefine:: OSQP_WARM_STARTING

.. doxygendefine:: OSQP_SCALING
   
.. doxygendefine:: OSQP_POLISHING

ADMM parameters:

.. doxygendefine:: OSQP_RHO

.. doxygendefine:: OSQP_SIGMA

.. doxygendefine:: OSQP_ALPHA

.. doxygendefine:: OSQP_RHO_MIN

.. doxygendefine:: OSQP_RHO_MAX

.. doxygendefine:: OSQP_RHO_TOL

.. doxygendefine:: OSQP_RHO_EQ_OVER_RHO_INEQ

.. doxygendefine:: OSQP_RHO_IS_VEC

CG parameters:

.. doxygendefine:: OSQP_CG_MAX_ITER

.. doxygendefine:: OSQP_CG_TOL_REDUCTION

.. doxygendefine:: OSQP_CG_TOL_FRACTION

Adaptive rho logic
.. doxygendefine:: OSQP_ADAPTIVE_RHO

.. doxygendefine:: OSQP_ADAPTIVE_RHO_INTERVAL
   
.. doxygendefine:: OSQP_ADAPTIVE_RHO_TOLERANCE

.. doxygendefine:: OSQP_ADAPTIVE_RHO_FRACTION
   
.. doxygendefine:: OSQP_ADAPTIVE_RHO_MULTIPLE_TERMINATION
   
.. doxygendefine:: OSQP_ADAPTIVE_RHO_FIXED

Termination parameters
.. doxygendefine:: OSQP_MAX_ITER
   
.. doxygendefine:: OSQP_EPS_ABS
   
.. doxygendefine:: OSQP_EPS_REL
   
.. doxygendefine:: OSQP_EPS_PRIM_INF
   
.. doxygendefine:: OSQP_EPS_DUAL_INF
   
.. doxygendefine:: OSQP_SCALED_TERMINATION
   
.. doxygendefine:: OSQP_TIME_LIMIT

.. doxygendefine:: OSQP_CHECK_TERMINATION

.. doxygendefine:: OSQP_DELTA
   
.. doxygendefine:: OSQP_POLISH_REFINE_ITER

Hard-coded values and settings:

.. doxygendefine:: OSQP_NULL

.. doxygendefine:: OSQP_NAN

.. doxygendefine:: OSQP_INFTY

.. doxygendefine:: OSQP_DIVISION_TOL

.. doxygendefine:: OSQP_PRINT_INTERVAL

.. doxygendefine:: OSQP_MIN_SCALING
   
.. doxygendefine:: OSQP_MAX_SCALING

.. doxygendefine:: OSQP_CG_TOL_MIN
   
.. doxygendefine:: OSQP_CG_POLISH_TOL

Main solver API
"""""""""""""""

.. doxygenfunction:: osqp_capabilities

.. doxygenfunction:: osqp_version

.. doxygenfunction:: osqp_error_message

.. doxygenfunction:: osqp_get_dimensions

.. doxygenfunction:: osqp_setup

.. doxygenfunction:: osqp_solve

.. doxygenfunction:: osqp_get_solution

.. doxygenfunction:: osqp_cleanup

Sublevel API
""""""""""""
These functions can be called without performing setup again.

.. doxygenfunction:: osqp_warm_start

.. doxygenfunction:: osqp_cold_start

.. doxygenfunction:: osqp_set_default_settings

.. doxygenfunction:: osqp_update_settings

.. doxygenfunction:: osqp_update_rho

.. _c_update_data:

.. doxygenfunction:: osqp_update_data_vec

.. doxygenfunction:: osqp_update_data_mat

Settings structure
^^^^^^^^^^^^^^^^^^

The setting structure has the following fields.

.. doxygenstruct:: OSQPSettings
  :members:

The Settings structure can be created and freed using the following functions.

.. doxygenfunction:: OSQPSettings_new

.. doxygenfunction:: OSQPSettings_free

.. _c_derivatives :

Compute solution derivatives
----------------------------
Adjoint derivatives of the QP problem can be computed at the current solution.

.. doxygenfunction:: osqp_adjoint_derivative_compute

.. doxygenfunction:: osqp_adjoint_derivative_get_mat

.. doxygenfunction:: osqp_adjoint_derivative_get_vec

Code generation functions
"""""""""""""""""""""""""

.. doxygenfunction:: osqp_codegen

.. doxygenfunction:: osqp_set_default_codegen_defines

.. doxygenfunction:: OSQPCodegenDefines_new

.. doxygenfunction:: OSQPCodegenDefines_free

Data types
----------

Primitive types
^^^^^^^^^^^^^^^

The most basic used datatypes are

.. doxygentypedef:: OSQPInt

.. doxygentypedef:: OSQPFloat

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