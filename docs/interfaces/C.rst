.. _c_interface:

C
=====


.. _C_main_API:

Main solver API
---------------

The main C API is imported from the header :code:`osqp.h`. It is divided into the following headers:
* Constants in osqp_api_constants.h
* Functions in osqp_api_functions.h
* Types in osqp_api_types.h


Main solver constants
^^^^^^^^^^^^^^^^^^^^^

Solver capabilities
"""""""""""""""""""

.. doxygenenum:: osqp_capabilities_type


Solver status
"""""""""""""

.. doxygenenum:: osqp_status_type

.. doxygendefine:: OSQP_STATUS_MESSAGE

Polish status
"""""""""""""

.. doxygenenum:: osqp_polish_status_type

Linear system solvers
""""""""""""""""""""""

.. doxygenenum:: osqp_linsys_solver_type

Preconditioners for CG method
"""""""""""""""""""""""""""""

.. doxygentypedef:: osqp_precond_type

Solver errors
"""""""""""""

.. doxygenenum:: osqp_error_type

.. doxygendefine:: OSQP_ERROR_MESSAGE


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

Main solver functions
^^^^^^^^^^^^^^^^^^^^^

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

.. doxygenfunction:: osqp_update_data_vec

.. doxygenfunction:: osqp_update_data_mat

.. doxygenfunction:: osqp_set_default_settings

.. doxygenfunction:: osqp_update_settings

.. doxygenfunction:: osqp_update_rho


Derivative functions
""""""""""""""""""""

.. doxygenfunction:: osqp_adjoint_derivative_compute

.. doxygenfunction:: osqp_adjoint_derivative_get_mat

.. doxygenfunction:: osqp_adjoint_derivative_get_vec

Code generation functions
"""""""""""""""""""""""""

.. doxygenfunction:: osqp_set_default_codegen_defines

.. doxygenfunction:: osqp_codegen


Main solver types
^^^^^^^^^^^^^^^^^

.. doxygentypedef:: OSQPInt

.. doxygentypedef:: OSQPFloat

.. doxygenstruct:: OSQPCscMatrix
   :members:

.. doxygenstruct:: OSQPSettings
   :members:

.. doxygenstruct:: OSQPInfo
   :members:

.. doxygenstruct:: OSQPSolution   
   :members:

.. doxygenstruct:: OSQPWorkspace
   :members:

.. doxygenstruct:: OSQPSolver
   :members:

.. doxygenstruct:: OSQPCodegenDefines   
   :members:

Main solver utils
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: csc_set_data
