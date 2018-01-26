#include "glob_optsf.h"

MODULE OSQP_TYPES

  USE iso_c_binding
  IMPLICIT NONE

  PRIVATE

!  integer and real precisions

#ifdef DLONG
    INTEGER, PARAMETER :: ip = c_long_long
#else
    INTEGER, PARAMETER :: ip = c_int
#endif

#ifdef DFLOAT
    INTEGER, PARAMETER :: wp = c_float
#else
    INTEGER, PARAMETER :: wp = c_double
#endif

!  parameters

    REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
    REAL ( KIND = wp ), PARAMETER :: biginf = HUGE( 1.0_wp )

!  --------------------------
!  OSQP_settings derived type
!  --------------------------

  TYPE, BIND( C ), PUBLIC :: OSQP_settings_type

!  ADMM step rho

    REAL ( KIND = wp ) :: rho = ten ** ( - 1 )

!  ADMM step sigma

    REAL ( KIND = wp ) :: sigma = ten ** ( - 6 )

!  heuristic data scaling iterations. If 0, scaling disabled

    INTEGER ( KIND = ip ) :: scaling = 10

#if EMBEDDED != 1

!  boolean, is rho step size adaptive?

    INTEGER ( KIND = ip ) :: adaptive_rho = 1

!  Number of iterations between rho adaptations rho. If 0, it is automatic

    INTEGER ( KIND = ip ) :: adaptive_rho_interval = 0

!  Tolerance X for adapting rho. The new rho has to be X times larger or 1/X
!  times smaller than the current one to trigger a new factorization.

    REAL ( KIND = wp ) :: adaptive_rho_tolerance = 5.0_wp

#ifdef PROFILING

!  Interval for adapting rho (fraction of the setup time)

    REAL ( KIND = wp ) :: adaptive_rho_fraction = 0.4_wp

#endif

#endif

!  maximum iterations

    INTEGER ( KIND = ip ) :: max_iter = 4000

!  absolute convergence tolerance

    REAL ( KIND = wp ) :: eps_abs = ten ** ( - 3 )

!  relative convergence tolerance

    REAL ( KIND = wp ) :: eps_rel = ten ** ( - 3 )

!  primal infeasibility tolerance

    REAL ( KIND = wp ) :: eps_prim_inf = ten ** ( - 4 )

!  dual infeasibility tolerance

    REAL ( KIND = wp ) :: eps_dual_inf = ten ** ( - 4 )

!  relaxation parameter

    REAL ( KIND = wp ) :: alpha = 1.6_c_double

!  linear system solver to use

!!   enum linsys_solver_type linsys_solver

#ifndef EMBEDDED

!  regularization parameter for polish

    REAL ( KIND = wp ) :: delta = ten ** ( - 6 )

!  boolean, polish ADMM solution

    INTEGER ( KIND = ip ) :: polish = 0

!  iterative refinement steps in polish

    INTEGER ( KIND = ip ) :: polish_refine_iter = 3

!  boolean, write out progres

    INTEGER ( KIND = ip ) :: verbose = 1

#endif

!  boolean, use scaled termination criteria

    INTEGER ( KIND = ip ) :: scaled_termination = 0

!  integer, check termination interval. If 0, termination checking is disabled

    INTEGER ( KIND = ip ) :: check_termination = 25

!  boolean, warm start

    INTEGER ( KIND = ip ) :: warm_start = 1

  END TYPE OSQP_settings_type

!  ------------------------
!  OSQP_inform derived type
!  ------------------------

  TYPE, BIND( C ), PUBLIC :: OSQP_info_type

!  number of iterations taken

    INTEGER ( KIND = ip ) :: iter = - 1

!  status string, e.g. 'solved'

    CHARACTER ( KIND = c_char, LEN = 32 ) :: status = REPEAT( ' ', 32 )

!  status as c_int, defined in constants.h

    INTEGER ( KIND = ip ) :: status_val = - 10

#ifndef EMBEDDED

!  polish status: successful (1), unperformed (0), (-1) unsuccessful

    INTEGER ( KIND = ip ) :: status_polish = 0

#endif

!  primal objective

    REAL ( KIND = wp ) :: obj_val = biginf

!  norm of primal residual

    REAL ( KIND = wp ) :: pri_res = biginf

!  norm of dual residual

    REAL ( KIND = wp ) :: dua_res = biginf

#ifdef PROFILING

!  time taken for setup phase (seconds)

    REAL ( KIND = wp ) :: setup_time = 0.0_wp

!  time taken for solve phase (seconds)

    REAL ( KIND = wp ) :: solve_time = 0.0_wp

!  time taken for polish phase (seconds)

    REAL ( KIND = wp ) :: polish_time = 0.0_wp

!  total time  (seconds)

    REAL ( KIND = wp ) :: run_time = 0.0_wp

#endif

#if EMBEDDED != 1

!  number of rho updates

    INTEGER ( KIND = ip ) :: rho_updates = 0

!  best rho estimate so far from residuals

    REAL ( KIND = wp ) :: rho_estimate = biginf

#endif

  END TYPE OSQP_info_type

!  ----------------------
!  OSQP_data derived type
!  ----------------------

  TYPE, BIND( C ), PUBLIC :: OSQP_data_type

!  internal structures

    TYPE ( c_ptr ) :: c_settings
    TYPE ( c_ptr ) :: c_work
    TYPE ( c_ptr ) :: c_data

  END TYPE OSQP_data_type

END MODULE OSQP_TYPES
