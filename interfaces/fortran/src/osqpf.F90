#include "glob_optsf.h"

MODULE OSQP

  USE iso_c_binding
  USE OSQP_types
  IMPLICIT NONE

  PRIVATE
  PUBLIC :: OSQP_settings_type, OSQP_info_type, OSQP_data_type,                &
            OSQP_solve, OSQP_settings, OSQP_cleanup

!  integer and real precisions as defined during osqp installation

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

!  ----------------------------
!  interface blocks for c calls
!  ----------------------------

!  settings procedure

  INTERFACE
    FUNCTION osqp_f2c_settings( settings, c_settings ) BIND( C )
    USE iso_c_binding
    USE OSQP_types
    TYPE ( OSQP_settings_type ) :: settings
    TYPE ( C_PTR ) :: c_settings
    END function osqp_f2c_settings
  END INTERFACE

!  solve procedure

  INTERFACE
    FUNCTION osqp_f2c_solve( m, n, P_nnz, P_val, P_row, P_ptr, A_nnz, A_val,   &
                             A_row, A_ptr, q, l, u, x, y, info,                &
                             c_settings, c_work, c_data ) BIND( C )
    USE iso_c_binding
    USE OSQP_types
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
    INTEGER ( KIND = ip ) :: osqp_interface
    INTEGER ( KIND = ip ), VALUE :: n
    INTEGER ( KIND = ip ), VALUE :: m
    INTEGER ( KIND = ip ), VALUE :: P_nnz
    INTEGER ( KIND = ip ), DIMENSION( n + 1 ) :: P_ptr
    INTEGER ( KIND = ip ), DIMENSION( P_nnz ) :: P_row
    REAL ( KIND = wp ), DIMENSION( P_nnz ) :: P_val
    INTEGER ( KIND = ip ), VALUE :: A_nnz
    INTEGER ( KIND = ip ), DIMENSION( n + 1 ) :: A_ptr
    INTEGER ( KIND = ip ), DIMENSION( A_nnz  ) :: A_row
    REAL ( KIND = wp ), DIMENSION( A_nnz ) :: A_val
    REAL ( KIND = wp ), DIMENSION( n ) :: q
    REAL ( KIND = wp ), DIMENSION( m ) :: l
    REAL ( KIND = wp ), DIMENSION( m ) :: u
    REAL ( KIND = wp ), DIMENSION( n ) :: x
    REAL ( KIND = wp ), DIMENSION( m ) :: y
    TYPE ( OSQP_info_type ) :: info
    TYPE ( C_PTR ), VALUE :: c_settings
    TYPE ( C_PTR ) :: c_work
    TYPE ( C_PTR ) :: c_data
    END function osqp_f2c_solve
  END INTERFACE

!  cleanup procedure

  INTERFACE
    FUNCTION osqp_f2c_cleanup( c_settings, c_work, c_data ) BIND( C )
    USE iso_c_binding
    TYPE ( C_PTR ), VALUE :: c_settings
    TYPE ( C_PTR ), VALUE :: c_work
    TYPE ( C_PTR ), VALUE :: c_data
    END function osqp_f2c_cleanup
  END INTERFACE

CONTAINS

!  copy settings into solver data

  SUBROUTINE OSQP_settings( settings, data )
  TYPE( OSQP_settings_type ), INTENT( IN ) :: settings
  TYPE( OSQP_data_type ), INTENT( INOUT ) :: data

!  local variables

  INTEGER ( ip ) :: status

!  copy the fortran solver settings into their C counterparts

  status = osqp_f2c_settings( settings, data%c_settings )

  RETURN
  END SUBROUTINE OSQP_settings

!  solve the given problem

  SUBROUTINE OSQP_solve( m, n, P_ptr, P_row, P_val, q, A_ptr, A_row, A_val, l, &
                         u, x, y, info, data )
  INTEGER ( KIND = ip ), INTENT( IN ) :: m
  INTEGER ( KIND = ip ), INTENT( IN ) :: n
  INTEGER ( KIND = ip ), INTENT( IN ), DIMENSION( n + 1 ) :: P_ptr
  INTEGER ( KIND = ip ), INTENT( IN ), DIMENSION( P_ptr( n + 1 ) - 1  ) :: P_row
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( P_ptr( n + 1 ) - 1  ) :: P_val
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: q
  INTEGER ( KIND = ip ), INTENT( IN ), DIMENSION( n + 1 ) :: A_ptr
  INTEGER ( KIND = ip ), INTENT( IN ), DIMENSION( A_ptr( n + 1 ) - 1  ) :: A_row
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ptr( n + 1 ) - 1  ) :: A_val
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: l
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: u
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: y
  TYPE( OSQP_info_type ), INTENT( INOUT ) :: info
  TYPE( OSQP_data_type ), INTENT( INOUT ) :: data

!  local variables

  INTEGER ( ip ) :: status, P_nnz, A_nnz
  P_nnz = P_ptr( n + 1 ) - 1
  A_nnz = A_ptr( n + 1 ) - 1

!  solve the problem provided

  status = osqp_f2c_solve( m, n, P_nnz, P_val, P_row, P_ptr, A_nnz, A_val,     &
                           A_row, A_ptr, q, l, u, x, y, info,                  &
                           data%c_settings, data%c_work, data%c_data )
  RETURN
  END SUBROUTINE OSQP_solve

!  clean up after solution

  SUBROUTINE OSQP_cleanup( data )
  TYPE( OSQP_data_type ), INTENT( INOUT ) :: data

!  local variables

  INTEGER ( ip ) :: status

!  free outstanding pointers

  status = osqp_f2c_cleanup( data%c_settings, data%c_work, data%c_data )
  RETURN

  END SUBROUTINE OSQP_cleanup

END MODULE OSQP
