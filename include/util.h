#ifndef UTIL_H
#define UTIL_H

# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */

# include "osqp.h"
# include "types.h"


/**********************
* Utility Functions  *
**********************/

# ifndef EMBEDDED

/**
 * Copy settings creating a new settings structure (uses MALLOC)
 * @param  settings Settings to be copied
 * @return          New settings structure
 */
OSQPSettings* copy_settings(const OSQPSettings *settings);

# endif /* ifndef EMBEDDED */

/**
 * Custom string copy to avoid string.h library
 * @param dest   destination string
 * @param source source string
 */
void c_strcpy(char       dest[],
              const char source[]);


# ifdef PRINTING

/**
 * Print Header before running the algorithm
 * @param solver     osqp solver
 */
void print_setup_header(const OSQPSolver *solver);

/**
 * Print header with data to be displayed per iteration
 */
void print_header(void);

/**
 * Print iteration summary
 * @param solver osqp solver
 */
void print_summary(OSQPSolver *solver);

/**
 * Print information after polish
 * @param solver osqp solver
 */
void print_polish(OSQPSolver *solver);

/**
 * Print footer when algorithm terminates
 * @param info      info structure
 * @param polishing is polishing enabled?
 */
void print_footer(OSQPInfo *info,
                  c_int     polishing);


# endif /* ifdef PRINTING */


/*********************************
* Timer Structs and Functions * *
*********************************/

/*! \cond PRIVATE */

# ifdef PROFILING

// Windows
#  ifdef IS_WINDOWS

  // Some R packages clash with elements
  // of the windows.h header, so use a
  // slimmer version for conflict avoidance
# ifdef R_LANG
#define NOGDI
# endif

#   include <windows.h>

struct OSQP_TIMER {
  LARGE_INTEGER tic;
  LARGE_INTEGER toc;
  LARGE_INTEGER freq;
};

// Mac
#  elif defined IS_MAC

#   include <mach/mach_time.h>

/* Use MAC OSX  mach_time for timing */
struct OSQP_TIMER {
  uint64_t                  tic;
  uint64_t                  toc;
  mach_timebase_info_data_t tinfo;
};

// Linux
#  else /* ifdef IS_WINDOWS */

/* Use POSIX clock_gettime() for timing on non-Windows machines */
#   include <time.h>
#   include <sys/time.h>


struct OSQP_TIMER {
  struct timespec tic;
  struct timespec toc;
};

#  endif /* ifdef IS_WINDOWS */

/*! \endcond */

/**
 * Timer Methods
 */

/**
 * Start timer
 * @param t Timer object
 */
void    osqp_tic(OSQPTimer *t);

/**
 * Report time
 * @param  t Timer object
 * @return   Reported time
 */
c_float osqp_toc(OSQPTimer *t);

# endif /* END #ifdef PROFILING */


/* ================================= DEBUG FUNCTIONS ======================= */

/*! \cond PRIVATE */


# ifdef PRINTING
#  include <stdio.h>


/* Print a csc sparse matrix */
void print_csc_matrix(const csc  *M,
                      const char *name);

/* Dump csc sparse matrix to file */
void dump_csc_matrix(const csc  *M,
                     const char *file_name);

/* Print a triplet format sparse matrix */
void print_trip_matrix(const csc  *M,
                       const char *name);

/* Print a dense matrix */
void print_dns_matrix(const c_float *M,
                      c_int          m,
                      c_int          n,
                      const char    *name);

/* Print vector  */
void print_vec(const c_float *v,
               c_int          n,
               const char    *name);

/* Dump vector to file */
void dump_vec(const c_float *v,
              c_int          len,
              const char    *file_name);

// Print int array
void print_vec_int(const c_int *x,
                   c_int        n,
                   const char  *name);

# endif /* ifdef PRINTING */

/*! \endcond */


# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef UTIL_H */
