#ifndef UTIL_H
#define UTIL_H


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


# ifdef OSQP_ENABLE_PRINTING

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


# endif /* ifdef OSQP_ENABLE_PRINTING */


/* ================================= DEBUG FUNCTIONS ======================= */

/*! \cond PRIVATE */


#if defined(DEBUG) && defined(OSQP_ENABLE_PRINTING)

#  include <stdio.h>

/* Print a csc sparse matrix */
void print_csc_matrix(const OSQPCscMatrix* M,
                      const char*          name);

/* Dump csc sparse matrix to file */
void dump_csc_matrix(const OSQPCscMatrix* M,
                     const char*          file_name);

/* Print a triplet format sparse matrix */
void print_trip_matrix(const OSQPCscMatrix* M,
                       const char*          name);

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

# endif /* #if defined(DEBUG) && defined(OSQP_ENABLE_PRINTING) */

/*! \endcond */


#endif /* ifndef UTIL_H */
