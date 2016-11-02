#ifndef UTIL_H
#define UTIL_H

#include "constants.h"
#include "cs.h"
#include "osqp.h"

/* ================================= OTHER FUNCTIONS ======================== */
// Add timings, etc....

/* ================================= DEBUG FUNCTIONS ======================= */
#if PRINTLEVEL > 2

#define TESTS_TOL 1e-10

/* Convert sparse CSC to dense */
c_float * csc_to_dns(csc * M);

/* Copy sparse CSC matrix B = A */
void copy_csc_mat(const csc* A, csc *B);

/* Compare CSC matrices */
c_int is_eq_csc(csc *A, csc *B);

/* Print a sparse matrix */
void print_csc_matrix(csc* M, char * name);

/* Print a dense matrix */
void print_dns_matrix(c_float * M, c_int m, c_int n, char *name);

/* Print vector  */
void print_vec(c_float * V, c_int n, char *name);

#endif



#endif
