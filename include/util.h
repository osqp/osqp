#ifndef UTIL_H
#define UTIL_H

#include "constants.h"
#include "cs.h"

/* ================================= DEBUG FUNCTIONS ======================= */
#if PRINTLEVEL > 2

/* Convert sparse CSC to dense */
c_float * csc_to_dns(csc * M);

/* Print a sparse matrix */
void print_csc_matrix(csc* M, char * name);

/* Print a dense matrix */
void print_dns_matrix(c_float * M, c_int m, c_int n, char *name);

/* Print vector  */
void print_vec(c_float * V, c_int n, char *name);

#endif

/* ================================= OTHER FUNCTIONS ======================== */
// Add timings, etc....

#endif
