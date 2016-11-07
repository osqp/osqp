#ifndef UTIL_H
#define UTIL_H

#include "constants.h"
#include "cs.h"
#include "osqp.h"


/* ================================= OTHER FUNCTIONS ======================== */
/* Set default settings from constants.h file */
/* assumes d->stgs already allocated memory */
void set_default_settings(Settings * settings);

/* Copy settings creating a new settings structure */
Settings * copy_settings(Settings * settings);

/* Print Header before running the algorithm */
void print_setup_header(const Data *data, const Settings *settings);

/* Print Header with data to be displayed per iteration */
void print_header();


/* ================================= DEBUG FUNCTIONS ======================= */
#if PRINTLEVEL > 2
#include <stdio.h>


#define TESTS_TOL 1e-10  // Define tests tolerance

/* Convert sparse CSC to dense */
c_float * csc_to_dns(csc * M);

/* Compare CSC matrices */
c_int is_eq_csc(csc *A, csc *B);

/* Print a csc sparse matrix */
void print_csc_matrix(csc* M, char * name);

/* Print a triplet format sparse matrix */
void print_trip_matrix(csc* M, char * name);

/* Print a dense matrix */
void print_dns_matrix(c_float * M, c_int m, c_int n, char *name);

/* Print vector  */
void print_vec(c_float * V, c_int n, char *name);

#endif



#endif
