/* NB: this is a subset of the routines in the CSPARSE package by
 Tim Davis et. al., for the full package please visit
 http://www.cise.ufl.edu/research/sparse/CSparse/ */

#ifndef CS_H
#define CS_H

#include <stdlib.h>


typedef struct cs_sparse    /* matrix in compressed-column or triplet form */
{
    int nzmax ;	    /* maximum number of entries */
    int m ;	        /* number of rows */
    int n ;	        /* number of columns */
    int *p ;	    /* column pointers (size n+1) or col indices (size nzmax) */
    int *i ;	    /* row indices, size nzmax */
    double *x ;	    /* numerical values, size nzmax */
    int nz ;	    /* # of entries in triplet matrix, -1 for compressed-col */
} cs ;



#endif
