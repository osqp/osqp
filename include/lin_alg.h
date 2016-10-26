#ifndef LIN_ALG_H
#define LIN_ALG_H
#define EPS 1e-4

#include <math.h>


// a matrix in compressed sparse column format
typedef struct CSC{
    int m, n;
    int *Ap, *Ai;
    double *Ax;
} csc;


#endif
