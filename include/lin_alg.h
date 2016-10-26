#ifndef LIN_ALG_H
#define LIN_ALG_H
#include <math.h>


// a matrix in compressed sparse column format
typedef struct CSC{
    int m, n;
    int *Ap, *Ai;
    double *Ax;
} csc;


#endif
