// Utilities for testing

/* create structure to hold problem data */
/* similar to OSQP internal container, but */
/* holding only bare array types and csc */
typedef struct {
c_int    n;
c_int    m;
csc     *P;
c_float *q;
csc     *A;
c_float *l;
c_float *u;
} OSQPTestData;
