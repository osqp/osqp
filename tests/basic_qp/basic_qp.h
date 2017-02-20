#ifndef BASIC_QP_H
#define BASIC_QP_H
#include "osqp.h"


/* function to generate problem data structure */
Data * generate_problem_basic_qp(){

Data * data = (Data *)c_malloc(sizeof(Data));

// Problem dimensions
data->n = 2;
data->m = 3;

// Problem vectors
data->l = c_malloc(3 * sizeof(c_float));
data->l[0] = 1.00000000000000000000;
data->l[1] = 0.00000000000000000000;
data->l[2] = 0.00000000000000000000;
data->u = c_malloc(3 * sizeof(c_float));
data->u[0] = 1.00000000000000000000;
data->u[1] = 0.69999999999999995559;
data->u[2] = 0.69999999999999995559;
data->q = c_malloc(2 * sizeof(c_float));
data->q[0] = 1.00000000000000000000;
data->q[1] = 1.00000000000000000000;

// Matrix A
//---------
data->A = c_malloc(sizeof(csc));
data->A->m = 3;
data->A->n = 2;
data->A->nz = -1;
data->A->nzmax = 4;
data->A->x = c_malloc(4 * sizeof(c_float));
data->A->x[0] = 1.00000000000000000000;
data->A->x[1] = 1.00000000000000000000;
data->A->x[2] = 1.00000000000000000000;
data->A->x[3] = 1.00000000000000000000;
data->A->i = c_malloc(4 * sizeof(c_int));
data->A->i[0] = 0;
data->A->i[1] = 1;
data->A->i[2] = 0;
data->A->i[3] = 2;
data->A->p = c_malloc((2 + 1) * sizeof(c_int));
data->A->p[0] = 0;
data->A->p[1] = 2;
data->A->p[2] = 4;

// Matrix P
//---------
data->P = c_malloc(sizeof(csc));
data->P->m = 2;
data->P->n = 2;
data->P->nz = -1;
data->P->nzmax = 4;
data->P->x = c_malloc(4 * sizeof(c_float));
data->P->x[0] = 4.00000000000000000000;
data->P->x[1] = 1.00000000000000000000;
data->P->x[2] = 1.00000000000000000000;
data->P->x[3] = 2.00000000000000000000;
data->P->i = c_malloc(4 * sizeof(c_int));
data->P->i[0] = 0;
data->P->i[1] = 1;
data->P->i[2] = 0;
data->P->i[3] = 1;
data->P->p = c_malloc((2 + 1) * sizeof(c_int));
data->P->p[0] = 0;
data->P->p[1] = 2;
data->P->p[2] = 4;

return data;

}

/* function to clean problem data structure */
c_int clean_problem_basic_qp(Data * data){

// Clean vectors
c_free(data->l);
c_free(data->u);
c_free(data->q);

//Clean Matrices
c_free(data->A->x);
c_free(data->A->i);
c_free(data->A->p);
c_free(data->A);
c_free(data->P->x);
c_free(data->P->i);
c_free(data->P->p);
c_free(data->P);

c_free(data);
return 0;

}

#endif
