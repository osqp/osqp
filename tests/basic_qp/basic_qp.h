#ifndef BASIC_QP_H
#define BASIC_QP_H
#include "osqp.h"


/* create additional data and solutions structure */
typedef struct {
c_float * x_test;
c_float * q_new;
c_float * y_test;
c_int status_test;
c_float * u_new;
c_float obj_value_test;
c_float * l_new;
} basic_qp_sols_data;

/* function to generate QP problem data */
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
void clean_problem_basic_qp(Data * data){

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

}

/* function to define solutions and additional data struct */
basic_qp_sols_data *  generate_problem_basic_qp_sols_data(){

basic_qp_sols_data * data = (basic_qp_sols_data *)c_malloc(sizeof(basic_qp_sols_data));

data->x_test = c_malloc(2 * sizeof(c_float));
data->x_test[0] = 0.30000000110359281713;
data->x_test[1] = 0.69999999882402397233;
data->q_new = c_malloc(2 * sizeof(c_float));
data->q_new[0] = 2.50000000000000000000;
data->q_new[1] = 3.20000000000000017764;
data->y_test = c_malloc(3 * sizeof(c_float));
data->y_test[0] = -2.89998849516666235004;
data->y_test[1] = -0.00000000034767583971;
data->y_test[2] = 0.19998958767346983190;
data->status_test = 1;
data->u_new = c_malloc(3 * sizeof(c_float));
data->u_new[0] = 1.60000000000000008882;
data->u_new[1] = 1.00000000000000000000;
data->u_new[2] = 15.00000000000000000000;
data->obj_value_test = 1.88000000002528411258;
data->l_new = c_malloc(3 * sizeof(c_float));
data->l_new[0] = 0.80000000000000004441;
data->l_new[1] = -3.39999999999999991118;
data->l_new[2] = -11.00000000000000000000;

return data;

}

/* function to clean solutions and additional data struct */
void clean_problem_basic_qp_sols_data(basic_qp_sols_data * data){

c_free(data->x_test);
c_free(data->q_new);
c_free(data->y_test);
c_free(data->u_new);
c_free(data->l_new);

c_free(data);

}

#endif
