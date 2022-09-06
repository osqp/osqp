#include "no_active_set_data.h"


/* function to generate QP problem data */
OSQPTestData * generate_problem_no_active_set(){

OSQPTestData * data = (OSQPTestData *)c_malloc(sizeof(OSQPTestData));

// Problem dimensions
data->n = 2;
data->m = 3;

// Problem vectors
data->l = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->l[0] = -1.00000000000000000000;
data->l[1] = -0.50000000000000000000;
data->l[2] = -0.25000000000000000000;

data->u = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->u[0] = 1.00000000000000000000;
data->u[1] = 0.69999999999999995559;
data->u[2] = 0.69999999999999995559;

data->q = (OSQPFloat*) c_malloc(2 * sizeof(OSQPFloat));
data->q[0] = 0.00000000000000000000;
data->q[1] = 0.00000000000000000000;



// Matrix A
//---------
data->A = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->A->m = 3;
data->A->n = 2;
data->A->nz = -1;
data->A->nzmax = 4;
data->A->x = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
data->A->x[0] = 1.00000000000000000000;
data->A->x[1] = 1.00000000000000000000;
data->A->x[2] = 1.00000000000000000000;
data->A->x[3] = 1.00000000000000000000;
data->A->i = (OSQPInt*) c_malloc(4 * sizeof(OSQPInt));
data->A->i[0] = 0;
data->A->i[1] = 1;
data->A->i[2] = 0;
data->A->i[3] = 2;
data->A->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->A->p[0] = 0;
data->A->p[1] = 2;
data->A->p[2] = 4;


// Matrix P
//---------
data->P = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->P->m = 2;
data->P->n = 2;
data->P->nz = -1;
data->P->nzmax = 2;
data->P->x = (OSQPFloat*) c_malloc(2 * sizeof(OSQPFloat));
data->P->x[0] = 11.00000000000000000000;
data->P->x[1] = 5.00000000000000000000;
data->P->i = (OSQPInt*) c_malloc(2 * sizeof(OSQPInt));
data->P->i[0] = 0;
data->P->i[1] = 1;
data->P->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->P->p[0] = 0;
data->P->p[1] = 1;
data->P->p[2] = 2;

return data;

}

/* function to clean problem data structure */
void clean_problem_no_active_set(OSQPTestData * data){

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
no_active_set_sols_data *  generate_problem_no_active_set_sols_data(){

no_active_set_sols_data * data = (no_active_set_sols_data *)c_malloc(sizeof(no_active_set_sols_data));

data->x_test = (OSQPFloat*) c_malloc(2 * sizeof(OSQPFloat));
data->x_test[0] = 0.00000000000000000000;
data->x_test[1] = 0.00000000000000000000;

data->y_test = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->y_test[0] = 0.00000000000000000000;
data->y_test[1] = 0.00000000000000000000;
data->y_test[2] = 0.00000000000000000000;

data->obj_value_test = 0.00000000000000000000;
data->status_test = OSQP_SOLVED;

return data;

}

/* function to clean solutions and additional data struct */
void clean_problem_no_active_set_sols_data(no_active_set_sols_data * data){

c_free(data->x_test);
c_free(data->y_test);

c_free(data);

}

