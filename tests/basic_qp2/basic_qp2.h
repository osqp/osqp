#ifndef BASIC_QP2_DATA_H
#define BASIC_QP2_DATA_H
#include "osqp.h"


/* create additional data and solutions structure */
typedef struct {
c_int status_test;
c_int status_test_new;
c_float obj_value_test;
c_float * y_test_new;
c_float * q_new;
c_float * x_test;
c_float obj_value_test_new;
c_float * x_test_new;
c_float * u_new;
c_float * y_test;
} basic_qp2_sols_data;

/* function to generate QP problem data */
Data * generate_problem_basic_qp2(){

Data * data = (Data *)c_malloc(sizeof(Data));

// Problem dimensions
data->n = 2;
data->m = 5;

// Problem vectors
data->l = c_malloc(5 * sizeof(c_float));
data->l[0] = -OSQP_INFTY;
data->l[1] = -OSQP_INFTY;
data->l[2] = -OSQP_INFTY;
data->l[3] = -OSQP_INFTY;
data->l[4] = -OSQP_INFTY;
data->u = c_malloc(5 * sizeof(c_float));
data->u[0] = 0.00000000000000000000;
data->u[1] = 0.00000000000000000000;
data->u[2] = -15.00000000000000000000;
data->u[3] = 100.00000000000000000000;
data->u[4] = 80.00000000000000000000;
data->q = c_malloc(2 * sizeof(c_float));
data->q[0] = 3.00000000000000000000;
data->q[1] = 4.00000000000000000000;

// Matrix A
//---------
data->A = c_malloc(sizeof(csc));
data->A->m = 5;
data->A->n = 2;
data->A->nz = -1;
data->A->nzmax = 8;
data->A->x = c_malloc(8 * sizeof(c_float));
data->A->x[0] = -1.00000000000000000000;
data->A->x[1] = -1.00000000000000000000;
data->A->x[2] = 2.00000000000000000000;
data->A->x[3] = 3.00000000000000000000;
data->A->x[4] = -1.00000000000000000000;
data->A->x[5] = 3.00000000000000000000;
data->A->x[6] = 5.00000000000000000000;
data->A->x[7] = 4.00000000000000000000;
data->A->i = c_malloc(8 * sizeof(c_int));
data->A->i[0] = 0;
data->A->i[1] = 2;
data->A->i[2] = 3;
data->A->i[3] = 4;
data->A->i[4] = 1;
data->A->i[5] = 2;
data->A->i[6] = 3;
data->A->i[7] = 4;
data->A->p = c_malloc((2 + 1) * sizeof(c_int));
data->A->p[0] = 0;
data->A->p[1] = 4;
data->A->p[2] = 8;

// Matrix P
//---------
data->P = c_malloc(sizeof(csc));
data->P->m = 2;
data->P->n = 2;
data->P->nz = -1;
data->P->nzmax = 1;
data->P->x = c_malloc(1 * sizeof(c_float));
data->P->x[0] = 11.00000000000000000000;
data->P->i = c_malloc(1 * sizeof(c_int));
data->P->i[0] = 0;
data->P->p = c_malloc((2 + 1) * sizeof(c_int));
data->P->p[0] = 0;
data->P->p[1] = 1;
data->P->p[2] = 1;

return data;

}

/* function to clean problem data structure */
void clean_problem_basic_qp2(Data * data){

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
basic_qp2_sols_data *  generate_problem_basic_qp2_sols_data(){

basic_qp2_sols_data * data = (basic_qp2_sols_data *)c_malloc(sizeof(basic_qp2_sols_data));

data->status_test = 1;
data->status_test_new = 1;
data->obj_value_test = 1282.49996453762651071884;
data->y_test_new = c_malloc(5 * sizeof(c_float));
data->y_test_new[0] = 0.00000023554710108178;
data->y_test_new[1] = 664.00627950372688701464;
data->y_test_new[2] = 221.00209276760267584905;
data->y_test_new[3] = 0.00000007052751803276;
data->y_test_new[4] = 0.00000021334766013396;
data->q_new = c_malloc(2 * sizeof(c_float));
data->q_new[0] = 1.00000000000000000000;
data->q_new[1] = 1.00000000000000000000;
data->x_test = c_malloc(2 * sizeof(c_float));
data->x_test[0] = 14.99999979004714845132;
data->x_test[1] = -0.00000004757366358703;
data->obj_value_test_new = 2219.99999821535357114044;
data->x_test_new = c_malloc(2 * sizeof(c_float));
data->x_test_new[0] = 19.99999999193508770645;
data->x_test_new[1] = -0.00000000230088233073;
data->u_new = c_malloc(5 * sizeof(c_float));
data->u_new[0] = -2.00000000000000000000;
data->u_new[1] = 0.00000000000000000000;
data->u_new[2] = -20.00000000000000000000;
data->u_new[3] = 100.00000000000000000000;
data->u_new[4] = 80.00000000000000000000;
data->y_test = c_malloc(5 * sizeof(c_float));
data->y_test[0] = 0.00000214870518246836;
data->y_test[1] = 508.03014794000051779221;
data->y_test[2] = 168.01004735181336968708;
data->y_test[3] = 0.00000046063264419062;
data->y_test[4] = 0.00000092937866875409;

return data;

}

/* function to clean solutions and additional data struct */
void clean_problem_basic_qp2_sols_data(basic_qp2_sols_data * data){

c_free(data->y_test_new);
c_free(data->q_new);
c_free(data->x_test);
c_free(data->x_test_new);
c_free(data->u_new);
c_free(data->y_test);

c_free(data);

}

#endif
