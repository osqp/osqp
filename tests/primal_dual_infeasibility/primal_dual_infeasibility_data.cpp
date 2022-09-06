#include "primal_dual_infeasibility_data.h"


/* function to define problem data */
primal_dual_infeasibility_sols_data *  generate_problem_primal_dual_infeasibility_sols_data(){

primal_dual_infeasibility_sols_data * data = (primal_dual_infeasibility_sols_data *)c_malloc(sizeof(primal_dual_infeasibility_sols_data));


// Matrix P
//---------
data->P = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->P->m = 2;
data->P->n = 2;
data->P->nz = -1;
data->P->nzmax = 1;
data->P->x = (OSQPFloat*) c_malloc(1 * sizeof(OSQPFloat));
data->P->x[0] = 1.00000000000000000000;
data->P->i = (OSQPInt*) c_malloc(1 * sizeof(OSQPInt));
data->P->i[0] = 0;
data->P->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->P->p[0] = 0;
data->P->p[1] = 1;
data->P->p[2] = 1;

data->q = (OSQPFloat*) c_malloc(2 * sizeof(OSQPFloat));
data->q[0] = 1.00000000000000000000;
data->q[1] = -1.00000000000000000000;


// Matrix A12
//-----------
data->A12 = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->A12->m = 3;
data->A12->n = 2;
data->A12->nz = -1;
data->A12->nzmax = 4;
data->A12->x = (OSQPFloat*) c_malloc(4 * sizeof(OSQPFloat));
data->A12->x[0] = 1.00000000000000000000;
data->A12->x[1] = 1.00000000000000000000;
data->A12->x[2] = 1.00000000000000000000;
data->A12->x[3] = 1.00000000000000000000;
data->A12->i = (OSQPInt*) c_malloc(4 * sizeof(OSQPInt));
data->A12->i[0] = 0;
data->A12->i[1] = 1;
data->A12->i[2] = 0;
data->A12->i[3] = 2;
data->A12->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->A12->p[0] = 0;
data->A12->p[1] = 2;
data->A12->p[2] = 4;


// Matrix A34
//-----------
data->A34 = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->A34->m = 3;
data->A34->n = 2;
data->A34->nz = -1;
data->A34->nzmax = 3;
data->A34->x = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->A34->x[0] = 1.00000000000000000000;
data->A34->x[1] = 1.00000000000000000000;
data->A34->x[2] = 1.00000000000000000000;
data->A34->i = (OSQPInt*) c_malloc(3 * sizeof(OSQPInt));
data->A34->i[0] = 0;
data->A34->i[1] = 1;
data->A34->i[2] = 2;
data->A34->p = (OSQPInt*) c_malloc((2 + 1) * sizeof(OSQPInt));
data->A34->p[0] = 0;
data->A34->p[1] = 2;
data->A34->p[2] = 3;

data->l = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->l[0] = 0.00000000000000000000;
data->l[1] = 1.00000000000000000000;
data->l[2] = 1.00000000000000000000;

data->u1 = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->u1[0] = 5.00000000000000000000;
data->u1[1] = 3.00000000000000000000;
data->u1[2] = 3.00000000000000000000;

data->u2 = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->u2[0] = 0.00000000000000000000;
data->u2[1] = 3.00000000000000000000;
data->u2[2] = 3.00000000000000000000;

data->u3 = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->u3[0] = 2.00000000000000000000;
data->u3[1] = 3.00000000000000000000;
data->u3[2] = OSQP_INFTY;

data->u4 = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->u4[0] = 0.00000000000000000000;
data->u4[1] = 3.00000000000000000000;
data->u4[2] = OSQP_INFTY;

data->x1 = (OSQPFloat*) c_malloc(2 * sizeof(OSQPFloat));
data->x1[0] = 1.00000000000000000000;
data->x1[1] = 3.00000000000000000000;

data->y1 = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->y1[0] = 0.00000000000000000000;
data->y1[1] = -2.00000000000000000000;
data->y1[2] = 1.00000000000000000000;

data->obj_value1 = -1.50000000000000000000;
data->status1 = OSQP_SOLVED;
data->status2 = OSQP_PRIMAL_INFEASIBLE;
data->status3 = OSQP_DUAL_INFEASIBLE;
data->status4 = OSQP_PRIMAL_INFEASIBLE;

return data;

}

/* function to clean data struct */
void clean_problem_primal_dual_infeasibility_sols_data(primal_dual_infeasibility_sols_data * data){

c_free(data->P->x);
c_free(data->P->i);
c_free(data->P->p);
c_free(data->P);
c_free(data->q);
c_free(data->A12->x);
c_free(data->A12->i);
c_free(data->A12->p);
c_free(data->A12);
c_free(data->A34->x);
c_free(data->A34->i);
c_free(data->A34->p);
c_free(data->A34);
c_free(data->l);
c_free(data->u1);
c_free(data->u2);
c_free(data->u3);
c_free(data->u4);
c_free(data->x1);
c_free(data->y1);

c_free(data);

}

