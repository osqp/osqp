#ifndef SOLVE_LINSYS_DATA_H
#define SOLVE_LINSYS_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
csc * test_solve_KKT_Pu;
c_int test_solve_KKT_m;
csc * test_solve_KKT_KKT;
c_float test_solve_KKT_rho;
c_float test_solve_KKT_sigma;
c_int test_solve_KKT_n;
csc * test_solve_KKT_P;
c_float * test_solve_KKT_x;
c_float * test_solve_KKT_rhs;
csc * test_solve_KKT_A;
} solve_linsys_sols_data;

/* function to define problem data */
solve_linsys_sols_data *  generate_problem_solve_linsys_sols_data(){

solve_linsys_sols_data * data = (solve_linsys_sols_data *)c_malloc(sizeof(solve_linsys_sols_data));


// Matrix test_solve_KKT_Pu
//-------------------------
data->test_solve_KKT_Pu = c_malloc(sizeof(csc));
data->test_solve_KKT_Pu->m = 2;
data->test_solve_KKT_Pu->n = 2;
data->test_solve_KKT_Pu->nz = -1;
data->test_solve_KKT_Pu->nzmax = 2;
data->test_solve_KKT_Pu->x = c_malloc(2 * sizeof(c_float));
data->test_solve_KKT_Pu->x[0] = 0.20000000000000001110;
data->test_solve_KKT_Pu->x[1] = 0.59999999999999997780;
data->test_solve_KKT_Pu->i = c_malloc(2 * sizeof(c_int));
data->test_solve_KKT_Pu->i[0] = 0;
data->test_solve_KKT_Pu->i[1] = 1;
data->test_solve_KKT_Pu->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_solve_KKT_Pu->p[0] = 0;
data->test_solve_KKT_Pu->p[1] = 1;
data->test_solve_KKT_Pu->p[2] = 2;

data->test_solve_KKT_m = 2;

// Matrix test_solve_KKT_KKT
//--------------------------
data->test_solve_KKT_KKT = c_malloc(sizeof(csc));
data->test_solve_KKT_KKT->m = 4;
data->test_solve_KKT_KKT->n = 4;
data->test_solve_KKT_KKT->nz = -1;
data->test_solve_KKT_KKT->nzmax = 12;
data->test_solve_KKT_KKT->x = c_malloc(12 * sizeof(c_float));
data->test_solve_KKT_KKT->x[0] = 1.19999999999999995559;
data->test_solve_KKT_KKT->x[1] = -3.00000000000000000000;
data->test_solve_KKT_KKT->x[2] = 2.00000000000000000000;
data->test_solve_KKT_KKT->x[3] = 1.60000000000000008882;
data->test_solve_KKT_KKT->x[4] = 4.00000000000000000000;
data->test_solve_KKT_KKT->x[5] = -3.00000000000000000000;
data->test_solve_KKT_KKT->x[6] = -3.00000000000000000000;
data->test_solve_KKT_KKT->x[7] = 4.00000000000000000000;
data->test_solve_KKT_KKT->x[8] = -0.25000000000000000000;
data->test_solve_KKT_KKT->x[9] = 2.00000000000000000000;
data->test_solve_KKT_KKT->x[10] = -3.00000000000000000000;
data->test_solve_KKT_KKT->x[11] = -0.25000000000000000000;
data->test_solve_KKT_KKT->i = c_malloc(12 * sizeof(c_int));
data->test_solve_KKT_KKT->i[0] = 0;
data->test_solve_KKT_KKT->i[1] = 2;
data->test_solve_KKT_KKT->i[2] = 3;
data->test_solve_KKT_KKT->i[3] = 1;
data->test_solve_KKT_KKT->i[4] = 2;
data->test_solve_KKT_KKT->i[5] = 3;
data->test_solve_KKT_KKT->i[6] = 0;
data->test_solve_KKT_KKT->i[7] = 1;
data->test_solve_KKT_KKT->i[8] = 2;
data->test_solve_KKT_KKT->i[9] = 0;
data->test_solve_KKT_KKT->i[10] = 1;
data->test_solve_KKT_KKT->i[11] = 3;
data->test_solve_KKT_KKT->p = c_malloc((4 + 1) * sizeof(c_int));
data->test_solve_KKT_KKT->p[0] = 0;
data->test_solve_KKT_KKT->p[1] = 3;
data->test_solve_KKT_KKT->p[2] = 6;
data->test_solve_KKT_KKT->p[3] = 9;
data->test_solve_KKT_KKT->p[4] = 12;

data->test_solve_KKT_rho = 4.00000000000000000000;
data->test_solve_KKT_sigma = 1.00000000000000000000;
data->test_solve_KKT_n = 2;

// Matrix test_solve_KKT_P
//------------------------
data->test_solve_KKT_P = c_malloc(sizeof(csc));
data->test_solve_KKT_P->m = 2;
data->test_solve_KKT_P->n = 2;
data->test_solve_KKT_P->nz = -1;
data->test_solve_KKT_P->nzmax = 2;
data->test_solve_KKT_P->x = c_malloc(2 * sizeof(c_float));
data->test_solve_KKT_P->x[0] = 0.20000000000000001110;
data->test_solve_KKT_P->x[1] = 0.59999999999999997780;
data->test_solve_KKT_P->i = c_malloc(2 * sizeof(c_int));
data->test_solve_KKT_P->i[0] = 0;
data->test_solve_KKT_P->i[1] = 1;
data->test_solve_KKT_P->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_solve_KKT_P->p[0] = 0;
data->test_solve_KKT_P->p[1] = 1;
data->test_solve_KKT_P->p[2] = 2;

data->test_solve_KKT_x = c_malloc(4 * sizeof(c_float));
data->test_solve_KKT_x[0] = -0.79816491468410666332;
data->test_solve_KKT_x[1] = -0.97806969572154012216;
data->test_solve_KKT_x[2] = -1.18541949495228715605;
data->test_solve_KKT_x[3] = -1.86638953513327532363;
data->test_solve_KKT_rhs = c_malloc(4 * sizeof(c_float));
data->test_solve_KKT_rhs[0] = -1.13431848303061832972;
data->test_solve_KKT_rhs[1] = -0.70742088756378673775;
data->test_solve_KKT_rhs[2] = -1.22142916509576848760;
data->test_solve_KKT_rhs[3] = 1.80447664157972575971;

// Matrix test_solve_KKT_A
//------------------------
data->test_solve_KKT_A = c_malloc(sizeof(csc));
data->test_solve_KKT_A->m = 2;
data->test_solve_KKT_A->n = 2;
data->test_solve_KKT_A->nz = -1;
data->test_solve_KKT_A->nzmax = 4;
data->test_solve_KKT_A->x = c_malloc(4 * sizeof(c_float));
data->test_solve_KKT_A->x[0] = -3.00000000000000000000;
data->test_solve_KKT_A->x[1] = 2.00000000000000000000;
data->test_solve_KKT_A->x[2] = 4.00000000000000000000;
data->test_solve_KKT_A->x[3] = -3.00000000000000000000;
data->test_solve_KKT_A->i = c_malloc(4 * sizeof(c_int));
data->test_solve_KKT_A->i[0] = 0;
data->test_solve_KKT_A->i[1] = 1;
data->test_solve_KKT_A->i[2] = 0;
data->test_solve_KKT_A->i[3] = 1;
data->test_solve_KKT_A->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_solve_KKT_A->p[0] = 0;
data->test_solve_KKT_A->p[1] = 2;
data->test_solve_KKT_A->p[2] = 4;


return data;

}

/* function to clean data struct */
void clean_problem_solve_linsys_sols_data(solve_linsys_sols_data * data){

c_free(data->test_solve_KKT_Pu->x);
c_free(data->test_solve_KKT_Pu->i);
c_free(data->test_solve_KKT_Pu->p);
c_free(data->test_solve_KKT_Pu);
c_free(data->test_solve_KKT_KKT->x);
c_free(data->test_solve_KKT_KKT->i);
c_free(data->test_solve_KKT_KKT->p);
c_free(data->test_solve_KKT_KKT);
c_free(data->test_solve_KKT_P->x);
c_free(data->test_solve_KKT_P->i);
c_free(data->test_solve_KKT_P->p);
c_free(data->test_solve_KKT_P);
c_free(data->test_solve_KKT_x);
c_free(data->test_solve_KKT_rhs);
c_free(data->test_solve_KKT_A->x);
c_free(data->test_solve_KKT_A->i);
c_free(data->test_solve_KKT_A->p);
c_free(data->test_solve_KKT_A);

c_free(data);

}

#endif
