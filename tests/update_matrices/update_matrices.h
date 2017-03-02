#ifndef UPDATE_MATRICES_DATA_H
#define UPDATE_MATRICES_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
csc * test_form_KKT_KKTu_new;
csc * test_form_KKT_KKT_new;
csc * test_form_KKT_P;
csc * test_form_KKT_KKT;
c_float test_form_KKT_sigma;
c_int test_form_KKT_n;
csc * test_form_KKT_Pu;
csc * test_form_KKT_A_new;
c_int test_form_KKT_m;
csc * test_form_KKT_A;
c_float test_form_KKT_rho;
csc * test_form_KKT_Pu_new;
csc * test_form_KKT_P_new;
csc * test_form_KKT_KKTu;
} update_matrices_sols_data;

/* function to define problem data */
update_matrices_sols_data *  generate_problem_update_matrices_sols_data(){

update_matrices_sols_data * data = (update_matrices_sols_data *)c_malloc(sizeof(update_matrices_sols_data));


// Matrix test_form_KKT_KKTu_new
//------------------------------
data->test_form_KKT_KKTu_new = c_malloc(sizeof(csc));
data->test_form_KKT_KKTu_new->m = 5;
data->test_form_KKT_KKTu_new->n = 5;
data->test_form_KKT_KKTu_new->nz = -1;
data->test_form_KKT_KKTu_new->nzmax = 6;
data->test_form_KKT_KKTu_new->x = c_malloc(6 * sizeof(c_float));
data->test_form_KKT_KKTu_new->x[0] = 0.10000000000000000555;
data->test_form_KKT_KKTu_new->x[1] = -0.84159511110243967469;
data->test_form_KKT_KKTu_new->x[2] = -0.60407144205381213542;
data->test_form_KKT_KKTu_new->x[3] = -0.62500000000000000000;
data->test_form_KKT_KKTu_new->x[4] = -0.62500000000000000000;
data->test_form_KKT_KKTu_new->x[5] = -0.62500000000000000000;
data->test_form_KKT_KKTu_new->i = c_malloc(6 * sizeof(c_int));
data->test_form_KKT_KKTu_new->i[0] = 0;
data->test_form_KKT_KKTu_new->i[1] = 1;
data->test_form_KKT_KKTu_new->i[2] = 1;
data->test_form_KKT_KKTu_new->i[3] = 2;
data->test_form_KKT_KKTu_new->i[4] = 3;
data->test_form_KKT_KKTu_new->i[5] = 4;
data->test_form_KKT_KKTu_new->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_form_KKT_KKTu_new->p[0] = 0;
data->test_form_KKT_KKTu_new->p[1] = 1;
data->test_form_KKT_KKTu_new->p[2] = 2;
data->test_form_KKT_KKTu_new->p[3] = 4;
data->test_form_KKT_KKTu_new->p[4] = 5;
data->test_form_KKT_KKTu_new->p[5] = 6;


// Matrix test_form_KKT_KKT_new
//-----------------------------
data->test_form_KKT_KKT_new = c_malloc(sizeof(csc));
data->test_form_KKT_KKT_new->m = 5;
data->test_form_KKT_KKT_new->n = 5;
data->test_form_KKT_KKT_new->nz = -1;
data->test_form_KKT_KKT_new->nzmax = 7;
data->test_form_KKT_KKT_new->x = c_malloc(7 * sizeof(c_float));
data->test_form_KKT_KKT_new->x[0] = 0.10000000000000000555;
data->test_form_KKT_KKT_new->x[1] = -0.84159511110243967469;
data->test_form_KKT_KKT_new->x[2] = -0.60407144205381213542;
data->test_form_KKT_KKT_new->x[3] = -0.60407144205381213542;
data->test_form_KKT_KKT_new->x[4] = -0.62500000000000000000;
data->test_form_KKT_KKT_new->x[5] = -0.62500000000000000000;
data->test_form_KKT_KKT_new->x[6] = -0.62500000000000000000;
data->test_form_KKT_KKT_new->i = c_malloc(7 * sizeof(c_int));
data->test_form_KKT_KKT_new->i[0] = 0;
data->test_form_KKT_KKT_new->i[1] = 1;
data->test_form_KKT_KKT_new->i[2] = 2;
data->test_form_KKT_KKT_new->i[3] = 1;
data->test_form_KKT_KKT_new->i[4] = 2;
data->test_form_KKT_KKT_new->i[5] = 3;
data->test_form_KKT_KKT_new->i[6] = 4;
data->test_form_KKT_KKT_new->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_form_KKT_KKT_new->p[0] = 0;
data->test_form_KKT_KKT_new->p[1] = 1;
data->test_form_KKT_KKT_new->p[2] = 3;
data->test_form_KKT_KKT_new->p[3] = 5;
data->test_form_KKT_KKT_new->p[4] = 6;
data->test_form_KKT_KKT_new->p[5] = 7;


// Matrix test_form_KKT_P
//-----------------------
data->test_form_KKT_P = c_malloc(sizeof(csc));
data->test_form_KKT_P->m = 2;
data->test_form_KKT_P->n = 2;
data->test_form_KKT_P->nz = -1;
data->test_form_KKT_P->nzmax = 1;
data->test_form_KKT_P->x = c_malloc(1 * sizeof(c_float));
data->test_form_KKT_P->x[0] = 0.80090841211752494821;
data->test_form_KKT_P->i = c_malloc(1 * sizeof(c_int));
data->test_form_KKT_P->i[0] = 1;
data->test_form_KKT_P->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_form_KKT_P->p[0] = 0;
data->test_form_KKT_P->p[1] = 0;
data->test_form_KKT_P->p[2] = 1;


// Matrix test_form_KKT_KKT
//-------------------------
data->test_form_KKT_KKT = c_malloc(sizeof(csc));
data->test_form_KKT_KKT->m = 5;
data->test_form_KKT_KKT->n = 5;
data->test_form_KKT_KKT->nz = -1;
data->test_form_KKT_KKT->nzmax = 7;
data->test_form_KKT_KKT->x = c_malloc(7 * sizeof(c_float));
data->test_form_KKT_KKT->x[0] = 0.10000000000000000555;
data->test_form_KKT_KKT->x[1] = 0.90090841211752492601;
data->test_form_KKT_KKT->x[2] = 0.75808567763249168348;
data->test_form_KKT_KKT->x[3] = 0.75808567763249168348;
data->test_form_KKT_KKT->x[4] = -0.62500000000000000000;
data->test_form_KKT_KKT->x[5] = -0.62500000000000000000;
data->test_form_KKT_KKT->x[6] = -0.62500000000000000000;
data->test_form_KKT_KKT->i = c_malloc(7 * sizeof(c_int));
data->test_form_KKT_KKT->i[0] = 0;
data->test_form_KKT_KKT->i[1] = 1;
data->test_form_KKT_KKT->i[2] = 2;
data->test_form_KKT_KKT->i[3] = 1;
data->test_form_KKT_KKT->i[4] = 2;
data->test_form_KKT_KKT->i[5] = 3;
data->test_form_KKT_KKT->i[6] = 4;
data->test_form_KKT_KKT->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_form_KKT_KKT->p[0] = 0;
data->test_form_KKT_KKT->p[1] = 1;
data->test_form_KKT_KKT->p[2] = 3;
data->test_form_KKT_KKT->p[3] = 5;
data->test_form_KKT_KKT->p[4] = 6;
data->test_form_KKT_KKT->p[5] = 7;

data->test_form_KKT_sigma = 0.10000000000000000555;
data->test_form_KKT_n = 2;

// Matrix test_form_KKT_Pu
//------------------------
data->test_form_KKT_Pu = c_malloc(sizeof(csc));
data->test_form_KKT_Pu->m = 2;
data->test_form_KKT_Pu->n = 2;
data->test_form_KKT_Pu->nz = -1;
data->test_form_KKT_Pu->nzmax = 1;
data->test_form_KKT_Pu->x = c_malloc(1 * sizeof(c_float));
data->test_form_KKT_Pu->x[0] = 0.80090841211752494821;
data->test_form_KKT_Pu->i = c_malloc(1 * sizeof(c_int));
data->test_form_KKT_Pu->i[0] = 1;
data->test_form_KKT_Pu->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_form_KKT_Pu->p[0] = 0;
data->test_form_KKT_Pu->p[1] = 0;
data->test_form_KKT_Pu->p[2] = 1;


// Matrix test_form_KKT_A_new
//---------------------------
data->test_form_KKT_A_new = c_malloc(sizeof(csc));
data->test_form_KKT_A_new->m = 3;
data->test_form_KKT_A_new->n = 2;
data->test_form_KKT_A_new->nz = -1;
data->test_form_KKT_A_new->nzmax = 1;
data->test_form_KKT_A_new->x = c_malloc(1 * sizeof(c_float));
data->test_form_KKT_A_new->x[0] = -0.60407144205381213542;
data->test_form_KKT_A_new->i = c_malloc(1 * sizeof(c_int));
data->test_form_KKT_A_new->i[0] = 0;
data->test_form_KKT_A_new->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_form_KKT_A_new->p[0] = 0;
data->test_form_KKT_A_new->p[1] = 0;
data->test_form_KKT_A_new->p[2] = 1;

data->test_form_KKT_m = 3;

// Matrix test_form_KKT_A
//-----------------------
data->test_form_KKT_A = c_malloc(sizeof(csc));
data->test_form_KKT_A->m = 3;
data->test_form_KKT_A->n = 2;
data->test_form_KKT_A->nz = -1;
data->test_form_KKT_A->nzmax = 1;
data->test_form_KKT_A->x = c_malloc(1 * sizeof(c_float));
data->test_form_KKT_A->x[0] = 0.75808567763249168348;
data->test_form_KKT_A->i = c_malloc(1 * sizeof(c_int));
data->test_form_KKT_A->i[0] = 0;
data->test_form_KKT_A->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_form_KKT_A->p[0] = 0;
data->test_form_KKT_A->p[1] = 0;
data->test_form_KKT_A->p[2] = 1;

data->test_form_KKT_rho = 1.60000000000000008882;

// Matrix test_form_KKT_Pu_new
//----------------------------
data->test_form_KKT_Pu_new = c_malloc(sizeof(csc));
data->test_form_KKT_Pu_new->m = 2;
data->test_form_KKT_Pu_new->n = 2;
data->test_form_KKT_Pu_new->nz = -1;
data->test_form_KKT_Pu_new->nzmax = 1;
data->test_form_KKT_Pu_new->x = c_malloc(1 * sizeof(c_float));
data->test_form_KKT_Pu_new->x[0] = -0.94159511110243965248;
data->test_form_KKT_Pu_new->i = c_malloc(1 * sizeof(c_int));
data->test_form_KKT_Pu_new->i[0] = 1;
data->test_form_KKT_Pu_new->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_form_KKT_Pu_new->p[0] = 0;
data->test_form_KKT_Pu_new->p[1] = 0;
data->test_form_KKT_Pu_new->p[2] = 1;


// Matrix test_form_KKT_P_new
//---------------------------
data->test_form_KKT_P_new = c_malloc(sizeof(csc));
data->test_form_KKT_P_new->m = 2;
data->test_form_KKT_P_new->n = 2;
data->test_form_KKT_P_new->nz = -1;
data->test_form_KKT_P_new->nzmax = 1;
data->test_form_KKT_P_new->x = c_malloc(1 * sizeof(c_float));
data->test_form_KKT_P_new->x[0] = -0.94159511110243965248;
data->test_form_KKT_P_new->i = c_malloc(1 * sizeof(c_int));
data->test_form_KKT_P_new->i[0] = 1;
data->test_form_KKT_P_new->p = c_malloc((2 + 1) * sizeof(c_int));
data->test_form_KKT_P_new->p[0] = 0;
data->test_form_KKT_P_new->p[1] = 0;
data->test_form_KKT_P_new->p[2] = 1;


// Matrix test_form_KKT_KKTu
//--------------------------
data->test_form_KKT_KKTu = c_malloc(sizeof(csc));
data->test_form_KKT_KKTu->m = 5;
data->test_form_KKT_KKTu->n = 5;
data->test_form_KKT_KKTu->nz = -1;
data->test_form_KKT_KKTu->nzmax = 6;
data->test_form_KKT_KKTu->x = c_malloc(6 * sizeof(c_float));
data->test_form_KKT_KKTu->x[0] = 0.10000000000000000555;
data->test_form_KKT_KKTu->x[1] = 0.90090841211752492601;
data->test_form_KKT_KKTu->x[2] = 0.75808567763249168348;
data->test_form_KKT_KKTu->x[3] = -0.62500000000000000000;
data->test_form_KKT_KKTu->x[4] = -0.62500000000000000000;
data->test_form_KKT_KKTu->x[5] = -0.62500000000000000000;
data->test_form_KKT_KKTu->i = c_malloc(6 * sizeof(c_int));
data->test_form_KKT_KKTu->i[0] = 0;
data->test_form_KKT_KKTu->i[1] = 1;
data->test_form_KKT_KKTu->i[2] = 1;
data->test_form_KKT_KKTu->i[3] = 2;
data->test_form_KKT_KKTu->i[4] = 3;
data->test_form_KKT_KKTu->i[5] = 4;
data->test_form_KKT_KKTu->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_form_KKT_KKTu->p[0] = 0;
data->test_form_KKT_KKTu->p[1] = 1;
data->test_form_KKT_KKTu->p[2] = 2;
data->test_form_KKT_KKTu->p[3] = 4;
data->test_form_KKT_KKTu->p[4] = 5;
data->test_form_KKT_KKTu->p[5] = 6;


return data;

}

/* function to clean data struct */
void clean_problem_update_matrices_sols_data(update_matrices_sols_data * data){

c_free(data->test_form_KKT_KKTu_new->x);
c_free(data->test_form_KKT_KKTu_new->i);
c_free(data->test_form_KKT_KKTu_new->p);
c_free(data->test_form_KKT_KKTu_new);
c_free(data->test_form_KKT_KKT_new->x);
c_free(data->test_form_KKT_KKT_new->i);
c_free(data->test_form_KKT_KKT_new->p);
c_free(data->test_form_KKT_KKT_new);
c_free(data->test_form_KKT_P->x);
c_free(data->test_form_KKT_P->i);
c_free(data->test_form_KKT_P->p);
c_free(data->test_form_KKT_P);
c_free(data->test_form_KKT_KKT->x);
c_free(data->test_form_KKT_KKT->i);
c_free(data->test_form_KKT_KKT->p);
c_free(data->test_form_KKT_KKT);
c_free(data->test_form_KKT_Pu->x);
c_free(data->test_form_KKT_Pu->i);
c_free(data->test_form_KKT_Pu->p);
c_free(data->test_form_KKT_Pu);
c_free(data->test_form_KKT_A_new->x);
c_free(data->test_form_KKT_A_new->i);
c_free(data->test_form_KKT_A_new->p);
c_free(data->test_form_KKT_A_new);
c_free(data->test_form_KKT_A->x);
c_free(data->test_form_KKT_A->i);
c_free(data->test_form_KKT_A->p);
c_free(data->test_form_KKT_A);
c_free(data->test_form_KKT_Pu_new->x);
c_free(data->test_form_KKT_Pu_new->i);
c_free(data->test_form_KKT_Pu_new->p);
c_free(data->test_form_KKT_Pu_new);
c_free(data->test_form_KKT_P_new->x);
c_free(data->test_form_KKT_P_new->i);
c_free(data->test_form_KKT_P_new->p);
c_free(data->test_form_KKT_P_new);
c_free(data->test_form_KKT_KKTu->x);
c_free(data->test_form_KKT_KKTu->i);
c_free(data->test_form_KKT_KKTu->p);
c_free(data->test_form_KKT_KKTu);

c_free(data);

}

#endif
