#ifndef UPDATE_MATRICES_DATA_H
#define UPDATE_MATRICES_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
c_int test_form_KKT_m;
csc * test_form_KKT_A;
c_float test_form_KKT_rho;
csc * test_form_KKT_P;
csc * test_form_KKT_Pu;
csc * test_form_KKT_KKT;
csc * test_form_KKT_KKTu;
c_int test_form_KKT_n;
c_float test_form_KKT_sigma;
} update_matrices_sols_data;

/* function to define problem data */
update_matrices_sols_data *  generate_problem_update_matrices_sols_data(){

update_matrices_sols_data * data = (update_matrices_sols_data *)c_malloc(sizeof(update_matrices_sols_data));

data->test_form_KKT_m = 5;

// Matrix test_form_KKT_A
//-----------------------
data->test_form_KKT_A = c_malloc(sizeof(csc));
data->test_form_KKT_A->m = 5;
data->test_form_KKT_A->n = 3;
data->test_form_KKT_A->nz = -1;
data->test_form_KKT_A->nzmax = 4;
data->test_form_KKT_A->x = c_malloc(4 * sizeof(c_float));
data->test_form_KKT_A->x[0] = 0.55502377651519185786;
data->test_form_KKT_A->x[1] = 0.36673608472927032853;
data->test_form_KKT_A->x[2] = 0.70525094474394578459;
data->test_form_KKT_A->x[3] = 0.31046795398335302885;
data->test_form_KKT_A->i = c_malloc(4 * sizeof(c_int));
data->test_form_KKT_A->i[0] = 0;
data->test_form_KKT_A->i[1] = 0;
data->test_form_KKT_A->i[2] = 1;
data->test_form_KKT_A->i[3] = 4;
data->test_form_KKT_A->p = c_malloc((3 + 1) * sizeof(c_int));
data->test_form_KKT_A->p[0] = 0;
data->test_form_KKT_A->p[1] = 0;
data->test_form_KKT_A->p[2] = 1;
data->test_form_KKT_A->p[3] = 4;

data->test_form_KKT_rho = 1.60000000000000008882;

// Matrix test_form_KKT_P
//-----------------------
data->test_form_KKT_P = c_malloc(sizeof(csc));
data->test_form_KKT_P->m = 3;
data->test_form_KKT_P->n = 3;
data->test_form_KKT_P->nz = -1;
data->test_form_KKT_P->nzmax = 3;
data->test_form_KKT_P->x = c_malloc(3 * sizeof(c_float));
data->test_form_KKT_P->x[0] = 0.65594726674217362916;
data->test_form_KKT_P->x[1] = 0.65594726674217362916;
data->test_form_KKT_P->x[2] = 1.26676927136856587452;
data->test_form_KKT_P->i = c_malloc(3 * sizeof(c_int));
data->test_form_KKT_P->i[0] = 1;
data->test_form_KKT_P->i[1] = 0;
data->test_form_KKT_P->i[2] = 1;
data->test_form_KKT_P->p = c_malloc((3 + 1) * sizeof(c_int));
data->test_form_KKT_P->p[0] = 0;
data->test_form_KKT_P->p[1] = 1;
data->test_form_KKT_P->p[2] = 3;
data->test_form_KKT_P->p[3] = 3;


// Matrix test_form_KKT_Pu
//------------------------
data->test_form_KKT_Pu = c_malloc(sizeof(csc));
data->test_form_KKT_Pu->m = 3;
data->test_form_KKT_Pu->n = 3;
data->test_form_KKT_Pu->nz = -1;
data->test_form_KKT_Pu->nzmax = 2;
data->test_form_KKT_Pu->x = c_malloc(2 * sizeof(c_float));
data->test_form_KKT_Pu->x[0] = 0.65594726674217362916;
data->test_form_KKT_Pu->x[1] = 1.26676927136856587452;
data->test_form_KKT_Pu->i = c_malloc(2 * sizeof(c_int));
data->test_form_KKT_Pu->i[0] = 0;
data->test_form_KKT_Pu->i[1] = 1;
data->test_form_KKT_Pu->p = c_malloc((3 + 1) * sizeof(c_int));
data->test_form_KKT_Pu->p[0] = 0;
data->test_form_KKT_Pu->p[1] = 0;
data->test_form_KKT_Pu->p[2] = 2;
data->test_form_KKT_Pu->p[3] = 2;


// Matrix test_form_KKT_KKT
//-------------------------
data->test_form_KKT_KKT = c_malloc(sizeof(csc));
data->test_form_KKT_KKT->m = 8;
data->test_form_KKT_KKT->n = 8;
data->test_form_KKT_KKT->nz = -1;
data->test_form_KKT_KKT->nzmax = 18;
data->test_form_KKT_KKT->x = c_malloc(18 * sizeof(c_float));
data->test_form_KKT_KKT->x[0] = 0.10000000000000000555;
data->test_form_KKT_KKT->x[1] = 0.65594726674217362916;
data->test_form_KKT_KKT->x[2] = 0.65594726674217362916;
data->test_form_KKT_KKT->x[3] = 1.36676927136856596334;
data->test_form_KKT_KKT->x[4] = 0.55502377651519185786;
data->test_form_KKT_KKT->x[5] = 0.10000000000000000555;
data->test_form_KKT_KKT->x[6] = 0.36673608472927032853;
data->test_form_KKT_KKT->x[7] = 0.70525094474394578459;
data->test_form_KKT_KKT->x[8] = 0.31046795398335302885;
data->test_form_KKT_KKT->x[9] = 0.55502377651519185786;
data->test_form_KKT_KKT->x[10] = 0.36673608472927032853;
data->test_form_KKT_KKT->x[11] = -0.62500000000000000000;
data->test_form_KKT_KKT->x[12] = 0.70525094474394578459;
data->test_form_KKT_KKT->x[13] = -0.62500000000000000000;
data->test_form_KKT_KKT->x[14] = -0.62500000000000000000;
data->test_form_KKT_KKT->x[15] = -0.62500000000000000000;
data->test_form_KKT_KKT->x[16] = 0.31046795398335302885;
data->test_form_KKT_KKT->x[17] = -0.62500000000000000000;
data->test_form_KKT_KKT->i = c_malloc(18 * sizeof(c_int));
data->test_form_KKT_KKT->i[0] = 0;
data->test_form_KKT_KKT->i[1] = 1;
data->test_form_KKT_KKT->i[2] = 0;
data->test_form_KKT_KKT->i[3] = 1;
data->test_form_KKT_KKT->i[4] = 3;
data->test_form_KKT_KKT->i[5] = 2;
data->test_form_KKT_KKT->i[6] = 3;
data->test_form_KKT_KKT->i[7] = 4;
data->test_form_KKT_KKT->i[8] = 7;
data->test_form_KKT_KKT->i[9] = 1;
data->test_form_KKT_KKT->i[10] = 2;
data->test_form_KKT_KKT->i[11] = 3;
data->test_form_KKT_KKT->i[12] = 2;
data->test_form_KKT_KKT->i[13] = 4;
data->test_form_KKT_KKT->i[14] = 5;
data->test_form_KKT_KKT->i[15] = 6;
data->test_form_KKT_KKT->i[16] = 2;
data->test_form_KKT_KKT->i[17] = 7;
data->test_form_KKT_KKT->p = c_malloc((8 + 1) * sizeof(c_int));
data->test_form_KKT_KKT->p[0] = 0;
data->test_form_KKT_KKT->p[1] = 2;
data->test_form_KKT_KKT->p[2] = 5;
data->test_form_KKT_KKT->p[3] = 9;
data->test_form_KKT_KKT->p[4] = 12;
data->test_form_KKT_KKT->p[5] = 14;
data->test_form_KKT_KKT->p[6] = 15;
data->test_form_KKT_KKT->p[7] = 16;
data->test_form_KKT_KKT->p[8] = 18;


// Matrix test_form_KKT_KKTu
//--------------------------
data->test_form_KKT_KKTu = c_malloc(sizeof(csc));
data->test_form_KKT_KKTu->m = 8;
data->test_form_KKT_KKTu->n = 8;
data->test_form_KKT_KKTu->nz = -1;
data->test_form_KKT_KKTu->nzmax = 13;
data->test_form_KKT_KKTu->x = c_malloc(13 * sizeof(c_float));
data->test_form_KKT_KKTu->x[0] = 0.10000000000000000555;
data->test_form_KKT_KKTu->x[1] = 0.65594726674217362916;
data->test_form_KKT_KKTu->x[2] = 1.36676927136856596334;
data->test_form_KKT_KKTu->x[3] = 0.10000000000000000555;
data->test_form_KKT_KKTu->x[4] = 0.55502377651519185786;
data->test_form_KKT_KKTu->x[5] = 0.36673608472927032853;
data->test_form_KKT_KKTu->x[6] = -0.62500000000000000000;
data->test_form_KKT_KKTu->x[7] = 0.70525094474394578459;
data->test_form_KKT_KKTu->x[8] = -0.62500000000000000000;
data->test_form_KKT_KKTu->x[9] = -0.62500000000000000000;
data->test_form_KKT_KKTu->x[10] = -0.62500000000000000000;
data->test_form_KKT_KKTu->x[11] = 0.31046795398335302885;
data->test_form_KKT_KKTu->x[12] = -0.62500000000000000000;
data->test_form_KKT_KKTu->i = c_malloc(13 * sizeof(c_int));
data->test_form_KKT_KKTu->i[0] = 0;
data->test_form_KKT_KKTu->i[1] = 0;
data->test_form_KKT_KKTu->i[2] = 1;
data->test_form_KKT_KKTu->i[3] = 2;
data->test_form_KKT_KKTu->i[4] = 1;
data->test_form_KKT_KKTu->i[5] = 2;
data->test_form_KKT_KKTu->i[6] = 3;
data->test_form_KKT_KKTu->i[7] = 2;
data->test_form_KKT_KKTu->i[8] = 4;
data->test_form_KKT_KKTu->i[9] = 5;
data->test_form_KKT_KKTu->i[10] = 6;
data->test_form_KKT_KKTu->i[11] = 2;
data->test_form_KKT_KKTu->i[12] = 7;
data->test_form_KKT_KKTu->p = c_malloc((8 + 1) * sizeof(c_int));
data->test_form_KKT_KKTu->p[0] = 0;
data->test_form_KKT_KKTu->p[1] = 1;
data->test_form_KKT_KKTu->p[2] = 3;
data->test_form_KKT_KKTu->p[3] = 4;
data->test_form_KKT_KKTu->p[4] = 7;
data->test_form_KKT_KKTu->p[5] = 9;
data->test_form_KKT_KKTu->p[6] = 10;
data->test_form_KKT_KKTu->p[7] = 11;
data->test_form_KKT_KKTu->p[8] = 13;

data->test_form_KKT_n = 3;
data->test_form_KKT_sigma = 0.10000000000000000555;

return data;

}

/* function to clean data struct */
void clean_problem_update_matrices_sols_data(update_matrices_sols_data * data){

c_free(data->test_form_KKT_A->x);
c_free(data->test_form_KKT_A->i);
c_free(data->test_form_KKT_A->p);
c_free(data->test_form_KKT_A);
c_free(data->test_form_KKT_P->x);
c_free(data->test_form_KKT_P->i);
c_free(data->test_form_KKT_P->p);
c_free(data->test_form_KKT_P);
c_free(data->test_form_KKT_Pu->x);
c_free(data->test_form_KKT_Pu->i);
c_free(data->test_form_KKT_Pu->p);
c_free(data->test_form_KKT_Pu);
c_free(data->test_form_KKT_KKT->x);
c_free(data->test_form_KKT_KKT->i);
c_free(data->test_form_KKT_KKT->p);
c_free(data->test_form_KKT_KKT);
c_free(data->test_form_KKT_KKTu->x);
c_free(data->test_form_KKT_KKTu->i);
c_free(data->test_form_KKT_KKTu->p);
c_free(data->test_form_KKT_KKTu);

c_free(data);

}

#endif
