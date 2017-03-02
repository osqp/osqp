#ifndef SOLVE_LINSYS_DATA_H
#define SOLVE_LINSYS_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
csc * test_solve_KKT_P;
c_int test_solve_KKT_n;
csc * test_solve_KKT_Pu;
c_int test_solve_KKT_m;
c_float * test_solve_KKT_rhs;
c_float test_solve_KKT_sigma;
csc * test_solve_KKT_A;
c_float * test_solve_KKT_x;
csc * test_solve_KKT_KKT;
c_float test_solve_KKT_rho;
} solve_linsys_sols_data;

/* function to define problem data */
solve_linsys_sols_data *  generate_problem_solve_linsys_sols_data(){

solve_linsys_sols_data * data = (solve_linsys_sols_data *)c_malloc(sizeof(solve_linsys_sols_data));


// Matrix test_solve_KKT_P
//------------------------
data->test_solve_KKT_P = c_malloc(sizeof(csc));
data->test_solve_KKT_P->m = 5;
data->test_solve_KKT_P->n = 5;
data->test_solve_KKT_P->nz = -1;
data->test_solve_KKT_P->nzmax = 10;
data->test_solve_KKT_P->x = c_malloc(10 * sizeof(c_float));
data->test_solve_KKT_P->x[0] = 0.72563007567660042785;
data->test_solve_KKT_P->x[1] = 0.78194852972542450154;
data->test_solve_KKT_P->x[2] = 0.24233750402107157029;
data->test_solve_KKT_P->x[3] = 0.64159768639692393855;
data->test_solve_KKT_P->x[4] = 0.64159768639692393855;
data->test_solve_KKT_P->x[5] = 1.46937060617989567746;
data->test_solve_KKT_P->x[6] = 0.97355131118922055844;
data->test_solve_KKT_P->x[7] = 0.72563007567660042785;
data->test_solve_KKT_P->x[8] = 0.24233750402107157029;
data->test_solve_KKT_P->x[9] = 0.97355131118922055844;
data->test_solve_KKT_P->i = c_malloc(10 * sizeof(c_int));
data->test_solve_KKT_P->i[0] = 4;
data->test_solve_KKT_P->i[1] = 1;
data->test_solve_KKT_P->i[2] = 4;
data->test_solve_KKT_P->i[3] = 3;
data->test_solve_KKT_P->i[4] = 2;
data->test_solve_KKT_P->i[5] = 3;
data->test_solve_KKT_P->i[6] = 4;
data->test_solve_KKT_P->i[7] = 0;
data->test_solve_KKT_P->i[8] = 1;
data->test_solve_KKT_P->i[9] = 3;
data->test_solve_KKT_P->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_P->p[0] = 0;
data->test_solve_KKT_P->p[1] = 1;
data->test_solve_KKT_P->p[2] = 3;
data->test_solve_KKT_P->p[3] = 4;
data->test_solve_KKT_P->p[4] = 7;
data->test_solve_KKT_P->p[5] = 10;

data->test_solve_KKT_n = 5;

// Matrix test_solve_KKT_Pu
//-------------------------
data->test_solve_KKT_Pu = c_malloc(sizeof(csc));
data->test_solve_KKT_Pu->m = 5;
data->test_solve_KKT_Pu->n = 5;
data->test_solve_KKT_Pu->nz = -1;
data->test_solve_KKT_Pu->nzmax = 6;
data->test_solve_KKT_Pu->x = c_malloc(6 * sizeof(c_float));
data->test_solve_KKT_Pu->x[0] = 0.78194852972542450154;
data->test_solve_KKT_Pu->x[1] = 0.64159768639692393855;
data->test_solve_KKT_Pu->x[2] = 1.46937060617989567746;
data->test_solve_KKT_Pu->x[3] = 0.72563007567660042785;
data->test_solve_KKT_Pu->x[4] = 0.24233750402107157029;
data->test_solve_KKT_Pu->x[5] = 0.97355131118922055844;
data->test_solve_KKT_Pu->i = c_malloc(6 * sizeof(c_int));
data->test_solve_KKT_Pu->i[0] = 1;
data->test_solve_KKT_Pu->i[1] = 2;
data->test_solve_KKT_Pu->i[2] = 3;
data->test_solve_KKT_Pu->i[3] = 0;
data->test_solve_KKT_Pu->i[4] = 1;
data->test_solve_KKT_Pu->i[5] = 3;
data->test_solve_KKT_Pu->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_Pu->p[0] = 0;
data->test_solve_KKT_Pu->p[1] = 0;
data->test_solve_KKT_Pu->p[2] = 1;
data->test_solve_KKT_Pu->p[3] = 1;
data->test_solve_KKT_Pu->p[4] = 3;
data->test_solve_KKT_Pu->p[5] = 6;

data->test_solve_KKT_m = 6;
data->test_solve_KKT_rhs = c_malloc(11 * sizeof(c_float));
data->test_solve_KKT_rhs[0] = -0.35320473073431862820;
data->test_solve_KKT_rhs[1] = -1.37085293769168314881;
data->test_solve_KKT_rhs[2] = 0.73986272992459967135;
data->test_solve_KKT_rhs[3] = 0.45850031626670639806;
data->test_solve_KKT_rhs[4] = -1.28447707382065767767;
data->test_solve_KKT_rhs[5] = -1.40259457427372180582;
data->test_solve_KKT_rhs[6] = -0.98172210999902354001;
data->test_solve_KKT_rhs[7] = -1.71298369093390934204;
data->test_solve_KKT_rhs[8] = 1.02651276195357921139;
data->test_solve_KKT_rhs[9] = 0.04958306349164018356;
data->test_solve_KKT_rhs[10] = 0.50996086466596246556;
data->test_solve_KKT_sigma = 0.10000000000000000555;

// Matrix test_solve_KKT_A
//------------------------
data->test_solve_KKT_A = c_malloc(sizeof(csc));
data->test_solve_KKT_A->m = 6;
data->test_solve_KKT_A->n = 5;
data->test_solve_KKT_A->nz = -1;
data->test_solve_KKT_A->nzmax = 9;
data->test_solve_KKT_A->x = c_malloc(9 * sizeof(c_float));
data->test_solve_KKT_A->x[0] = 0.85511966080419055114;
data->test_solve_KKT_A->x[1] = 0.47220858741580140627;
data->test_solve_KKT_A->x[2] = 0.78165942566608137554;
data->test_solve_KKT_A->x[3] = 0.38079258371620805512;
data->test_solve_KKT_A->x[4] = 0.25948040440427710962;
data->test_solve_KKT_A->x[5] = 0.17830982777772019787;
data->test_solve_KKT_A->x[6] = 0.98048507899877845873;
data->test_solve_KKT_A->x[7] = 0.38690034301236819747;
data->test_solve_KKT_A->x[8] = 0.69239924588123102911;
data->test_solve_KKT_A->i = c_malloc(9 * sizeof(c_int));
data->test_solve_KKT_A->i[0] = 2;
data->test_solve_KKT_A->i[1] = 3;
data->test_solve_KKT_A->i[2] = 0;
data->test_solve_KKT_A->i[3] = 2;
data->test_solve_KKT_A->i[4] = 3;
data->test_solve_KKT_A->i[5] = 5;
data->test_solve_KKT_A->i[6] = 1;
data->test_solve_KKT_A->i[7] = 4;
data->test_solve_KKT_A->i[8] = 5;
data->test_solve_KKT_A->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_A->p[0] = 0;
data->test_solve_KKT_A->p[1] = 2;
data->test_solve_KKT_A->p[2] = 2;
data->test_solve_KKT_A->p[3] = 6;
data->test_solve_KKT_A->p[4] = 8;
data->test_solve_KKT_A->p[5] = 9;

data->test_solve_KKT_x = c_malloc(11 * sizeof(c_float));
data->test_solve_KKT_x[0] = -4.94004298345342007792;
data->test_solve_KKT_x[1] = -3.42924575108818308067;
data->test_solve_KKT_x[2] = 1.62271638072502177863;
data->test_solve_KKT_x[3] = -2.60975952019867918352;
data->test_solve_KKT_x[4] = 6.82339829003074793690;
data->test_solve_KKT_x[5] = 4.27360980552029445789;
data->test_solve_KKT_x[6] = -2.52337305492926811112;
data->test_solve_KKT_x[7] = -3.02948132188800967057;
data->test_solve_KKT_x[8] = -4.70108860518317861477;
data->test_solve_KKT_x[9] = -1.69487986725808426058;
data->test_solve_KKT_x[10] = 7.20624199052443792368;

// Matrix test_solve_KKT_KKT
//--------------------------
data->test_solve_KKT_KKT = c_malloc(sizeof(csc));
data->test_solve_KKT_KKT->m = 11;
data->test_solve_KKT_KKT->n = 11;
data->test_solve_KKT_KKT->nz = -1;
data->test_solve_KKT_KKT->nzmax = 37;
data->test_solve_KKT_KKT->x = c_malloc(37 * sizeof(c_float));
data->test_solve_KKT_KKT->x[0] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[1] = 0.72563007567660042785;
data->test_solve_KKT_KKT->x[2] = 0.85511966080419055114;
data->test_solve_KKT_KKT->x[3] = 0.47220858741580140627;
data->test_solve_KKT_KKT->x[4] = 0.88194852972542447933;
data->test_solve_KKT_KKT->x[5] = 0.24233750402107157029;
data->test_solve_KKT_KKT->x[6] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[7] = 0.64159768639692393855;
data->test_solve_KKT_KKT->x[8] = 0.78165942566608137554;
data->test_solve_KKT_KKT->x[9] = 0.38079258371620805512;
data->test_solve_KKT_KKT->x[10] = 0.25948040440427710962;
data->test_solve_KKT_KKT->x[11] = 0.17830982777772019787;
data->test_solve_KKT_KKT->x[12] = 0.64159768639692393855;
data->test_solve_KKT_KKT->x[13] = 1.56937060617989576627;
data->test_solve_KKT_KKT->x[14] = 0.97355131118922055844;
data->test_solve_KKT_KKT->x[15] = 0.98048507899877845873;
data->test_solve_KKT_KKT->x[16] = 0.38690034301236819747;
data->test_solve_KKT_KKT->x[17] = 0.72563007567660042785;
data->test_solve_KKT_KKT->x[18] = 0.24233750402107157029;
data->test_solve_KKT_KKT->x[19] = 0.97355131118922055844;
data->test_solve_KKT_KKT->x[20] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[21] = 0.69239924588123102911;
data->test_solve_KKT_KKT->x[22] = 0.78165942566608137554;
data->test_solve_KKT_KKT->x[23] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[24] = 0.98048507899877845873;
data->test_solve_KKT_KKT->x[25] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[26] = 0.85511966080419055114;
data->test_solve_KKT_KKT->x[27] = 0.38079258371620805512;
data->test_solve_KKT_KKT->x[28] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[29] = 0.47220858741580140627;
data->test_solve_KKT_KKT->x[30] = 0.25948040440427710962;
data->test_solve_KKT_KKT->x[31] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[32] = 0.38690034301236819747;
data->test_solve_KKT_KKT->x[33] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[34] = 0.17830982777772019787;
data->test_solve_KKT_KKT->x[35] = 0.69239924588123102911;
data->test_solve_KKT_KKT->x[36] = -0.62500000000000000000;
data->test_solve_KKT_KKT->i = c_malloc(37 * sizeof(c_int));
data->test_solve_KKT_KKT->i[0] = 0;
data->test_solve_KKT_KKT->i[1] = 4;
data->test_solve_KKT_KKT->i[2] = 7;
data->test_solve_KKT_KKT->i[3] = 8;
data->test_solve_KKT_KKT->i[4] = 1;
data->test_solve_KKT_KKT->i[5] = 4;
data->test_solve_KKT_KKT->i[6] = 2;
data->test_solve_KKT_KKT->i[7] = 3;
data->test_solve_KKT_KKT->i[8] = 5;
data->test_solve_KKT_KKT->i[9] = 7;
data->test_solve_KKT_KKT->i[10] = 8;
data->test_solve_KKT_KKT->i[11] = 10;
data->test_solve_KKT_KKT->i[12] = 2;
data->test_solve_KKT_KKT->i[13] = 3;
data->test_solve_KKT_KKT->i[14] = 4;
data->test_solve_KKT_KKT->i[15] = 6;
data->test_solve_KKT_KKT->i[16] = 9;
data->test_solve_KKT_KKT->i[17] = 0;
data->test_solve_KKT_KKT->i[18] = 1;
data->test_solve_KKT_KKT->i[19] = 3;
data->test_solve_KKT_KKT->i[20] = 4;
data->test_solve_KKT_KKT->i[21] = 10;
data->test_solve_KKT_KKT->i[22] = 2;
data->test_solve_KKT_KKT->i[23] = 5;
data->test_solve_KKT_KKT->i[24] = 3;
data->test_solve_KKT_KKT->i[25] = 6;
data->test_solve_KKT_KKT->i[26] = 0;
data->test_solve_KKT_KKT->i[27] = 2;
data->test_solve_KKT_KKT->i[28] = 7;
data->test_solve_KKT_KKT->i[29] = 0;
data->test_solve_KKT_KKT->i[30] = 2;
data->test_solve_KKT_KKT->i[31] = 8;
data->test_solve_KKT_KKT->i[32] = 3;
data->test_solve_KKT_KKT->i[33] = 9;
data->test_solve_KKT_KKT->i[34] = 2;
data->test_solve_KKT_KKT->i[35] = 4;
data->test_solve_KKT_KKT->i[36] = 10;
data->test_solve_KKT_KKT->p = c_malloc((11 + 1) * sizeof(c_int));
data->test_solve_KKT_KKT->p[0] = 0;
data->test_solve_KKT_KKT->p[1] = 4;
data->test_solve_KKT_KKT->p[2] = 6;
data->test_solve_KKT_KKT->p[3] = 12;
data->test_solve_KKT_KKT->p[4] = 17;
data->test_solve_KKT_KKT->p[5] = 22;
data->test_solve_KKT_KKT->p[6] = 24;
data->test_solve_KKT_KKT->p[7] = 26;
data->test_solve_KKT_KKT->p[8] = 29;
data->test_solve_KKT_KKT->p[9] = 32;
data->test_solve_KKT_KKT->p[10] = 34;
data->test_solve_KKT_KKT->p[11] = 37;

data->test_solve_KKT_rho = 1.60000000000000008882;

return data;

}

/* function to clean data struct */
void clean_problem_solve_linsys_sols_data(solve_linsys_sols_data * data){

c_free(data->test_solve_KKT_P->x);
c_free(data->test_solve_KKT_P->i);
c_free(data->test_solve_KKT_P->p);
c_free(data->test_solve_KKT_P);
c_free(data->test_solve_KKT_Pu->x);
c_free(data->test_solve_KKT_Pu->i);
c_free(data->test_solve_KKT_Pu->p);
c_free(data->test_solve_KKT_Pu);
c_free(data->test_solve_KKT_rhs);
c_free(data->test_solve_KKT_A->x);
c_free(data->test_solve_KKT_A->i);
c_free(data->test_solve_KKT_A->p);
c_free(data->test_solve_KKT_A);
c_free(data->test_solve_KKT_x);
c_free(data->test_solve_KKT_KKT->x);
c_free(data->test_solve_KKT_KKT->i);
c_free(data->test_solve_KKT_KKT->p);
c_free(data->test_solve_KKT_KKT);

c_free(data);

}

#endif
