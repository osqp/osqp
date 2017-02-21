#ifndef SOLVE_LINSYS_DATA_H
#define SOLVE_LINSYS_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
csc * test_solve_KKT_KKT;
csc * test_solve_KKT_A;
c_float * test_solve_KKT_rhs;
c_int test_solve_KKT_m;
c_float * test_solve_KKT_x;
csc * test_solve_KKT_P;
csc * test_solve_KKT_Pu;
c_int test_solve_KKT_n;
c_float test_solve_KKT_rho;
c_float test_solve_KKT_sigma;
} solve_linsys_sols_data;

/* function to define problem data */
solve_linsys_sols_data *  generate_problem_solve_linsys_sols_data(){

solve_linsys_sols_data * data = (solve_linsys_sols_data *)c_malloc(sizeof(solve_linsys_sols_data));


// Matrix test_solve_KKT_KKT
//--------------------------
data->test_solve_KKT_KKT = c_malloc(sizeof(csc));
data->test_solve_KKT_KKT->m = 11;
data->test_solve_KKT_KKT->n = 11;
data->test_solve_KKT_KKT->nz = -1;
data->test_solve_KKT_KKT->nzmax = 41;
data->test_solve_KKT_KKT->x = c_malloc(41 * sizeof(c_float));
data->test_solve_KKT_KKT->x[0] = 0.38583202383480597053;
data->test_solve_KKT_KKT->x[1] = 0.07929770875182784451;
data->test_solve_KKT_KKT->x[2] = 0.04515175329353582345;
data->test_solve_KKT_KKT->x[3] = 0.27175024687649951272;
data->test_solve_KKT_KKT->x[4] = 0.88560667687033711726;
data->test_solve_KKT_KKT->x[5] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[6] = 0.52602191788437746567;
data->test_solve_KKT_KKT->x[7] = 0.51689350893549323995;
data->test_solve_KKT_KKT->x[8] = 0.61429163578146950275;
data->test_solve_KKT_KKT->x[9] = 0.00488120135985858905;
data->test_solve_KKT_KKT->x[10] = 0.58125609523170518322;
data->test_solve_KKT_KKT->x[11] = 0.52602191788437746567;
data->test_solve_KKT_KKT->x[12] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[13] = 0.92842679258754678973;
data->test_solve_KKT_KKT->x[14] = 0.04543687423830278238;
data->test_solve_KKT_KKT->x[15] = 0.07929770875182784451;
data->test_solve_KKT_KKT->x[16] = 0.51689350893549323995;
data->test_solve_KKT_KKT->x[17] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[18] = 0.95603192972316619613;
data->test_solve_KKT_KKT->x[19] = 0.73452989909685484360;
data->test_solve_KKT_KKT->x[20] = 0.42483712527322337049;
data->test_solve_KKT_KKT->x[21] = 0.04515175329353582345;
data->test_solve_KKT_KKT->x[22] = 0.61429163578146950275;
data->test_solve_KKT_KKT->x[23] = 0.95603192972316619613;
data->test_solve_KKT_KKT->x[24] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[25] = 0.97590361329168595628;
data->test_solve_KKT_KKT->x[26] = 0.27175024687649951272;
data->test_solve_KKT_KKT->x[27] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[28] = 0.00488120135985858905;
data->test_solve_KKT_KKT->x[29] = 0.73452989909685484360;
data->test_solve_KKT_KKT->x[30] = 0.97590361329168595628;
data->test_solve_KKT_KKT->x[31] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[32] = 0.58125609523170518322;
data->test_solve_KKT_KKT->x[33] = 0.92842679258754678973;
data->test_solve_KKT_KKT->x[34] = 0.42483712527322337049;
data->test_solve_KKT_KKT->x[35] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[36] = 0.88560667687033711726;
data->test_solve_KKT_KKT->x[37] = 0.04543687423830278238;
data->test_solve_KKT_KKT->x[38] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[39] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[40] = -0.62500000000000000000;
data->test_solve_KKT_KKT->i = c_malloc(41 * sizeof(c_int));
data->test_solve_KKT_KKT->i[0] = 0;
data->test_solve_KKT_KKT->i[1] = 3;
data->test_solve_KKT_KKT->i[2] = 4;
data->test_solve_KKT_KKT->i[3] = 5;
data->test_solve_KKT_KKT->i[4] = 8;
data->test_solve_KKT_KKT->i[5] = 1;
data->test_solve_KKT_KKT->i[6] = 2;
data->test_solve_KKT_KKT->i[7] = 3;
data->test_solve_KKT_KKT->i[8] = 4;
data->test_solve_KKT_KKT->i[9] = 6;
data->test_solve_KKT_KKT->i[10] = 7;
data->test_solve_KKT_KKT->i[11] = 1;
data->test_solve_KKT_KKT->i[12] = 2;
data->test_solve_KKT_KKT->i[13] = 7;
data->test_solve_KKT_KKT->i[14] = 8;
data->test_solve_KKT_KKT->i[15] = 0;
data->test_solve_KKT_KKT->i[16] = 1;
data->test_solve_KKT_KKT->i[17] = 3;
data->test_solve_KKT_KKT->i[18] = 4;
data->test_solve_KKT_KKT->i[19] = 6;
data->test_solve_KKT_KKT->i[20] = 7;
data->test_solve_KKT_KKT->i[21] = 0;
data->test_solve_KKT_KKT->i[22] = 1;
data->test_solve_KKT_KKT->i[23] = 3;
data->test_solve_KKT_KKT->i[24] = 4;
data->test_solve_KKT_KKT->i[25] = 6;
data->test_solve_KKT_KKT->i[26] = 0;
data->test_solve_KKT_KKT->i[27] = 5;
data->test_solve_KKT_KKT->i[28] = 1;
data->test_solve_KKT_KKT->i[29] = 3;
data->test_solve_KKT_KKT->i[30] = 4;
data->test_solve_KKT_KKT->i[31] = 6;
data->test_solve_KKT_KKT->i[32] = 1;
data->test_solve_KKT_KKT->i[33] = 2;
data->test_solve_KKT_KKT->i[34] = 3;
data->test_solve_KKT_KKT->i[35] = 7;
data->test_solve_KKT_KKT->i[36] = 0;
data->test_solve_KKT_KKT->i[37] = 2;
data->test_solve_KKT_KKT->i[38] = 8;
data->test_solve_KKT_KKT->i[39] = 9;
data->test_solve_KKT_KKT->i[40] = 10;
data->test_solve_KKT_KKT->p = c_malloc((11 + 1) * sizeof(c_int));
data->test_solve_KKT_KKT->p[0] = 0;
data->test_solve_KKT_KKT->p[1] = 5;
data->test_solve_KKT_KKT->p[2] = 11;
data->test_solve_KKT_KKT->p[3] = 15;
data->test_solve_KKT_KKT->p[4] = 21;
data->test_solve_KKT_KKT->p[5] = 26;
data->test_solve_KKT_KKT->p[6] = 28;
data->test_solve_KKT_KKT->p[7] = 32;
data->test_solve_KKT_KKT->p[8] = 36;
data->test_solve_KKT_KKT->p[9] = 39;
data->test_solve_KKT_KKT->p[10] = 40;
data->test_solve_KKT_KKT->p[11] = 41;


// Matrix test_solve_KKT_A
//------------------------
data->test_solve_KKT_A = c_malloc(sizeof(csc));
data->test_solve_KKT_A->m = 6;
data->test_solve_KKT_A->n = 5;
data->test_solve_KKT_A->nz = -1;
data->test_solve_KKT_A->nzmax = 9;
data->test_solve_KKT_A->x = c_malloc(9 * sizeof(c_float));
data->test_solve_KKT_A->x[0] = 0.27175024687649951272;
data->test_solve_KKT_A->x[1] = 0.88560667687033711726;
data->test_solve_KKT_A->x[2] = 0.00488120135985858905;
data->test_solve_KKT_A->x[3] = 0.58125609523170518322;
data->test_solve_KKT_A->x[4] = 0.92842679258754678973;
data->test_solve_KKT_A->x[5] = 0.04543687423830278238;
data->test_solve_KKT_A->x[6] = 0.73452989909685484360;
data->test_solve_KKT_A->x[7] = 0.42483712527322337049;
data->test_solve_KKT_A->x[8] = 0.97590361329168595628;
data->test_solve_KKT_A->i = c_malloc(9 * sizeof(c_int));
data->test_solve_KKT_A->i[0] = 0;
data->test_solve_KKT_A->i[1] = 3;
data->test_solve_KKT_A->i[2] = 1;
data->test_solve_KKT_A->i[3] = 2;
data->test_solve_KKT_A->i[4] = 2;
data->test_solve_KKT_A->i[5] = 3;
data->test_solve_KKT_A->i[6] = 1;
data->test_solve_KKT_A->i[7] = 2;
data->test_solve_KKT_A->i[8] = 1;
data->test_solve_KKT_A->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_A->p[0] = 0;
data->test_solve_KKT_A->p[1] = 2;
data->test_solve_KKT_A->p[2] = 4;
data->test_solve_KKT_A->p[3] = 6;
data->test_solve_KKT_A->p[4] = 8;
data->test_solve_KKT_A->p[5] = 9;

data->test_solve_KKT_rhs = c_malloc(11 * sizeof(c_float));
data->test_solve_KKT_rhs[0] = 0.91018657136135150409;
data->test_solve_KKT_rhs[1] = -1.21004623673560418595;
data->test_solve_KKT_rhs[2] = 1.47661021277678461416;
data->test_solve_KKT_rhs[3] = 0.16473891796260137221;
data->test_solve_KKT_rhs[4] = -0.78655171155552727758;
data->test_solve_KKT_rhs[5] = 0.39352554996587990610;
data->test_solve_KKT_rhs[6] = -0.38460081683973729172;
data->test_solve_KKT_rhs[7] = 1.01754962847917340696;
data->test_solve_KKT_rhs[8] = 2.60532694482443050177;
data->test_solve_KKT_rhs[9] = -0.44192265720361662007;
data->test_solve_KKT_rhs[10] = -2.22486413611011801805;
data->test_solve_KKT_m = 6;
data->test_solve_KKT_x = c_malloc(11 * sizeof(c_float));
data->test_solve_KKT_x[0] = 2.81521382753420779110;
data->test_solve_KKT_x[1] = 3.55997997137135735457;
data->test_solve_KKT_x[2] = -0.68614028788413783388;
data->test_solve_KKT_x[3] = -1.47837066700653174678;
data->test_solve_KKT_x[4] = -0.38134805901637880421;
data->test_solve_KKT_x[5] = 0.59441520428268190823;
data->test_solve_KKT_x[6] = -1.68974177546055748245;
data->test_solve_KKT_x[7] = -0.34141974755433490696;
data->test_solve_KKT_x[8] = -0.22932936370058104592;
data->test_solve_KKT_x[9] = 0.70707625152578656991;
data->test_solve_KKT_x[10] = 3.55978261777618865125;

// Matrix test_solve_KKT_P
//------------------------
data->test_solve_KKT_P = c_malloc(sizeof(csc));
data->test_solve_KKT_P->m = 5;
data->test_solve_KKT_P->n = 5;
data->test_solve_KKT_P->nz = -1;
data->test_solve_KKT_P->nzmax = 13;
data->test_solve_KKT_P->x = c_malloc(13 * sizeof(c_float));
data->test_solve_KKT_P->x[0] = 0.28583202383480599273;
data->test_solve_KKT_P->x[1] = 0.07929770875182784451;
data->test_solve_KKT_P->x[2] = 0.04515175329353582345;
data->test_solve_KKT_P->x[3] = 0.52602191788437746567;
data->test_solve_KKT_P->x[4] = 0.51689350893549323995;
data->test_solve_KKT_P->x[5] = 0.61429163578146950275;
data->test_solve_KKT_P->x[6] = 0.52602191788437746567;
data->test_solve_KKT_P->x[7] = 0.07929770875182784451;
data->test_solve_KKT_P->x[8] = 0.51689350893549323995;
data->test_solve_KKT_P->x[9] = 0.95603192972316619613;
data->test_solve_KKT_P->x[10] = 0.04515175329353582345;
data->test_solve_KKT_P->x[11] = 0.61429163578146950275;
data->test_solve_KKT_P->x[12] = 0.95603192972316619613;
data->test_solve_KKT_P->i = c_malloc(13 * sizeof(c_int));
data->test_solve_KKT_P->i[0] = 0;
data->test_solve_KKT_P->i[1] = 3;
data->test_solve_KKT_P->i[2] = 4;
data->test_solve_KKT_P->i[3] = 2;
data->test_solve_KKT_P->i[4] = 3;
data->test_solve_KKT_P->i[5] = 4;
data->test_solve_KKT_P->i[6] = 1;
data->test_solve_KKT_P->i[7] = 0;
data->test_solve_KKT_P->i[8] = 1;
data->test_solve_KKT_P->i[9] = 4;
data->test_solve_KKT_P->i[10] = 0;
data->test_solve_KKT_P->i[11] = 1;
data->test_solve_KKT_P->i[12] = 3;
data->test_solve_KKT_P->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_P->p[0] = 0;
data->test_solve_KKT_P->p[1] = 3;
data->test_solve_KKT_P->p[2] = 6;
data->test_solve_KKT_P->p[3] = 7;
data->test_solve_KKT_P->p[4] = 10;
data->test_solve_KKT_P->p[5] = 13;


// Matrix test_solve_KKT_Pu
//-------------------------
data->test_solve_KKT_Pu = c_malloc(sizeof(csc));
data->test_solve_KKT_Pu->m = 5;
data->test_solve_KKT_Pu->n = 5;
data->test_solve_KKT_Pu->nz = -1;
data->test_solve_KKT_Pu->nzmax = 7;
data->test_solve_KKT_Pu->x = c_malloc(7 * sizeof(c_float));
data->test_solve_KKT_Pu->x[0] = 0.28583202383480599273;
data->test_solve_KKT_Pu->x[1] = 0.52602191788437746567;
data->test_solve_KKT_Pu->x[2] = 0.07929770875182784451;
data->test_solve_KKT_Pu->x[3] = 0.51689350893549323995;
data->test_solve_KKT_Pu->x[4] = 0.04515175329353582345;
data->test_solve_KKT_Pu->x[5] = 0.61429163578146950275;
data->test_solve_KKT_Pu->x[6] = 0.95603192972316619613;
data->test_solve_KKT_Pu->i = c_malloc(7 * sizeof(c_int));
data->test_solve_KKT_Pu->i[0] = 0;
data->test_solve_KKT_Pu->i[1] = 1;
data->test_solve_KKT_Pu->i[2] = 0;
data->test_solve_KKT_Pu->i[3] = 1;
data->test_solve_KKT_Pu->i[4] = 0;
data->test_solve_KKT_Pu->i[5] = 1;
data->test_solve_KKT_Pu->i[6] = 3;
data->test_solve_KKT_Pu->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_Pu->p[0] = 0;
data->test_solve_KKT_Pu->p[1] = 1;
data->test_solve_KKT_Pu->p[2] = 1;
data->test_solve_KKT_Pu->p[3] = 2;
data->test_solve_KKT_Pu->p[4] = 4;
data->test_solve_KKT_Pu->p[5] = 7;

data->test_solve_KKT_n = 5;
data->test_solve_KKT_rho = 1.60000000000000008882;
data->test_solve_KKT_sigma = 0.10000000000000000555;

return data;

}

/* function to clean data struct */
void clean_problem_solve_linsys_sols_data(solve_linsys_sols_data * data){

c_free(data->test_solve_KKT_KKT->x);
c_free(data->test_solve_KKT_KKT->i);
c_free(data->test_solve_KKT_KKT->p);
c_free(data->test_solve_KKT_KKT);
c_free(data->test_solve_KKT_A->x);
c_free(data->test_solve_KKT_A->i);
c_free(data->test_solve_KKT_A->p);
c_free(data->test_solve_KKT_A);
c_free(data->test_solve_KKT_rhs);
c_free(data->test_solve_KKT_x);
c_free(data->test_solve_KKT_P->x);
c_free(data->test_solve_KKT_P->i);
c_free(data->test_solve_KKT_P->p);
c_free(data->test_solve_KKT_P);
c_free(data->test_solve_KKT_Pu->x);
c_free(data->test_solve_KKT_Pu->i);
c_free(data->test_solve_KKT_Pu->p);
c_free(data->test_solve_KKT_Pu);

c_free(data);

}

#endif
