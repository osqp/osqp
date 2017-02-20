#ifndef SOLVE_LINSYS_DATA_H
#define SOLVE_LINSYS_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
c_float * test_solve_KKT_rhs;
csc * test_solve_KKT_A;
c_float test_solve_KKT_rho;
csc * test_solve_KKT_Pu;
c_float test_solve_KKT_sigma;
csc * test_solve_KKT_P;
c_float * test_solve_KKT_x;
c_int test_solve_KKT_n;
csc * test_solve_KKT_KKT;
c_int test_solve_KKT_m;
} solve_linsys_sols_data;

/* function to define problem data */
solve_linsys_sols_data *  generate_problem_solve_linsys_sols_data(){

solve_linsys_sols_data * data = (solve_linsys_sols_data *)c_malloc(sizeof(solve_linsys_sols_data));

data->test_solve_KKT_rhs = c_malloc(11 * sizeof(c_float));
data->test_solve_KKT_rhs[0] = 0.05529837503177303937;
data->test_solve_KKT_rhs[1] = -1.00366377963895736514;
data->test_solve_KKT_rhs[2] = -0.87942710897738607390;
data->test_solve_KKT_rhs[3] = 0.69226041725091003176;
data->test_solve_KKT_rhs[4] = 0.27726691923095164727;
data->test_solve_KKT_rhs[5] = -0.66139644008715403611;
data->test_solve_KKT_rhs[6] = -2.38094349265301197249;
data->test_solve_KKT_rhs[7] = -0.33253877085428801275;
data->test_solve_KKT_rhs[8] = -2.76938822651497318361;
data->test_solve_KKT_rhs[9] = 2.10640279063116597769;
data->test_solve_KKT_rhs[10] = 0.01783199125216421502;

// Matrix test_solve_KKT_A
//------------------------
data->test_solve_KKT_A = c_malloc(sizeof(csc));
data->test_solve_KKT_A->m = 6;
data->test_solve_KKT_A->n = 5;
data->test_solve_KKT_A->nz = -1;
data->test_solve_KKT_A->nzmax = 9;
data->test_solve_KKT_A->x = c_malloc(9 * sizeof(c_float));
data->test_solve_KKT_A->x[0] = 0.56106361226346357363;
data->test_solve_KKT_A->x[1] = 0.21952590502886148993;
data->test_solve_KKT_A->x[2] = 0.74113200931357947621;
data->test_solve_KKT_A->x[3] = 0.25401823067353768160;
data->test_solve_KKT_A->x[4] = 0.93382023173701012020;
data->test_solve_KKT_A->x[5] = 0.03363339061963410703;
data->test_solve_KKT_A->x[6] = 0.49404447041760291004;
data->test_solve_KKT_A->x[7] = 0.89967863414819937429;
data->test_solve_KKT_A->x[8] = 0.72795660713678056464;
data->test_solve_KKT_A->i = c_malloc(9 * sizeof(c_int));
data->test_solve_KKT_A->i[0] = 0;
data->test_solve_KKT_A->i[1] = 1;
data->test_solve_KKT_A->i[2] = 3;
data->test_solve_KKT_A->i[3] = 3;
data->test_solve_KKT_A->i[4] = 4;
data->test_solve_KKT_A->i[5] = 0;
data->test_solve_KKT_A->i[6] = 1;
data->test_solve_KKT_A->i[7] = 1;
data->test_solve_KKT_A->i[8] = 4;
data->test_solve_KKT_A->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_A->p[0] = 0;
data->test_solve_KKT_A->p[1] = 0;
data->test_solve_KKT_A->p[2] = 3;
data->test_solve_KKT_A->p[3] = 5;
data->test_solve_KKT_A->p[4] = 7;
data->test_solve_KKT_A->p[5] = 9;

data->test_solve_KKT_rho = 1.60000000000000008882;

// Matrix test_solve_KKT_Pu
//-------------------------
data->test_solve_KKT_Pu = c_malloc(sizeof(csc));
data->test_solve_KKT_Pu->m = 5;
data->test_solve_KKT_Pu->n = 5;
data->test_solve_KKT_Pu->nz = -1;
data->test_solve_KKT_Pu->nzmax = 6;
data->test_solve_KKT_Pu->x = c_malloc(6 * sizeof(c_float));
data->test_solve_KKT_Pu->x[0] = 0.93770807415643064875;
data->test_solve_KKT_Pu->x[1] = 0.82108903896528750987;
data->test_solve_KKT_Pu->x[2] = 0.52636465867351811543;
data->test_solve_KKT_Pu->x[3] = 0.04203361318437348615;
data->test_solve_KKT_Pu->x[4] = 0.06894914624773396117;
data->test_solve_KKT_Pu->x[5] = 1.19947449969966091210;
data->test_solve_KKT_Pu->i = c_malloc(6 * sizeof(c_int));
data->test_solve_KKT_Pu->i[0] = 0;
data->test_solve_KKT_Pu->i[1] = 0;
data->test_solve_KKT_Pu->i[2] = 0;
data->test_solve_KKT_Pu->i[3] = 1;
data->test_solve_KKT_Pu->i[4] = 3;
data->test_solve_KKT_Pu->i[5] = 4;
data->test_solve_KKT_Pu->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_Pu->p[0] = 0;
data->test_solve_KKT_Pu->p[1] = 0;
data->test_solve_KKT_Pu->p[2] = 0;
data->test_solve_KKT_Pu->p[3] = 1;
data->test_solve_KKT_Pu->p[4] = 2;
data->test_solve_KKT_Pu->p[5] = 6;

data->test_solve_KKT_sigma = 0.10000000000000000555;

// Matrix test_solve_KKT_P
//------------------------
data->test_solve_KKT_P = c_malloc(sizeof(csc));
data->test_solve_KKT_P->m = 5;
data->test_solve_KKT_P->n = 5;
data->test_solve_KKT_P->nz = -1;
data->test_solve_KKT_P->nzmax = 11;
data->test_solve_KKT_P->x = c_malloc(11 * sizeof(c_float));
data->test_solve_KKT_P->x[0] = 0.93770807415643064875;
data->test_solve_KKT_P->x[1] = 0.82108903896528750987;
data->test_solve_KKT_P->x[2] = 0.52636465867351811543;
data->test_solve_KKT_P->x[3] = 0.04203361318437348615;
data->test_solve_KKT_P->x[4] = 0.93770807415643064875;
data->test_solve_KKT_P->x[5] = 0.82108903896528750987;
data->test_solve_KKT_P->x[6] = 0.06894914624773396117;
data->test_solve_KKT_P->x[7] = 0.52636465867351811543;
data->test_solve_KKT_P->x[8] = 0.04203361318437348615;
data->test_solve_KKT_P->x[9] = 0.06894914624773396117;
data->test_solve_KKT_P->x[10] = 1.19947449969966091210;
data->test_solve_KKT_P->i = c_malloc(11 * sizeof(c_int));
data->test_solve_KKT_P->i[0] = 2;
data->test_solve_KKT_P->i[1] = 3;
data->test_solve_KKT_P->i[2] = 4;
data->test_solve_KKT_P->i[3] = 4;
data->test_solve_KKT_P->i[4] = 0;
data->test_solve_KKT_P->i[5] = 0;
data->test_solve_KKT_P->i[6] = 4;
data->test_solve_KKT_P->i[7] = 0;
data->test_solve_KKT_P->i[8] = 1;
data->test_solve_KKT_P->i[9] = 3;
data->test_solve_KKT_P->i[10] = 4;
data->test_solve_KKT_P->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_P->p[0] = 0;
data->test_solve_KKT_P->p[1] = 3;
data->test_solve_KKT_P->p[2] = 4;
data->test_solve_KKT_P->p[3] = 5;
data->test_solve_KKT_P->p[4] = 7;
data->test_solve_KKT_P->p[5] = 11;

data->test_solve_KKT_x = c_malloc(11 * sizeof(c_float));
data->test_solve_KKT_x[0] = 0.20832841813877206461;
data->test_solve_KKT_x[1] = -3.73864560382775357184;
data->test_solve_KKT_x[2] = 1.23766442247771690788;
data->test_solve_KKT_x[3] = -1.42725298962075930476;
data->test_solve_KKT_x[4] = 0.08701077478842512980;
data->test_solve_KKT_x[5] = -2.37475987949170974645;
data->test_solve_KKT_x[6] = 1.49339075279879707203;
data->test_solve_KKT_x[7] = 0.53206203336686086480;
data->test_solve_KKT_x[8] = 0.50071619968656866373;
data->test_solve_KKT_x[9] = -1.41969063107367521326;
data->test_solve_KKT_x[10] = -0.02853118600346274403;
data->test_solve_KKT_n = 5;

// Matrix test_solve_KKT_KKT
//--------------------------
data->test_solve_KKT_KKT = c_malloc(sizeof(csc));
data->test_solve_KKT_KKT->m = 11;
data->test_solve_KKT_KKT->n = 11;
data->test_solve_KKT_KKT->nz = -1;
data->test_solve_KKT_KKT->nzmax = 39;
data->test_solve_KKT_KKT->x = c_malloc(39 * sizeof(c_float));
data->test_solve_KKT_KKT->x[0] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[1] = 0.93770807415643064875;
data->test_solve_KKT_KKT->x[2] = 0.82108903896528750987;
data->test_solve_KKT_KKT->x[3] = 0.52636465867351811543;
data->test_solve_KKT_KKT->x[4] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[5] = 0.04203361318437348615;
data->test_solve_KKT_KKT->x[6] = 0.56106361226346357363;
data->test_solve_KKT_KKT->x[7] = 0.21952590502886148993;
data->test_solve_KKT_KKT->x[8] = 0.74113200931357947621;
data->test_solve_KKT_KKT->x[9] = 0.93770807415643064875;
data->test_solve_KKT_KKT->x[10] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[11] = 0.25401823067353768160;
data->test_solve_KKT_KKT->x[12] = 0.93382023173701012020;
data->test_solve_KKT_KKT->x[13] = 0.82108903896528750987;
data->test_solve_KKT_KKT->x[14] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[15] = 0.06894914624773396117;
data->test_solve_KKT_KKT->x[16] = 0.03363339061963410703;
data->test_solve_KKT_KKT->x[17] = 0.49404447041760291004;
data->test_solve_KKT_KKT->x[18] = 0.52636465867351811543;
data->test_solve_KKT_KKT->x[19] = 0.04203361318437348615;
data->test_solve_KKT_KKT->x[20] = 0.06894914624773396117;
data->test_solve_KKT_KKT->x[21] = 1.29947449969966100092;
data->test_solve_KKT_KKT->x[22] = 0.89967863414819937429;
data->test_solve_KKT_KKT->x[23] = 0.72795660713678056464;
data->test_solve_KKT_KKT->x[24] = 0.56106361226346357363;
data->test_solve_KKT_KKT->x[25] = 0.03363339061963410703;
data->test_solve_KKT_KKT->x[26] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[27] = 0.21952590502886148993;
data->test_solve_KKT_KKT->x[28] = 0.49404447041760291004;
data->test_solve_KKT_KKT->x[29] = 0.89967863414819937429;
data->test_solve_KKT_KKT->x[30] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[31] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[32] = 0.74113200931357947621;
data->test_solve_KKT_KKT->x[33] = 0.25401823067353768160;
data->test_solve_KKT_KKT->x[34] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[35] = 0.93382023173701012020;
data->test_solve_KKT_KKT->x[36] = 0.72795660713678056464;
data->test_solve_KKT_KKT->x[37] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[38] = -0.62500000000000000000;
data->test_solve_KKT_KKT->i = c_malloc(39 * sizeof(c_int));
data->test_solve_KKT_KKT->i[0] = 0;
data->test_solve_KKT_KKT->i[1] = 2;
data->test_solve_KKT_KKT->i[2] = 3;
data->test_solve_KKT_KKT->i[3] = 4;
data->test_solve_KKT_KKT->i[4] = 1;
data->test_solve_KKT_KKT->i[5] = 4;
data->test_solve_KKT_KKT->i[6] = 5;
data->test_solve_KKT_KKT->i[7] = 6;
data->test_solve_KKT_KKT->i[8] = 8;
data->test_solve_KKT_KKT->i[9] = 0;
data->test_solve_KKT_KKT->i[10] = 2;
data->test_solve_KKT_KKT->i[11] = 8;
data->test_solve_KKT_KKT->i[12] = 9;
data->test_solve_KKT_KKT->i[13] = 0;
data->test_solve_KKT_KKT->i[14] = 3;
data->test_solve_KKT_KKT->i[15] = 4;
data->test_solve_KKT_KKT->i[16] = 5;
data->test_solve_KKT_KKT->i[17] = 6;
data->test_solve_KKT_KKT->i[18] = 0;
data->test_solve_KKT_KKT->i[19] = 1;
data->test_solve_KKT_KKT->i[20] = 3;
data->test_solve_KKT_KKT->i[21] = 4;
data->test_solve_KKT_KKT->i[22] = 6;
data->test_solve_KKT_KKT->i[23] = 9;
data->test_solve_KKT_KKT->i[24] = 1;
data->test_solve_KKT_KKT->i[25] = 3;
data->test_solve_KKT_KKT->i[26] = 5;
data->test_solve_KKT_KKT->i[27] = 1;
data->test_solve_KKT_KKT->i[28] = 3;
data->test_solve_KKT_KKT->i[29] = 4;
data->test_solve_KKT_KKT->i[30] = 6;
data->test_solve_KKT_KKT->i[31] = 7;
data->test_solve_KKT_KKT->i[32] = 1;
data->test_solve_KKT_KKT->i[33] = 2;
data->test_solve_KKT_KKT->i[34] = 8;
data->test_solve_KKT_KKT->i[35] = 2;
data->test_solve_KKT_KKT->i[36] = 4;
data->test_solve_KKT_KKT->i[37] = 9;
data->test_solve_KKT_KKT->i[38] = 10;
data->test_solve_KKT_KKT->p = c_malloc((11 + 1) * sizeof(c_int));
data->test_solve_KKT_KKT->p[0] = 0;
data->test_solve_KKT_KKT->p[1] = 4;
data->test_solve_KKT_KKT->p[2] = 9;
data->test_solve_KKT_KKT->p[3] = 13;
data->test_solve_KKT_KKT->p[4] = 18;
data->test_solve_KKT_KKT->p[5] = 24;
data->test_solve_KKT_KKT->p[6] = 27;
data->test_solve_KKT_KKT->p[7] = 31;
data->test_solve_KKT_KKT->p[8] = 32;
data->test_solve_KKT_KKT->p[9] = 35;
data->test_solve_KKT_KKT->p[10] = 38;
data->test_solve_KKT_KKT->p[11] = 39;

data->test_solve_KKT_m = 6;

return data;

}

/* function to clean data struct */
void clean_problem_solve_linsys_sols_data(solve_linsys_sols_data * data){

c_free(data->test_solve_KKT_rhs);
c_free(data->test_solve_KKT_A->x);
c_free(data->test_solve_KKT_A->i);
c_free(data->test_solve_KKT_A->p);
c_free(data->test_solve_KKT_A);
c_free(data->test_solve_KKT_Pu->x);
c_free(data->test_solve_KKT_Pu->i);
c_free(data->test_solve_KKT_Pu->p);
c_free(data->test_solve_KKT_Pu);
c_free(data->test_solve_KKT_P->x);
c_free(data->test_solve_KKT_P->i);
c_free(data->test_solve_KKT_P->p);
c_free(data->test_solve_KKT_P);
c_free(data->test_solve_KKT_x);
c_free(data->test_solve_KKT_KKT->x);
c_free(data->test_solve_KKT_KKT->i);
c_free(data->test_solve_KKT_KKT->p);
c_free(data->test_solve_KKT_KKT);

c_free(data);

}

#endif
