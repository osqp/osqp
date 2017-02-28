#ifndef SOLVE_LINSYS_DATA_H
#define SOLVE_LINSYS_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
c_float test_solve_KKT_rho;
c_float test_solve_KKT_sigma;
csc * test_solve_KKT_KKT;
csc * test_solve_KKT_P;
c_int test_solve_KKT_m;
c_float * test_solve_KKT_x;
c_int test_solve_KKT_n;
c_float * test_solve_KKT_rhs;
csc * test_solve_KKT_A;
csc * test_solve_KKT_Pu;
} solve_linsys_sols_data;

/* function to define problem data */
solve_linsys_sols_data *  generate_problem_solve_linsys_sols_data(){

solve_linsys_sols_data * data = (solve_linsys_sols_data *)c_malloc(sizeof(solve_linsys_sols_data));

data->test_solve_KKT_rho = 1.60000000000000008882;
data->test_solve_KKT_sigma = 0.10000000000000000555;

// Matrix test_solve_KKT_KKT
//--------------------------
data->test_solve_KKT_KKT = c_malloc(sizeof(csc));
data->test_solve_KKT_KKT->m = 11;
data->test_solve_KKT_KKT->n = 11;
data->test_solve_KKT_KKT->nz = -1;
data->test_solve_KKT_KKT->nzmax = 35;
data->test_solve_KKT_KKT->x = c_malloc(35 * sizeof(c_float));
data->test_solve_KKT_KKT->x[0] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[1] = 0.79829796685560061587;
data->test_solve_KKT_KKT->x[2] = 0.88822378780708466373;
data->test_solve_KKT_KKT->x[3] = 0.43778068488451560292;
data->test_solve_KKT_KKT->x[4] = 0.33218410520500962768;
data->test_solve_KKT_KKT->x[5] = 0.80724617766881634484;
data->test_solve_KKT_KKT->x[6] = 0.28615891707078788819;
data->test_solve_KKT_KKT->x[7] = 1.58188392538091671113;
data->test_solve_KKT_KKT->x[8] = 0.83663968613138317565;
data->test_solve_KKT_KKT->x[9] = 0.79829796685560061587;
data->test_solve_KKT_KKT->x[10] = 1.43249900737379465276;
data->test_solve_KKT_KKT->x[11] = 1.09497887083512313033;
data->test_solve_KKT_KKT->x[12] = 0.02341429694385355198;
data->test_solve_KKT_KKT->x[13] = 0.88822378780708466373;
data->test_solve_KKT_KKT->x[14] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[15] = 0.24226130805096113274;
data->test_solve_KKT_KKT->x[16] = 0.55321317150429660803;
data->test_solve_KKT_KKT->x[17] = 1.09497887083512313033;
data->test_solve_KKT_KKT->x[18] = 1.69641511191144234161;
data->test_solve_KKT_KKT->x[19] = 0.04636044263643468444;
data->test_solve_KKT_KKT->x[20] = 0.43778068488451560292;
data->test_solve_KKT_KKT->x[21] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[22] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[23] = 0.33218410520500962768;
data->test_solve_KKT_KKT->x[24] = 0.24226130805096113274;
data->test_solve_KKT_KKT->x[25] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[26] = 0.80724617766881634484;
data->test_solve_KKT_KKT->x[27] = 0.02341429694385355198;
data->test_solve_KKT_KKT->x[28] = 0.55321317150429660803;
data->test_solve_KKT_KKT->x[29] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[30] = 0.83663968613138317565;
data->test_solve_KKT_KKT->x[31] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[32] = 0.28615891707078788819;
data->test_solve_KKT_KKT->x[33] = 0.04636044263643468444;
data->test_solve_KKT_KKT->x[34] = -0.62500000000000000000;
data->test_solve_KKT_KKT->i = c_malloc(35 * sizeof(c_int));
data->test_solve_KKT_KKT->i[0] = 0;
data->test_solve_KKT_KKT->i[1] = 2;
data->test_solve_KKT_KKT->i[2] = 3;
data->test_solve_KKT_KKT->i[3] = 5;
data->test_solve_KKT_KKT->i[4] = 7;
data->test_solve_KKT_KKT->i[5] = 8;
data->test_solve_KKT_KKT->i[6] = 10;
data->test_solve_KKT_KKT->i[7] = 1;
data->test_solve_KKT_KKT->i[8] = 9;
data->test_solve_KKT_KKT->i[9] = 0;
data->test_solve_KKT_KKT->i[10] = 2;
data->test_solve_KKT_KKT->i[11] = 4;
data->test_solve_KKT_KKT->i[12] = 8;
data->test_solve_KKT_KKT->i[13] = 0;
data->test_solve_KKT_KKT->i[14] = 3;
data->test_solve_KKT_KKT->i[15] = 7;
data->test_solve_KKT_KKT->i[16] = 8;
data->test_solve_KKT_KKT->i[17] = 2;
data->test_solve_KKT_KKT->i[18] = 4;
data->test_solve_KKT_KKT->i[19] = 10;
data->test_solve_KKT_KKT->i[20] = 0;
data->test_solve_KKT_KKT->i[21] = 5;
data->test_solve_KKT_KKT->i[22] = 6;
data->test_solve_KKT_KKT->i[23] = 0;
data->test_solve_KKT_KKT->i[24] = 3;
data->test_solve_KKT_KKT->i[25] = 7;
data->test_solve_KKT_KKT->i[26] = 0;
data->test_solve_KKT_KKT->i[27] = 2;
data->test_solve_KKT_KKT->i[28] = 3;
data->test_solve_KKT_KKT->i[29] = 8;
data->test_solve_KKT_KKT->i[30] = 1;
data->test_solve_KKT_KKT->i[31] = 9;
data->test_solve_KKT_KKT->i[32] = 0;
data->test_solve_KKT_KKT->i[33] = 4;
data->test_solve_KKT_KKT->i[34] = 10;
data->test_solve_KKT_KKT->p = c_malloc((11 + 1) * sizeof(c_int));
data->test_solve_KKT_KKT->p[0] = 0;
data->test_solve_KKT_KKT->p[1] = 7;
data->test_solve_KKT_KKT->p[2] = 9;
data->test_solve_KKT_KKT->p[3] = 13;
data->test_solve_KKT_KKT->p[4] = 17;
data->test_solve_KKT_KKT->p[5] = 20;
data->test_solve_KKT_KKT->p[6] = 22;
data->test_solve_KKT_KKT->p[7] = 23;
data->test_solve_KKT_KKT->p[8] = 26;
data->test_solve_KKT_KKT->p[9] = 30;
data->test_solve_KKT_KKT->p[10] = 32;
data->test_solve_KKT_KKT->p[11] = 35;


// Matrix test_solve_KKT_P
//------------------------
data->test_solve_KKT_P = c_malloc(sizeof(csc));
data->test_solve_KKT_P->m = 5;
data->test_solve_KKT_P->n = 5;
data->test_solve_KKT_P->nz = -1;
data->test_solve_KKT_P->nzmax = 9;
data->test_solve_KKT_P->x = c_malloc(9 * sizeof(c_float));
data->test_solve_KKT_P->x[0] = 0.79829796685560061587;
data->test_solve_KKT_P->x[1] = 0.88822378780708466373;
data->test_solve_KKT_P->x[2] = 1.48188392538091662232;
data->test_solve_KKT_P->x[3] = 0.79829796685560061587;
data->test_solve_KKT_P->x[4] = 1.33249900737379456395;
data->test_solve_KKT_P->x[5] = 1.09497887083512313033;
data->test_solve_KKT_P->x[6] = 0.88822378780708466373;
data->test_solve_KKT_P->x[7] = 1.09497887083512313033;
data->test_solve_KKT_P->x[8] = 1.59641511191144225279;
data->test_solve_KKT_P->i = c_malloc(9 * sizeof(c_int));
data->test_solve_KKT_P->i[0] = 2;
data->test_solve_KKT_P->i[1] = 3;
data->test_solve_KKT_P->i[2] = 1;
data->test_solve_KKT_P->i[3] = 0;
data->test_solve_KKT_P->i[4] = 2;
data->test_solve_KKT_P->i[5] = 4;
data->test_solve_KKT_P->i[6] = 0;
data->test_solve_KKT_P->i[7] = 2;
data->test_solve_KKT_P->i[8] = 4;
data->test_solve_KKT_P->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_P->p[0] = 0;
data->test_solve_KKT_P->p[1] = 2;
data->test_solve_KKT_P->p[2] = 3;
data->test_solve_KKT_P->p[3] = 6;
data->test_solve_KKT_P->p[4] = 7;
data->test_solve_KKT_P->p[5] = 9;

data->test_solve_KKT_m = 6;
data->test_solve_KKT_x = c_malloc(11 * sizeof(c_float));
data->test_solve_KKT_x[0] = 0.48347838626670036621;
data->test_solve_KKT_x[1] = -0.82640521499645525072;
data->test_solve_KKT_x[2] = 1.28560659778111752161;
data->test_solve_KKT_x[3] = -1.23883073799993992381;
data->test_solve_KKT_x[4] = -0.92420581681939917296;
data->test_solve_KKT_x[5] = -1.21089740762467590329;
data->test_solve_KKT_x[6] = 1.00766113787128075430;
data->test_solve_KKT_x[7] = -1.04246304560009983575;
data->test_solve_KKT_x[8] = 0.39486417207819424213;
data->test_solve_KKT_x[9] = -0.92508295981412913545;
data->test_solve_KKT_x[10] = -1.38677818544097064546;
data->test_solve_KKT_n = 5;
data->test_solve_KKT_rhs = c_malloc(11 * sizeof(c_float));
data->test_solve_KKT_rhs[0] = -0.98019745934814039856;
data->test_solve_KKT_rhs[1] = -2.08123824259823697602;
data->test_solve_KKT_rhs[2] = 1.22484961322735519396;
data->test_solve_KKT_rhs[3] = 0.27144952969959845746;
data->test_solve_KKT_rhs[4] = -0.22441630390793010363;
data->test_solve_KKT_rhs[5] = 0.96846837883211889242;
data->test_solve_KKT_rhs[6] = -0.62978821116955052695;
data->test_solve_KKT_rhs[7] = 0.51202248358642532544;
data->test_solve_KKT_rhs[8] = -0.51173993514212057221;
data->test_solve_KKT_rhs[9] = -0.11322654980814174375;
data->test_solve_KKT_rhs[10] = 0.96224142658690281493;

// Matrix test_solve_KKT_A
//------------------------
data->test_solve_KKT_A = c_malloc(sizeof(csc));
data->test_solve_KKT_A->m = 6;
data->test_solve_KKT_A->n = 5;
data->test_solve_KKT_A->nz = -1;
data->test_solve_KKT_A->nzmax = 9;
data->test_solve_KKT_A->x = c_malloc(9 * sizeof(c_float));
data->test_solve_KKT_A->x[0] = 0.43778068488451560292;
data->test_solve_KKT_A->x[1] = 0.33218410520500962768;
data->test_solve_KKT_A->x[2] = 0.80724617766881634484;
data->test_solve_KKT_A->x[3] = 0.28615891707078788819;
data->test_solve_KKT_A->x[4] = 0.83663968613138317565;
data->test_solve_KKT_A->x[5] = 0.02341429694385355198;
data->test_solve_KKT_A->x[6] = 0.24226130805096113274;
data->test_solve_KKT_A->x[7] = 0.55321317150429660803;
data->test_solve_KKT_A->x[8] = 0.04636044263643468444;
data->test_solve_KKT_A->i = c_malloc(9 * sizeof(c_int));
data->test_solve_KKT_A->i[0] = 0;
data->test_solve_KKT_A->i[1] = 2;
data->test_solve_KKT_A->i[2] = 3;
data->test_solve_KKT_A->i[3] = 5;
data->test_solve_KKT_A->i[4] = 4;
data->test_solve_KKT_A->i[5] = 3;
data->test_solve_KKT_A->i[6] = 2;
data->test_solve_KKT_A->i[7] = 3;
data->test_solve_KKT_A->i[8] = 5;
data->test_solve_KKT_A->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_A->p[0] = 0;
data->test_solve_KKT_A->p[1] = 4;
data->test_solve_KKT_A->p[2] = 5;
data->test_solve_KKT_A->p[3] = 6;
data->test_solve_KKT_A->p[4] = 8;
data->test_solve_KKT_A->p[5] = 9;


// Matrix test_solve_KKT_Pu
//-------------------------
data->test_solve_KKT_Pu = c_malloc(sizeof(csc));
data->test_solve_KKT_Pu->m = 5;
data->test_solve_KKT_Pu->n = 5;
data->test_solve_KKT_Pu->nz = -1;
data->test_solve_KKT_Pu->nzmax = 6;
data->test_solve_KKT_Pu->x = c_malloc(6 * sizeof(c_float));
data->test_solve_KKT_Pu->x[0] = 1.48188392538091662232;
data->test_solve_KKT_Pu->x[1] = 0.79829796685560061587;
data->test_solve_KKT_Pu->x[2] = 1.33249900737379456395;
data->test_solve_KKT_Pu->x[3] = 0.88822378780708466373;
data->test_solve_KKT_Pu->x[4] = 1.09497887083512313033;
data->test_solve_KKT_Pu->x[5] = 1.59641511191144225279;
data->test_solve_KKT_Pu->i = c_malloc(6 * sizeof(c_int));
data->test_solve_KKT_Pu->i[0] = 1;
data->test_solve_KKT_Pu->i[1] = 0;
data->test_solve_KKT_Pu->i[2] = 2;
data->test_solve_KKT_Pu->i[3] = 0;
data->test_solve_KKT_Pu->i[4] = 2;
data->test_solve_KKT_Pu->i[5] = 4;
data->test_solve_KKT_Pu->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_Pu->p[0] = 0;
data->test_solve_KKT_Pu->p[1] = 0;
data->test_solve_KKT_Pu->p[2] = 1;
data->test_solve_KKT_Pu->p[3] = 3;
data->test_solve_KKT_Pu->p[4] = 4;
data->test_solve_KKT_Pu->p[5] = 6;


return data;

}

/* function to clean data struct */
void clean_problem_solve_linsys_sols_data(solve_linsys_sols_data * data){

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
c_free(data->test_solve_KKT_Pu->x);
c_free(data->test_solve_KKT_Pu->i);
c_free(data->test_solve_KKT_Pu->p);
c_free(data->test_solve_KKT_Pu);

c_free(data);

}

#endif
