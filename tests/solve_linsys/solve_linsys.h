#ifndef SOLVE_LINSYS_DATA_H
#define SOLVE_LINSYS_DATA_H
#include "osqp.h"


/* create data and solutions structure */
typedef struct {
c_float test_solve_KKT_rho;
c_float * test_solve_KKT_x;
csc * test_solve_KKT_P;
csc * test_solve_KKT_A;
c_float * test_solve_KKT_rhs;
csc * test_solve_KKT_Pu;
c_int test_solve_KKT_n;
c_float test_solve_KKT_sigma;
csc * test_solve_KKT_KKT;
c_int test_solve_KKT_m;
} solve_linsys_sols_data;

/* function to define problem data */
solve_linsys_sols_data *  generate_problem_solve_linsys_sols_data(){

solve_linsys_sols_data * data = (solve_linsys_sols_data *)c_malloc(sizeof(solve_linsys_sols_data));

data->test_solve_KKT_rho = 1.60000000000000008882;
data->test_solve_KKT_x = c_malloc(11 * sizeof(c_float));
data->test_solve_KKT_x[0] = 4.36904791302812967047;
data->test_solve_KKT_x[1] = 1.71644777902693190796;
data->test_solve_KKT_x[2] = -0.77430459200348589199;
data->test_solve_KKT_x[3] = -5.22034979883037042470;
data->test_solve_KKT_x[4] = -1.87931577242024028251;
data->test_solve_KKT_x[5] = 3.45571033802188942730;
data->test_solve_KKT_x[6] = -0.26210865694291873851;
data->test_solve_KKT_x[7] = -0.65325653245744597086;
data->test_solve_KKT_x[8] = -4.40683435830947800582;
data->test_solve_KKT_x[9] = 0.97782795993364113318;
data->test_solve_KKT_x[10] = 0.81593955034821374905;

// Matrix test_solve_KKT_P
//------------------------
data->test_solve_KKT_P = c_malloc(sizeof(csc));
data->test_solve_KKT_P->m = 5;
data->test_solve_KKT_P->n = 5;
data->test_solve_KKT_P->nz = -1;
data->test_solve_KKT_P->nzmax = 10;
data->test_solve_KKT_P->x = c_malloc(10 * sizeof(c_float));
data->test_solve_KKT_P->x[0] = 0.28900172828689429938;
data->test_solve_KKT_P->x[1] = 0.45702529406566139158;
data->test_solve_KKT_P->x[2] = 1.56604925668349492796;
data->test_solve_KKT_P->x[3] = 0.52584196766631541298;
data->test_solve_KKT_P->x[4] = 1.56604925668349492796;
data->test_solve_KKT_P->x[5] = 0.76074538461119411981;
data->test_solve_KKT_P->x[6] = 0.28900172828689429938;
data->test_solve_KKT_P->x[7] = 0.52584196766631541298;
data->test_solve_KKT_P->x[8] = 0.76074538461119411981;
data->test_solve_KKT_P->x[9] = 1.36654062128004305521;
data->test_solve_KKT_P->i = c_malloc(10 * sizeof(c_int));
data->test_solve_KKT_P->i[0] = 3;
data->test_solve_KKT_P->i[1] = 1;
data->test_solve_KKT_P->i[2] = 2;
data->test_solve_KKT_P->i[3] = 4;
data->test_solve_KKT_P->i[4] = 1;
data->test_solve_KKT_P->i[5] = 4;
data->test_solve_KKT_P->i[6] = 0;
data->test_solve_KKT_P->i[7] = 1;
data->test_solve_KKT_P->i[8] = 2;
data->test_solve_KKT_P->i[9] = 4;
data->test_solve_KKT_P->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_P->p[0] = 0;
data->test_solve_KKT_P->p[1] = 1;
data->test_solve_KKT_P->p[2] = 4;
data->test_solve_KKT_P->p[3] = 6;
data->test_solve_KKT_P->p[4] = 7;
data->test_solve_KKT_P->p[5] = 10;


// Matrix test_solve_KKT_A
//------------------------
data->test_solve_KKT_A = c_malloc(sizeof(csc));
data->test_solve_KKT_A->m = 6;
data->test_solve_KKT_A->n = 5;
data->test_solve_KKT_A->nz = -1;
data->test_solve_KKT_A->nzmax = 9;
data->test_solve_KKT_A->x = c_malloc(9 * sizeof(c_float));
data->test_solve_KKT_A->x[0] = 0.27835072058477638990;
data->test_solve_KKT_A->x[1] = 0.78942743520155578274;
data->test_solve_KKT_A->x[2] = 0.83159905638861275623;
data->test_solve_KKT_A->x[3] = 0.64668094433879252936;
data->test_solve_KKT_A->x[4] = 0.44107209991876517030;
data->test_solve_KKT_A->x[5] = 0.61407400597261618813;
data->test_solve_KKT_A->x[6] = 0.45284344960673761360;
data->test_solve_KKT_A->x[7] = 0.16052888842366219713;
data->test_solve_KKT_A->x[8] = 0.23425878045345771561;
data->test_solve_KKT_A->i = c_malloc(9 * sizeof(c_int));
data->test_solve_KKT_A->i[0] = 0;
data->test_solve_KKT_A->i[1] = 0;
data->test_solve_KKT_A->i[2] = 1;
data->test_solve_KKT_A->i[3] = 2;
data->test_solve_KKT_A->i[4] = 4;
data->test_solve_KKT_A->i[5] = 0;
data->test_solve_KKT_A->i[6] = 3;
data->test_solve_KKT_A->i[7] = 3;
data->test_solve_KKT_A->i[8] = 5;
data->test_solve_KKT_A->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_A->p[0] = 0;
data->test_solve_KKT_A->p[1] = 1;
data->test_solve_KKT_A->p[2] = 5;
data->test_solve_KKT_A->p[3] = 7;
data->test_solve_KKT_A->p[4] = 8;
data->test_solve_KKT_A->p[5] = 9;

data->test_solve_KKT_rhs = c_malloc(11 * sizeof(c_float));
data->test_solve_KKT_rhs[0] = -0.10988586010065087839;
data->test_solve_KKT_rhs[1] = 1.27418991197260234038;
data->test_solve_KKT_rhs[2] = 1.30738632727282322321;
data->test_solve_KKT_rhs[3] = 0.03320319694371793445;
data->test_solve_KKT_rhs[4] = -2.25142028360815471544;
data->test_solve_KKT_rhs[5] = -0.06416068120282293619;
data->test_solve_KKT_rhs[6] = 1.59121426396845166984;
data->test_solve_KKT_rhs[7] = 1.51827940343526313072;
data->test_solve_KKT_rhs[8] = 1.56561576106529964925;
data->test_solve_KKT_rhs[9] = 0.14593475133778371911;
data->test_solve_KKT_rhs[10] = -0.95020843990174697069;

// Matrix test_solve_KKT_Pu
//-------------------------
data->test_solve_KKT_Pu = c_malloc(sizeof(csc));
data->test_solve_KKT_Pu->m = 5;
data->test_solve_KKT_Pu->n = 5;
data->test_solve_KKT_Pu->nz = -1;
data->test_solve_KKT_Pu->nzmax = 6;
data->test_solve_KKT_Pu->x = c_malloc(6 * sizeof(c_float));
data->test_solve_KKT_Pu->x[0] = 0.45702529406566139158;
data->test_solve_KKT_Pu->x[1] = 1.56604925668349492796;
data->test_solve_KKT_Pu->x[2] = 0.28900172828689429938;
data->test_solve_KKT_Pu->x[3] = 0.52584196766631541298;
data->test_solve_KKT_Pu->x[4] = 0.76074538461119411981;
data->test_solve_KKT_Pu->x[5] = 1.36654062128004305521;
data->test_solve_KKT_Pu->i = c_malloc(6 * sizeof(c_int));
data->test_solve_KKT_Pu->i[0] = 1;
data->test_solve_KKT_Pu->i[1] = 1;
data->test_solve_KKT_Pu->i[2] = 0;
data->test_solve_KKT_Pu->i[3] = 1;
data->test_solve_KKT_Pu->i[4] = 2;
data->test_solve_KKT_Pu->i[5] = 4;
data->test_solve_KKT_Pu->p = c_malloc((5 + 1) * sizeof(c_int));
data->test_solve_KKT_Pu->p[0] = 0;
data->test_solve_KKT_Pu->p[1] = 0;
data->test_solve_KKT_Pu->p[2] = 1;
data->test_solve_KKT_Pu->p[3] = 2;
data->test_solve_KKT_Pu->p[4] = 3;
data->test_solve_KKT_Pu->p[5] = 6;

data->test_solve_KKT_n = 5;
data->test_solve_KKT_sigma = 0.10000000000000000555;

// Matrix test_solve_KKT_KKT
//--------------------------
data->test_solve_KKT_KKT = c_malloc(sizeof(csc));
data->test_solve_KKT_KKT->m = 11;
data->test_solve_KKT_KKT->n = 11;
data->test_solve_KKT_KKT->nz = -1;
data->test_solve_KKT_KKT->nzmax = 37;
data->test_solve_KKT_KKT->x = c_malloc(37 * sizeof(c_float));
data->test_solve_KKT_KKT->x[0] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[1] = 0.28900172828689429938;
data->test_solve_KKT_KKT->x[2] = 0.27835072058477638990;
data->test_solve_KKT_KKT->x[3] = 0.55702529406566136938;
data->test_solve_KKT_KKT->x[4] = 1.56604925668349492796;
data->test_solve_KKT_KKT->x[5] = 0.52584196766631541298;
data->test_solve_KKT_KKT->x[6] = 0.78942743520155578274;
data->test_solve_KKT_KKT->x[7] = 0.83159905638861275623;
data->test_solve_KKT_KKT->x[8] = 0.64668094433879252936;
data->test_solve_KKT_KKT->x[9] = 0.44107209991876517030;
data->test_solve_KKT_KKT->x[10] = 1.56604925668349492796;
data->test_solve_KKT_KKT->x[11] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[12] = 0.76074538461119411981;
data->test_solve_KKT_KKT->x[13] = 0.61407400597261618813;
data->test_solve_KKT_KKT->x[14] = 0.45284344960673761360;
data->test_solve_KKT_KKT->x[15] = 0.28900172828689429938;
data->test_solve_KKT_KKT->x[16] = 0.10000000000000000555;
data->test_solve_KKT_KKT->x[17] = 0.16052888842366219713;
data->test_solve_KKT_KKT->x[18] = 0.52584196766631541298;
data->test_solve_KKT_KKT->x[19] = 0.76074538461119411981;
data->test_solve_KKT_KKT->x[20] = 1.46654062128004314403;
data->test_solve_KKT_KKT->x[21] = 0.23425878045345771561;
data->test_solve_KKT_KKT->x[22] = 0.27835072058477638990;
data->test_solve_KKT_KKT->x[23] = 0.78942743520155578274;
data->test_solve_KKT_KKT->x[24] = 0.61407400597261618813;
data->test_solve_KKT_KKT->x[25] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[26] = 0.83159905638861275623;
data->test_solve_KKT_KKT->x[27] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[28] = 0.64668094433879252936;
data->test_solve_KKT_KKT->x[29] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[30] = 0.45284344960673761360;
data->test_solve_KKT_KKT->x[31] = 0.16052888842366219713;
data->test_solve_KKT_KKT->x[32] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[33] = 0.44107209991876517030;
data->test_solve_KKT_KKT->x[34] = -0.62500000000000000000;
data->test_solve_KKT_KKT->x[35] = 0.23425878045345771561;
data->test_solve_KKT_KKT->x[36] = -0.62500000000000000000;
data->test_solve_KKT_KKT->i = c_malloc(37 * sizeof(c_int));
data->test_solve_KKT_KKT->i[0] = 0;
data->test_solve_KKT_KKT->i[1] = 3;
data->test_solve_KKT_KKT->i[2] = 5;
data->test_solve_KKT_KKT->i[3] = 1;
data->test_solve_KKT_KKT->i[4] = 2;
data->test_solve_KKT_KKT->i[5] = 4;
data->test_solve_KKT_KKT->i[6] = 5;
data->test_solve_KKT_KKT->i[7] = 6;
data->test_solve_KKT_KKT->i[8] = 7;
data->test_solve_KKT_KKT->i[9] = 9;
data->test_solve_KKT_KKT->i[10] = 1;
data->test_solve_KKT_KKT->i[11] = 2;
data->test_solve_KKT_KKT->i[12] = 4;
data->test_solve_KKT_KKT->i[13] = 5;
data->test_solve_KKT_KKT->i[14] = 8;
data->test_solve_KKT_KKT->i[15] = 0;
data->test_solve_KKT_KKT->i[16] = 3;
data->test_solve_KKT_KKT->i[17] = 8;
data->test_solve_KKT_KKT->i[18] = 1;
data->test_solve_KKT_KKT->i[19] = 2;
data->test_solve_KKT_KKT->i[20] = 4;
data->test_solve_KKT_KKT->i[21] = 10;
data->test_solve_KKT_KKT->i[22] = 0;
data->test_solve_KKT_KKT->i[23] = 1;
data->test_solve_KKT_KKT->i[24] = 2;
data->test_solve_KKT_KKT->i[25] = 5;
data->test_solve_KKT_KKT->i[26] = 1;
data->test_solve_KKT_KKT->i[27] = 6;
data->test_solve_KKT_KKT->i[28] = 1;
data->test_solve_KKT_KKT->i[29] = 7;
data->test_solve_KKT_KKT->i[30] = 2;
data->test_solve_KKT_KKT->i[31] = 3;
data->test_solve_KKT_KKT->i[32] = 8;
data->test_solve_KKT_KKT->i[33] = 1;
data->test_solve_KKT_KKT->i[34] = 9;
data->test_solve_KKT_KKT->i[35] = 4;
data->test_solve_KKT_KKT->i[36] = 10;
data->test_solve_KKT_KKT->p = c_malloc((11 + 1) * sizeof(c_int));
data->test_solve_KKT_KKT->p[0] = 0;
data->test_solve_KKT_KKT->p[1] = 3;
data->test_solve_KKT_KKT->p[2] = 10;
data->test_solve_KKT_KKT->p[3] = 15;
data->test_solve_KKT_KKT->p[4] = 18;
data->test_solve_KKT_KKT->p[5] = 22;
data->test_solve_KKT_KKT->p[6] = 26;
data->test_solve_KKT_KKT->p[7] = 28;
data->test_solve_KKT_KKT->p[8] = 30;
data->test_solve_KKT_KKT->p[9] = 33;
data->test_solve_KKT_KKT->p[10] = 35;
data->test_solve_KKT_KKT->p[11] = 37;

data->test_solve_KKT_m = 6;

return data;

}

/* function to clean data struct */
void clean_problem_solve_linsys_sols_data(solve_linsys_sols_data * data){

c_free(data->test_solve_KKT_x);
c_free(data->test_solve_KKT_P->x);
c_free(data->test_solve_KKT_P->i);
c_free(data->test_solve_KKT_P->p);
c_free(data->test_solve_KKT_P);
c_free(data->test_solve_KKT_A->x);
c_free(data->test_solve_KKT_A->i);
c_free(data->test_solve_KKT_A->p);
c_free(data->test_solve_KKT_A);
c_free(data->test_solve_KKT_rhs);
c_free(data->test_solve_KKT_Pu->x);
c_free(data->test_solve_KKT_Pu->i);
c_free(data->test_solve_KKT_Pu->p);
c_free(data->test_solve_KKT_Pu);
c_free(data->test_solve_KKT_KKT->x);
c_free(data->test_solve_KKT_KKT->i);
c_free(data->test_solve_KKT_KKT->p);
c_free(data->test_solve_KKT_KKT);

c_free(data);

}

#endif
