#include "solve_linsys_data.h"


/* function to define problem data */
solve_linsys_sols_data *  generate_problem_solve_linsys_sols_data(){

solve_linsys_sols_data * data = (solve_linsys_sols_data *)c_malloc(sizeof(solve_linsys_sols_data));

data->test_solve_KKT_n = 3;
data->test_solve_KKT_m = 4;

// Matrix test_solve_KKT_A
//------------------------
data->test_solve_KKT_A = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_solve_KKT_A->m = 4;
data->test_solve_KKT_A->n = 3;
data->test_solve_KKT_A->nz = -1;
data->test_solve_KKT_A->nzmax = 5;
data->test_solve_KKT_A->x = (OSQPFloat*) c_malloc(5 * sizeof(OSQPFloat));
data->test_solve_KKT_A->x[0] = 0.45349788948065150596;
data->test_solve_KKT_A->x[1] = 0.78842870342840432052;
data->test_solve_KKT_A->x[2] = 0.32973171649909216452;
data->test_solve_KKT_A->x[3] = 0.30319482929164498497;
data->test_solve_KKT_A->x[4] = 0.13404169724716474832;
data->test_solve_KKT_A->i = (OSQPInt*) c_malloc(5 * sizeof(OSQPInt));
data->test_solve_KKT_A->i[0] = 0;
data->test_solve_KKT_A->i[1] = 1;
data->test_solve_KKT_A->i[2] = 3;
data->test_solve_KKT_A->i[3] = 1;
data->test_solve_KKT_A->i[4] = 3;
data->test_solve_KKT_A->p = (OSQPInt*) c_malloc((3 + 1) * sizeof(OSQPInt));
data->test_solve_KKT_A->p[0] = 0;
data->test_solve_KKT_A->p[1] = 1;
data->test_solve_KKT_A->p[2] = 3;
data->test_solve_KKT_A->p[3] = 5;


// Matrix test_solve_KKT_Pu
//-------------------------
data->test_solve_KKT_Pu = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_solve_KKT_Pu->m = 3;
data->test_solve_KKT_Pu->n = 3;
data->test_solve_KKT_Pu->nz = -1;
data->test_solve_KKT_Pu->nzmax = 3;
data->test_solve_KKT_Pu->x = (OSQPFloat*) c_malloc(3 * sizeof(OSQPFloat));
data->test_solve_KKT_Pu->x[0] = 0.27644413686269836417;
data->test_solve_KKT_Pu->x[1] = 0.25810370166387341939;
data->test_solve_KKT_Pu->x[2] = 0.85253551702235474963;
data->test_solve_KKT_Pu->i = (OSQPInt*) c_malloc(3 * sizeof(OSQPInt));
data->test_solve_KKT_Pu->i[0] = 0;
data->test_solve_KKT_Pu->i[1] = 0;
data->test_solve_KKT_Pu->i[2] = 2;
data->test_solve_KKT_Pu->p = (OSQPInt*) c_malloc((3 + 1) * sizeof(OSQPInt));
data->test_solve_KKT_Pu->p[0] = 0;
data->test_solve_KKT_Pu->p[1] = 1;
data->test_solve_KKT_Pu->p[2] = 1;
data->test_solve_KKT_Pu->p[3] = 3;

data->test_solve_KKT_rho = 4.00000000000000000000;
data->test_solve_KKT_sigma = 1.00000000000000000000;

// Matrix test_solve_KKT_KKT
//--------------------------
data->test_solve_KKT_KKT = (OSQPCscMatrix*) c_malloc(sizeof(OSQPCscMatrix));
data->test_solve_KKT_KKT->m = 7;
data->test_solve_KKT_KKT->n = 7;
data->test_solve_KKT_KKT->nz = -1;
data->test_solve_KKT_KKT->nzmax = 19;
data->test_solve_KKT_KKT->x = (OSQPFloat*) c_malloc(19 * sizeof(OSQPFloat));
data->test_solve_KKT_KKT->x[0] = 1.27644413686269841968;
data->test_solve_KKT_KKT->x[1] = 0.25810370166387341939;
data->test_solve_KKT_KKT->x[2] = 0.45349788948065150596;
data->test_solve_KKT_KKT->x[3] = 1.00000000000000000000;
data->test_solve_KKT_KKT->x[4] = 0.78842870342840432052;
data->test_solve_KKT_KKT->x[5] = 0.32973171649909216452;
data->test_solve_KKT_KKT->x[6] = 0.25810370166387341939;
data->test_solve_KKT_KKT->x[7] = 1.85253551702235474963;
data->test_solve_KKT_KKT->x[8] = 0.30319482929164498497;
data->test_solve_KKT_KKT->x[9] = 0.13404169724716474832;
data->test_solve_KKT_KKT->x[10] = 0.45349788948065150596;
data->test_solve_KKT_KKT->x[11] = -0.25000000000000000000;
data->test_solve_KKT_KKT->x[12] = 0.78842870342840432052;
data->test_solve_KKT_KKT->x[13] = 0.30319482929164498497;
data->test_solve_KKT_KKT->x[14] = -0.25000000000000000000;
data->test_solve_KKT_KKT->x[15] = -0.25000000000000000000;
data->test_solve_KKT_KKT->x[16] = 0.32973171649909216452;
data->test_solve_KKT_KKT->x[17] = 0.13404169724716474832;
data->test_solve_KKT_KKT->x[18] = -0.25000000000000000000;
data->test_solve_KKT_KKT->i = (OSQPInt*) c_malloc(19 * sizeof(OSQPInt));
data->test_solve_KKT_KKT->i[0] = 0;
data->test_solve_KKT_KKT->i[1] = 2;
data->test_solve_KKT_KKT->i[2] = 3;
data->test_solve_KKT_KKT->i[3] = 1;
data->test_solve_KKT_KKT->i[4] = 4;
data->test_solve_KKT_KKT->i[5] = 6;
data->test_solve_KKT_KKT->i[6] = 0;
data->test_solve_KKT_KKT->i[7] = 2;
data->test_solve_KKT_KKT->i[8] = 4;
data->test_solve_KKT_KKT->i[9] = 6;
data->test_solve_KKT_KKT->i[10] = 0;
data->test_solve_KKT_KKT->i[11] = 3;
data->test_solve_KKT_KKT->i[12] = 1;
data->test_solve_KKT_KKT->i[13] = 2;
data->test_solve_KKT_KKT->i[14] = 4;
data->test_solve_KKT_KKT->i[15] = 5;
data->test_solve_KKT_KKT->i[16] = 1;
data->test_solve_KKT_KKT->i[17] = 2;
data->test_solve_KKT_KKT->i[18] = 6;
data->test_solve_KKT_KKT->p = (OSQPInt*) c_malloc((7 + 1) * sizeof(OSQPInt));
data->test_solve_KKT_KKT->p[0] = 0;
data->test_solve_KKT_KKT->p[1] = 3;
data->test_solve_KKT_KKT->p[2] = 6;
data->test_solve_KKT_KKT->p[3] = 10;
data->test_solve_KKT_KKT->p[4] = 12;
data->test_solve_KKT_KKT->p[5] = 15;
data->test_solve_KKT_KKT->p[6] = 16;
data->test_solve_KKT_KKT->p[7] = 19;

data->test_solve_KKT_rhs = (OSQPFloat*) c_malloc(7 * sizeof(OSQPFloat));
data->test_solve_KKT_rhs[0] = -0.29245675096508860769;
data->test_solve_KKT_rhs[1] = -0.78190846235684208221;
data->test_solve_KKT_rhs[2] = -0.25719224061887069332;
data->test_solve_KKT_rhs[3] = 0.00814218051834350760;
data->test_solve_KKT_rhs[4] = -0.27560290529937042647;
data->test_solve_KKT_rhs[5] = 1.29406381439820727941;
data->test_solve_KKT_rhs[6] = 1.00672431530579431502;

data->test_solve_KKT_x = (OSQPFloat*) c_malloc(7 * sizeof(OSQPFloat));
data->test_solve_KKT_x[0] = -0.13711645448932024971;
data->test_solve_KKT_x[1] = -0.09378417875684068317;
data->test_solve_KKT_x[2] = 0.03925652506759663013;
data->test_solve_KKT_x[3] = -0.06218202272397653496;
data->test_solve_KKT_x[4] = -0.06203976304290045873;
data->test_solve_KKT_x[5] = 0.00000000000000000000;
data->test_solve_KKT_x[6] = -0.02566160699386421662;


return data;

}

/* function to clean data struct */
void clean_problem_solve_linsys_sols_data(solve_linsys_sols_data * data){

c_free(data->test_solve_KKT_A->x);
c_free(data->test_solve_KKT_A->i);
c_free(data->test_solve_KKT_A->p);
c_free(data->test_solve_KKT_A);
c_free(data->test_solve_KKT_Pu->x);
c_free(data->test_solve_KKT_Pu->i);
c_free(data->test_solve_KKT_Pu->p);
c_free(data->test_solve_KKT_Pu);
c_free(data->test_solve_KKT_KKT->x);
c_free(data->test_solve_KKT_KKT->i);
c_free(data->test_solve_KKT_KKT->p);
c_free(data->test_solve_KKT_KKT);
c_free(data->test_solve_KKT_rhs);
c_free(data->test_solve_KKT_x);

c_free(data);

}

