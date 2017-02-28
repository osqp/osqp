#include "stdio.h"
#include "osqp.h"


int main(int argc, char **argv) {

    // Load problem data
    c_float basic_qp_P_x[4] = {4.00000000000000000000, 1.00000000000000000000, 1.00000000000000000000, 2.00000000000000000000, };
    c_int basic_qp_P_nnz = 4;
    c_int basic_qp_P_i[4] = {0, 1, 0, 1, };
    c_int basic_qp_P_p[3] = {0, 2, 4, };
    c_float basic_qp_q[2] = {1.00000000000000000000, 1.00000000000000000000, };
    c_float basic_qp_A_x[4] = {1.00000000000000000000, 1.00000000000000000000, 1.00000000000000000000, 1.00000000000000000000, };
    c_int basic_qp_A_nnz = 4;
    c_int basic_qp_A_i[4] = {0, 1, 0, 2, };
    c_int basic_qp_A_p[3] = {0, 2, 4, };
    c_float basic_qp_lA[3] = {1.00000000000000000000, 0.00000000000000000000, 0.00000000000000000000, };
    c_float basic_qp_uA[3] = {1.00000000000000000000, 0.69999999999999995559, 0.69999999999999995559, };
    c_int basic_qp_n = 2;
    c_int basic_qp_m = 3;


    // Problem settings
    OSQPSettings * settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

    // Structures
    OSQPWorkspace * work;  // Workspace
    OSQPData * data;  // OSQPData

    // Populate data
    data = (OSQPData *)c_malloc(sizeof(OSQPData));
    data->n = basic_qp_n;
    data->m = basic_qp_m;
    data->P = csc_matrix(data->n, data->n, basic_qp_P_nnz, basic_qp_P_x, basic_qp_P_i, basic_qp_P_p);
    data->q = basic_qp_q;
    data->A = csc_matrix(data->m, data->n, basic_qp_A_nnz, basic_qp_A_x, basic_qp_A_i, basic_qp_A_p);
    data->l = basic_qp_lA;
    data->u = basic_qp_uA;


    // Define Solver settings as default
    set_default_settings(settings);

    // Setup workspace
    work = osqp_setup(data, settings);

    // Solve Problem
    osqp_solve(work);

    // Clean workspace
    osqp_cleanup(work);
    c_free(data->A);
    c_free(data->P);
    c_free(data);
    c_free(settings);


    return 0;
};
