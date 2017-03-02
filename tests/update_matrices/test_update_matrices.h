#include <stdio.h>
#include "osqp.h"
#include "cs.h"
#include "util.h"
#include "minunit.h"
#include "lin_sys.h"


#include "update_matrices/update_matrices.h"



static char * test_form_KKT(){
    update_matrices_sols_data *  data;
    c_float rho, sigma;
    c_int * PtoKKT, * AtoKKT, * Pdiag_idx, Pdiag_n;
    csc * Ptriu;

    // Load problem data
    data = generate_problem_update_matrices_sols_data();

    // Define rho and sigma to form KKT
    rho = data->test_form_KKT_rho;
    sigma = data->test_form_KKT_sigma;


    // Allocate vectors of indeces
    PtoKKT = c_malloc((data->test_form_KKT_Pu->p[data->test_form_KKT_Pu->m]) *
                    sizeof(c_int));
    AtoKKT = c_malloc((data->test_form_KKT_A->p[data->test_form_KKT_A->m]) *
                    sizeof(c_int));

    // Form upper triangular part in P
    Ptriu = csc_to_triu(data->test_form_KKT_P, &Pdiag_idx, &Pdiag_n);



    // DEBUG
    // print_vec_int(Pdiag_idx, Pdiag_n, "Pdiag");


    // Solve  KKT x = b via LDL given factorization
    // solve_lin_sys(settings, p, data->test_solve_KKT_rhs);



    // mu_assert("Update matrices: error in forming KKT matrix, !", ... < TESTS_TOL);


    // Cleanup
    clean_problem_update_matrices_sols_data(data);

    return 0;
}


static char * test_update_matrices()
{
    mu_run_test(test_form_KKT);

    return 0;
}
