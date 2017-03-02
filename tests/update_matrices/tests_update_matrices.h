#include <stdio.h>
#include "osqp.h"
#include "cs.h"
#include "util.h"
#include "minunit.h"
#include "lin_sys.h"


#include "solve_linsys/solve_linsys.h"



static char * test_form_KKT(){
    update_matrices_sols_data *  data;
    c_float rho, sigma;
    c_int * PtoKKT, * AtoKKT, * PdiagIdx;


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


    // Form



    // Solve  KKT x = b via LDL given factorization
    solve_lin_sys(settings, p, data->test_solve_KKT_rhs);



    // mu_assert("Update matrices: error in forming KKT matrix, !", ... < TESTS_TOL);


    // Cleanup
    free_priv(p);
    c_free(settings);
    clean_problem_update_matrices_sols_data(data);

    return 0;
}


static char * tests_solve_linsys()
{
    mu_run_test(test_form_KKT);

    return 0;
}
