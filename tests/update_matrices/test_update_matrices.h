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
    csc * Ptriu, *Ptriu_new;
    csc * KKT;

    // Load problem data
    data = generate_problem_update_matrices_sols_data();

    // Define rho and sigma to form KKT
    rho = data->test_form_KKT_rho;
    sigma = data->test_form_KKT_sigma;


    // Allocate vectors of indeces
    PtoKKT = c_malloc((data->test_form_KKT_Pu->p[data->test_form_KKT_Pu->n]) *
                    sizeof(c_int));
    AtoKKT = c_malloc((data->test_form_KKT_A->p[data->test_form_KKT_A->n]) *
                    sizeof(c_int));

    // Form upper triangular part in P
    Ptriu = csc_to_triu(data->test_form_KKT_P);

    // Form KKT matrix storing the index vectors
    KKT = form_KKT(Ptriu, data->test_form_KKT_A, sigma, 1./rho, PtoKKT, AtoKKT, &Pdiag_idx, &Pdiag_n);

    // Assert if KKT matrix is the same as predicted one
    mu_assert("Update matrices: error in forming KKT matrix!", is_eq_csc(KKT, data->test_form_KKT_KKTu, TESTS_TOL));

    // Update KKT matrix with new P and new A
    update_KKT_P(KKT, data->test_form_KKT_Pu_new, PtoKKT, sigma, Pdiag_idx, Pdiag_n);
    update_KKT_A(KKT, data->test_form_KKT_A_new, AtoKKT);


    // Assert if KKT matrix is the same as predicted one
    mu_assert("Update matrices: error in updating KKT matrix!", is_eq_csc(KKT, data->test_form_KKT_KKTu_new, TESTS_TOL));


    // Solve  KKT x = b via LDL given factorization
    // solve_lin_sys(settings, p, data->test_solve_KKT_rhs);



    // mu_assert("Update matrices: error in forming KKT matrix, !", ... < TESTS_TOL);


    // Cleanup
    clean_problem_update_matrices_sols_data(data);
    c_free(Pdiag_idx);
    csc_spfree(Ptriu);
    csc_spfree(KKT);
    c_free(AtoKKT);
    c_free(PtoKKT);
    return 0;
}


static char * test_update_matrices()
{
    mu_run_test(test_form_KKT);

    return 0;
}
