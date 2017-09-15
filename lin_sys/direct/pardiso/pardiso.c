#include "lin_alg.h"
#include "kkt.h"


#include "pardiso.h"


// Free LDL Factorization structure
void free_linsys_solver_pardiso(pardiso_solver *s) {
    if (s) {
        // Free pardiso solver using internal function

        // Check each attribute of s and free it if it exists
        if (s->bp)
            c_free(s->bp);
        if (s->PtoKKT)
            c_free(s->PtoKKT);
        if (s->AtoKKT)
            c_free(s->AtoKKT);
        if (s->rhotoKKT)
            c_free(s->rhotoKKT);
        // TODO: Add the others

        c_free(s);

    }
}


// Initialize LDL Factorization structure
pardiso_solver *init_linsys_solver_pardiso(const csc * P, const csc * A, c_float sigma, c_float * rho_vec, c_int polish){
    c_int i;                     // loop counter
    c_int pardiso_error;         // pardiso error exitflag
    // Define Variables
    pardiso_solver * p;  // Initialize LDL solver
    c_int n_plus_m;              // Define n_plus_m dimension
    csc * KKT_temp;              // Temporary KKT pointer

    // Allocate private structure to store KKT factorization
    p = c_calloc(1, sizeof(pardiso_solver));

    // Size of KKT
    n_plus_m = P->m + A->m;

    // Form and permute KKT matrix
    if (polish){ // Called from polish()
        // Use p->bp for storing param2 = vec(delta)
        for (i = 0; i < A->m; i++){
            p->bp[i] = sigma;
        }

        KKT_temp = form_KKT(P, A, sigma, p->bp, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL);

        // Reordering and symbolic factorization
        // TODO: ADD!
        // TODO: Check for exitflag pardiso_error. If error exit and free solver
    }
    else { // Called from ADMM algorithm

        // Allocate vectors of indices
        p->PtoKKT = c_malloc((P->p[P->n]) * sizeof(c_int));
        p->AtoKKT = c_malloc((A->p[A->n]) * sizeof(c_int));
        p->rhotoKKT = c_malloc((A->m) * sizeof(c_int));

        // Use p->bp for storing param2 = rho_inv_vec
        for (i = 0; i < A->m; i++){
            p->bp[i] = 1. / rho_vec[i];
        }

        KKT_temp = form_KKT(P, A, sigma, p->bp,
                            p->PtoKKT, p->AtoKKT,
                            &(p->Pdiag_idx), &(p->Pdiag_n), p->rhotoKKT);

        // Reordering and symbolic factorization
        // TODO: ADD!
        // TODO: Check for exitflag pardiso_error. If error exit and free solver
    }

    // Check if matrix has been created
    if (!KKT_temp){
        #ifdef PRINTING
            c_print("Error forming and permuting KKT matrix!\n");
        #endif
        return OSQP_NULL;
    }


    // Factorize the KKT matrix
    // TODO factor
    // TODO check error flag. If error exit and free the solver as in suitesparse
    // (N.B. See here below)
    // if (LDL_factor(KKT_temp, p) < 0) {
    //     free_linsys_solver_pardiso(p);
    //     return OSQP_NULL;
    // }

    // N.B. This is internally handeld in pardiso
    // Invert elements of D that are stored in p->Dinv
    // vec_ew_recipr(p->Dinv, p->Dinv, KKT_temp->n);

    if (polish){ // If KKT passed, assign it to KKT_temp
        // Polish, no need for KKT_temp
        csc_spfree(KKT_temp);
    }
    else { // If not embedded option 1 copy pointer to KKT_temp. Do not free it.
        p->KKT = KKT_temp;
    }

    // Link Functions
    p->solve = &solve_linsys_pardiso;
    p->free = &free_linsys_solver_pardiso;
    p->update_matrices = &update_linsys_solver_matrices_pardiso;
    p->update_rho_vec = &update_linsys_solver_rho_vec_pardiso;

    // Assign type
    p->type = PARDISO;

    return p;
}


c_int solve_linsys_pardiso(pardiso_solver * s, c_float * b, const OSQPSettings *settings) {
    /* returns solution to linear system */
    /* Ax = b with solution stored in b */
    // TODO: Add pardiso command

    return 0;
}

// Update private structure with new P and A
c_int update_linsys_solver_matrices_pardiso(pardiso_solver * s,
		const csc *P, const csc *A, const OSQPSettings *settings){
    c_int exitflag_pardiso;

    // Update KKT matrix with new P
    update_KKT_P(s->KKT, P, s->PtoKKT, settings->sigma, s->Pdiag_idx, s->Pdiag_n);

    // Update KKT matrix with new A
    update_KKT_A(s->KKT, A, s->AtoKKT);

    // Perform numeric factorization
    // TODO: Add pardiso command
    // TODO: Check exitflag and return erro in case

    // This is handled internally in pardiso
    // Invert elements of D that are stored in s->Dinv
    // vec_ew_recipr(s->Dinv, s->Dinv, s->KKT->n);

    // return exit flag
    return exitflag_pardiso;

}



c_int update_linsys_solver_rho_vec_pardiso(pardiso_solver * s, const c_float * rho_vec, const c_int m){
    c_int exitflag_pardiso, i;

    // Use s->bp for storing param2 = rho_inv_vec
    for (i = 0; i < m; i++){
        s->bp[i] = 1. / rho_vec[i];
    }

    // Update KKT matrix with new rho
    update_KKT_param2(s->KKT, s->bp, s->rhotoKKT, m);

    // Perform numeric factorization
    // TODO: Add pardiso command
    // TODO: Check exitflag and return error in case

    // This is handled internally in pardiso
    // Invert elements of D that are stored in s->Dinv
    //  vec_ew_recipr(s->Dinv, s->Dinv, s->KKT->n);

    // return exit flag
    return exitflag_pardiso;
}
