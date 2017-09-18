#include "lin_alg.h"
#include "kkt.h"
#include "pardiso.h"


// Free LDL Factorization structure
void free_linsys_solver_pardiso(pardiso_solver *s) {
    if (s) {
        // Free pardiso solver using internal function
        phase = PARDISO_CLEANUP;
        PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
                 &(s->n), &(s->fdum), ia, ja, &(s->idum), &(s->nrhs),
                 s->iparm, &(s->msglvl), &(s->fdum), &(s->fdum), &(s->error));

        // Check each attribute of s and free it if it exists
        if (s->KKT)
            csc_spfree(s->KKT);
        if (s->bp)
            c_free(s->bp);
        if (s->Pdiag_idx)
            c_free(s->Pdiag_idx);
        if (s->PtoKKT)
            c_free(s->PtoKKT);
        if (s->AtoKKT)
            c_free(s->AtoKKT);
        if (s->rhotoKKT)
            c_free(s->rhotoKKT);

        c_free(s);

    }
}


// Initialize LDL Factorization structure
pardiso_solver *init_linsys_solver_pardiso(const csc * P, const csc * A, c_float sigma, c_float * rho_vec, c_int polish){
    c_int i;                     // loop counter
    // Define Variables
    pardiso_solver * s;          // Pardiso solver structure
    c_int n_plus_m;              // n_plus_m dimension

    // Size of KKT
    n_plus_m = P->m + A->m;

    // Allocate private structure to store KKT factorization
    s = c_calloc(1, sizeof(pardiso_solver));
    s->n = n_plus_m;

    // Working vector
    s->bp = c_malloc(sizeof(c_float) * n_plus_m);

    // Form KKT matrix
    if (polish){ // Called from polish()
        // Use s->bp for storing param2 = vec(delta)
        for (i = 0; i < A->m; i++){
            s->bp[i] = sigma;
        }

        s->KKT = form_KKT(P, A, 1, sigma, s->bp, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL);
    }
    else { // Called from ADMM algorithm

        // Allocate vectors of indices
        s->PtoKKT = c_malloc((P->p[P->n]) * sizeof(c_int));
        s->AtoKKT = c_malloc((A->p[A->n]) * sizeof(c_int));
        s->rhotoKKT = c_malloc((A->m) * sizeof(c_int));

        // Use s->bp for storing param2 = rho_inv_vec
        for (i = 0; i < A->m; i++){
            s->bp[i] = 1. / rho_vec[i];
        }

        s->KKT = form_KKT(P, A, 1, sigma, s->bp,
                          s->PtoKKT, s->AtoKKT,
                          &(s->Pdiag_idx), &(s->Pdiag_n), s->rhotoKKT);
    }

    // Check if matrix has been created
    if (!(s->KKT)) {
        #ifdef PRINTING
            c_print("Error in forming KKT matrix!\n");
        #endif
        return OSQP_NULL;
    }

    // Set Pardiso variables
    s->mtype = -2;        // Real symmetric matrix
    s->nrhs = 1;          // Number of right hand sides
    s->maxfct = 1;        // Maximum number of numerical factorizations
    s->mnum = 1;          // Which factorization to use
    s->msglvl = 0;        // Do not print statistical information
    s->error = 0;         // Initialize error flag
    for ( i = 0; i < 64; i++ ){
        s->iparm[i] = 0;  // Setup Pardiso control parameters
        s->pt[i] = 0;     // Initialize the internal solver memory pointer
    }
    s->iparm[0] = 1;      // No solver default
    s->iparm[1] = 3;      // Fill-in reordering from OpenMP
    s->iparm[5] = 1;      // Write solution into b
    s->iparm[7] = 2;      // Max number of iterative refinement steps
    s->iparm[9] = 13;     // Perturb the pivot elements with 1E-13
    s->iparm[34] = 1;     // PARDISO use C-style indexing for indices

    // Reordering and symbolic factorization
    s->phase = PARDISO_SYMBOLIC;
    PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
             &(s->n), s->KKT->x, s->KKT->p, s->KKT->i, &(s->idum), &(s->nrhs),
             s->iparm, &(s->msglvl), &(s->fdum), &(s->fdum), &(s->error));
    if ( s->error != 0 ){
        #ifdef PRINTING
            c_print("\nERROR during symbolic factorization: %d", s->error);
        #endif
        free_linsys_solver_pardiso(s);
        return OSQP_NULL;
    }

    // Numerical factorization
    s->phase = PARDISO_NUMERIC;
    PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
             &(s->n), s->KKT->x, s->KKT->p, s->KKT->i, &(s->idum), &(s->nrhs),
             s->iparm, &(s->msglvl), &(s->fdum), &(s->fdum), &(s->error));
    if ( s->error != 0 ){
        #ifdef PRINTING
            c_print("\nERROR during numerical factorization: %d", s->error);
        #endif
        free_linsys_solver_pardiso(s);
        return OSQP_NULL;
    }

    // Link Functions
    s->solve = &solve_linsys_pardiso;
    s->free = &free_linsys_solver_pardiso;
    s->update_matrices = &update_linsys_solver_matrices_pardiso;
    s->update_rho_vec = &update_linsys_solver_rho_vec_pardiso;

    // Assign type
    s->type = PARDISO_SOLVER;

    return p;
}

// Returns solution to linear system  Ax = b with solution stored in b
c_int solve_linsys_pardiso(pardiso_solver * s, c_float * b, const OSQPSettings *settings) {
    // Back substitution and iterative refinement
    s->phase = PARDISO_SOLVE;
    PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
             &(s->n), s->KKT->x, s->KKT->p, s->KKT->i, &(s->idum), &(s->nrhs),
             s->iparm, &(s->msglvl), b, s->bp, &(s->error));
    if ( s->error != 0 ){
        #ifdef PRINTING
        c_print("\nERROR during solution: %d", s->error);
        #endif
        return 1;
    }

    return 0;
}

// Update solver structure with new P and A
c_int update_linsys_solver_matrices_pardiso(pardiso_solver * s,
		const csc *P, const csc *A, const OSQPSettings *settings){

    // Update KKT matrix with new P
    update_KKT_P(s->KKT, P, s->PtoKKT, settings->sigma, s->Pdiag_idx, s->Pdiag_n);

    // Update KKT matrix with new A
    update_KKT_A(s->KKT, A, s->AtoKKT);

    // Perform numerical factorization
    s->phase = PARDISO_NUMERIC;
    PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
             &(s->n), s->KKT->x, s->KKT->p, s->KKT->i, &(s->idum), &(s->nrhs),
             s->iparm, &(s->msglvl), &(s->fdum), &(s->fdum), &(s->error));

    // Return exit flag
    return s->error;
}


c_int update_linsys_solver_rho_vec_pardiso(pardiso_solver * s, const c_float * rho_vec, const c_int m){
    c_int i;

    // Use s->bp for storing param2 = rho_inv_vec
    for (i = 0; i < m; i++){
        s->bp[i] = 1. / rho_vec[i];
    }

    // Update KKT matrix with new rho
    update_KKT_param2(s->KKT, s->bp, s->rhotoKKT, m);

    // Perform numerical factorization
    s->phase = PARDISO_NUMERIC;
    PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
             &(s->n), s->KKT->x, s->KKT->p, s->KKT->i, &(s->idum), &(s->nrhs),
             s->iparm, &(s->msglvl), &(s->fdum), &(s->fdum), &(s->error));

    // Return exit flag
    return s->error;
}
