#include "pardiso_interface.h"
#if EMBEDDED != 1
#include "kkt.h"
#endif

#define MKL_INT c_int
//#define MKL_INTERFACE_LP64  0x0
//#define MKL_INTERFACE_ILP64 0x1

#include "mkl_service.h"
#include "mkl_pardiso.h"

// Solver Phases
#define PARDISO_SYMBOLIC  (11)
#define PARDISO_NUMERIC   (22)
#define PARDISO_SOLVE     (33)
#define PARDISO_CLEANUP   (-1)

void update_settings_linsys_solver_pardiso(pardiso_solver     *s,
                                           const OSQPSettings *settings) {
    return;
}

// Warm starting not used by direct solvers
void warm_start_linsys_solver_pardiso(pardiso_solver    *s,
                                      const OSQPVectorf *x) {
  return;
}

// Free LDL Factorization structure
void free_linsys_solver_pardiso(pardiso_solver *s) {
    if (s) {

        // Free pardiso solver using internal function
        s->phase = PARDISO_CLEANUP;
        PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
                 &(s->nKKT), &(s->fdum), s->KKT_p, s->KKT_i, &(s->idum), &(s->nrhs),
                 s->iparm, &(s->msglvl), &(s->fdum), &(s->fdum), &(s->error));

      if ( s->error != 0 ){
#ifdef PRINTING
          c_eprint("Error during MKL Pardiso cleanup: %d", (int)s->error);
#endif
      }
        // Check each attribute of the structure and free it if it exists
        if (s->KKT)         csc_spfree(s->KKT);
        if (s->KKT_i)       c_free(s->KKT_i);
        if (s->KKT_p)       c_free(s->KKT_p);
        if (s->bp)          c_free(s->bp);
        if (s->sol)         c_free(s->sol);
        if (s->rho_inv_vec) c_free(s->rho_inv_vec);

        // These are required for matrix updates
        if (s->Pdiag_idx) c_free(s->Pdiag_idx);
        if (s->PtoKKT)    c_free(s->PtoKKT);
        if (s->AtoKKT)    c_free(s->AtoKKT);
        if (s->rhotoKKT)  c_free(s->rhotoKKT);

        c_free(s);

    }
}


// Initialize factorization structure
c_int init_linsys_solver_pardiso(pardiso_solver    **sp,
                                 const OSQPMatrix   *P,
                                 const OSQPMatrix   *A,
                                 const OSQPVectorf  *rho_vec,
                                 const OSQPSettings *settings,
                                 c_int               polishing) {

    c_int i;                     // loop counter
    c_int nnzKKT;                // Number of nonzeros in KKT
    // Define Variables
    c_int n_plus_m;              // n_plus_m dimension
    c_float* rhov;               //used for direct access to rho_vec data when polishing=false

    c_float sigma = settings->sigma;

    // Allocate private structure to store KKT factorization
    pardiso_solver *s = c_calloc(1, sizeof(pardiso_solver));
    *sp = s;

    // Size of KKT
    s->n = OSQPMatrix_get_n(P);
    s->m = OSQPMatrix_get_m(A);
    n_plus_m = s->n + s->m;
    s->nKKT = n_plus_m;

    // Sigma parameter
    s->sigma = sigma;

    // Polishing flag
    s->polishing = polishing;

    // Link Functions
    s->solve           = &solve_linsys_pardiso;
    s->free            = &free_linsys_solver_pardiso;
    s->warm_start      = &warm_start_linsys_solver_pardiso;
    s->update_matrices = &update_linsys_solver_matrices_pardiso;
    s->update_rho_vec  = &update_linsys_solver_rho_vec_pardiso;
    s->update_settings = &update_settings_linsys_solver_pardiso;

    // Assign type
    s->type = DIRECT_SOLVER;

    // Working vector
    s->bp = (c_float *)c_malloc(sizeof(c_float) * n_plus_m);

    // Solution vector
    s->sol  = (c_float *)c_malloc(sizeof(c_float) * n_plus_m);

    // Parameter vector
    if (rho_vec) {
        s->rho_inv_vec = (c_float *)c_malloc(sizeof(c_float) * n_plus_m);
    }
    // else it is NULL

    // Form KKT matrix
    if (polishing){ // Called from polish()
        s->KKT = form_KKT(OSQPMatrix_get_x(P),
                          OSQPMatrix_get_i(P),
                          OSQPMatrix_get_p(P),
                          OSQPMatrix_get_x(A),
                          OSQPMatrix_get_i(A),
                          OSQPMatrix_get_p(A),
                          OSQPMatrix_get_m(A),
                          OSQPMatrix_get_n(P),
                          1, sigma, s->rho_inv_vec, sigma,
                          OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL, OSQP_NULL);
    }
    else { // Called from ADMM algorithm

        // Allocate vectors of indices
        s->PtoKKT   = c_malloc(OSQPMatrix_get_nz(P) * sizeof(c_int));
        s->AtoKKT   = c_malloc(OSQPMatrix_get_nz(A) * sizeof(c_int));
        s->rhotoKKT = c_malloc(OSQPMatrix_get_m(A) * sizeof(c_int));

        // Use s->rho_inv_vec for storing param2 = rho_inv_vec
        if (rho_vec) {
            rhov = OSQPVectorf_data(rho_vec);
            for (i = 0; i < s->m; i++){
                s->rho_inv_vec[i] = 1. / rhov[i];
            }
        }
        else {
          s->rho_inv = 1. / settings->rho;
        }

        s->KKT = form_KKT(OSQPMatrix_get_x(P),
                          OSQPMatrix_get_i(P),
                          OSQPMatrix_get_p(P),
                          OSQPMatrix_get_x(A),
                          OSQPMatrix_get_i(A),
                          OSQPMatrix_get_p(A),
                          OSQPMatrix_get_m(A),
                          OSQPMatrix_get_n(P),
                          1, sigma, s->rho_inv_vec, s->rho_inv,
                          s->PtoKKT, s->AtoKKT,
                          &(s->Pdiag_idx), &(s->Pdiag_n), s->rhotoKKT);
    }

    // Check if matrix has been created
    if (!(s->KKT)) {
#ifdef PRINTING
	    c_eprint("Error in forming KKT matrix");
#endif
        free_linsys_solver_pardiso(s);
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    } else {
	    // Adjust indexing for Pardiso
	    nnzKKT = s->KKT->p[s->KKT->m];
	    s->KKT_i = c_malloc((nnzKKT) * sizeof(c_int));
	    s->KKT_p = c_malloc((s->KKT->m + 1) * sizeof(c_int));

	    for(i = 0; i < nnzKKT; i++){
	    	s->KKT_i[i] = s->KKT->i[i] + 1;
	    }
	    for(i = 0; i < n_plus_m+1; i++){
	    	s->KKT_p[i] = s->KKT->p[i] + 1;
	    }

    }

    // Set Pardiso variables
    s->mtype = -2;        // Real symmetric indefinite matrix
    s->nrhs = 1;          // Number of right hand sides
    s->maxfct = 1;        // Maximum number of numerical factorizations
    s->mnum = 1;          // Which factorization to use
    s->msglvl = 0;        // Do not print statistical information
    s->error = 0;         // Initialize error flag
    for ( i = 0; i < 64; i++ ) {
        s->iparm[i] = 0;  // Setup Pardiso control parameters
        s->pt[i] = 0;     // Initialize the internal solver memory pointer
    }
    s->iparm[0] = 1;      // No solver default
    s->iparm[1] = 3;      // Fill-in reordering from OpenMP
    if (polishing) {
        s->iparm[5] = 1;  // Write solution into b
    } else {
        s->iparm[5] = 0;  // Do NOT write solution into b
    }
    /* s->iparm[7] = 2;      // Max number of iterative refinement steps */
    s->iparm[7] = 0;      // Number of iterative refinement steps (auto, performs them only if perturbed pivots are obtained)
    s->iparm[9] = 13;     // Perturb the pivot elements with 1E-13
    s->iparm[34] = 0;     // Use Fortran-style indexing for indices
    /* s->iparm[34] = 1;     // Use C-style indexing for indices */

#if DFLOAT==1
    s->iparm[27] = 1;  // Input arrays and all internal arrays must be presented in single precision
#else
    s->iparm[27] = 0;  // Input arrays and all internal arrays must be presented in double precision
#endif

    // Print number of threads
    s->nthreads = mkl_get_max_threads();

    // Reordering and symbolic factorization
    s->phase = PARDISO_SYMBOLIC;
    PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
             &(s->nKKT), s->KKT->x, s->KKT_p, s->KKT_i, &(s->idum), &(s->nrhs),
             s->iparm, &(s->msglvl), &(s->fdum), &(s->fdum), &(s->error));
    if ( s->error != 0 ){
#ifdef PRINTING
        c_eprint("Error during symbolic factorization: %d", (int)s->error);
#endif
        free_linsys_solver_pardiso(s);
        *sp = OSQP_NULL;
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }

    // Numerical factorization
    s->phase = PARDISO_NUMERIC;
    PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
             &(s->nKKT), s->KKT->x, s->KKT_p, s->KKT_i, &(s->idum), &(s->nrhs),
             s->iparm, &(s->msglvl), &(s->fdum), &(s->fdum), &(s->error));
    if ( s->error ){
#ifdef PRINTING
        c_eprint("Error during numerical factorization: %d", (int)s->error);
#endif
        free_linsys_solver_pardiso(s);
        *sp = OSQP_NULL;
        return OSQP_LINSYS_SOLVER_INIT_ERROR;
    }


    // No error
    return 0;
}

// Returns solution to linear system  Ax = b with solution stored in b
c_int solve_linsys_pardiso(pardiso_solver *s,
                           OSQPVectorf    *b,
                           c_int           admm_iter) {

    c_int j;
    c_float* bv = OSQPVectorf_data(b);

    // Back substitution and iterative refinement
    s->phase = PARDISO_SOLVE;
    PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
             &(s->nKKT), s->KKT->x, s->KKT_p, s->KKT_i, &(s->idum), &(s->nrhs),
             s->iparm, &(s->msglvl), bv, s->sol, &(s->error));
    if ( s->error != 0 ){
#ifdef PRINTING
        c_eprint("Error during linear system solution: %d", (int)s->error);
#endif
        return 1;
    }

    if (!(s->polishing)) {
        /* copy x_tilde from s->sol */
        for (j = 0 ; j < s->n ; j++) {
            bv[j] = s->sol[j];
        }

        /* compute z_tilde from b and s->sol */
        if (s->rho_inv_vec) {
          for (j = 0 ; j < s->m ; j++) {
              bv[j + s->n] += s->rho_inv_vec[j] * s->sol[j + s->n];
          }
        }
        else {
          for (j = 0 ; j < s->m ; j++) {
              bv[j + s->n] += s->rho_inv * s->sol[j + s->n];
          }
        }
    }

    return 0;
}

// Update solver structure with new P and A
c_int update_linsys_solver_matrices_pardiso(
                  pardiso_solver * s,
                  const OSQPMatrix *P,
                  const OSQPMatrix *A) {

    // Update KKT matrix with new P
    update_KKT_P(s->KKT,
                 OSQPMatrix_get_x(P),
                 OSQPMatrix_get_p(P),
                 OSQPMatrix_get_n(P),
                 s->PtoKKT, s->sigma, s->Pdiag_idx, s->Pdiag_n);

    // Update KKT matrix with new A
    update_KKT_A(s->KKT,
                 OSQPMatrix_get_x(A),
                 OSQPMatrix_get_p(A),
                 OSQPMatrix_get_n(P),
                 s->AtoKKT);

    // Perform numerical factorization
    s->phase = PARDISO_NUMERIC;
    PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
             &(s->nKKT), s->KKT->x, s->KKT_p, s->KKT_i, &(s->idum), &(s->nrhs),
             s->iparm, &(s->msglvl), &(s->fdum), &(s->fdum), &(s->error));

    // Return exit flag
    return s->error;
}


c_int update_linsys_solver_rho_vec_pardiso(pardiso_solver    *s,
                                           const OSQPVectorf *rho_vec,
                                           c_float            rho_sc) {

    c_int i;
    c_float* rhov;

    // Update internal rho_inv_vec
    if (s->rho_inv_vec) {
      rhov = OSQPVectorf_data(rho_vec);
      for (i = 0; i < s->m; i++){
          s->rho_inv_vec[i] = 1. / rhov[i];
      }
    }
    else {
      s->rho_inv = 1. / rho_sc;
    }

    // Update KKT matrix with new rho_vec
    update_KKT_param2(s->KKT, s->rho_inv_vec, s->rho_inv, s->rhotoKKT, s->m);

    // Perform numerical factorization
    s->phase = PARDISO_NUMERIC;
    PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
             &(s->nKKT), s->KKT->x, s->KKT_p, s->KKT_i, &(s->idum), &(s->nrhs),
             s->iparm, &(s->msglvl), &(s->fdum), &(s->fdum), &(s->error));

    // Return exit flag
    return s->error;
}
