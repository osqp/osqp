#include "pardiso_interface.h"
#include "algebra_impl.h"
#include "printing.h"

#if OSQP_EMBEDDED_MODE != 1
#include "kkt.h"
#endif

#include "mkl_service.h"
#include "mkl_pardiso.h"

// Solver Phases
#define PARDISO_SYMBOLIC  (11)
#define PARDISO_NUMERIC   (22)
#define PARDISO_SOLVE     (33)
#define PARDISO_CLEANUP   (-1)

void update_settings_linsys_solver_pardiso(pardiso_solver*     s,
                                           const OSQPSettings* settings) {
  return;
}

// Warm starting not used by direct solvers
void warm_start_linsys_solver_pardiso(pardiso_solver*    s,
                                      const OSQPVectorf* x) {
  return;
}

// Free LDL Factorization structure
void free_linsys_solver_pardiso(pardiso_solver* s) {
  if (s) {

    // Free pardiso solver using internal function
    s->phase = PARDISO_CLEANUP;
    PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
             &(s->nKKT), &(s->fdum), s->KKT_p, s->KKT_i, &(s->idum), &(s->nrhs),
             s->iparm, &(s->msglvl), &(s->fdum), &(s->fdum), &(s->error));

    if ( s->error != 0 ){
      c_eprint("Error during MKL Pardiso cleanup: %d", (int)s->error);
    }
    // Check each attribute of the structure and free it if it exists
    if (s->KKT)         csc_spfree(s->KKT);
    if (s->KKT_i)       c_free(s->KKT_i);
    if (s->KKT_p)       c_free(s->KKT_p);
    if (s->bp)          c_free(s->bp);
    if (s->sol)         c_free(s->sol);
    if (s->rho_inv_vec) c_free(s->rho_inv_vec);

    // These are required for matrix updates
    if (s->PtoKKT)    c_free(s->PtoKKT);
    if (s->AtoKKT)    c_free(s->AtoKKT);
    if (s->rhotoKKT)  c_free(s->rhotoKKT);

    c_free(s);
  }
}


// Initialize factorization structure
OSQPInt init_linsys_solver_pardiso(pardiso_solver**    sp,
                                   const OSQPMatrix*   P,
                                   const OSQPMatrix*   A,
                                   const OSQPVectorf*  rho_vec,
                                   const OSQPSettings* settings,
                                   OSQPInt             polishing) {

  OSQPInt    i;         // loop counter
  OSQPInt    nnzKKT;    // Number of nonzeros in KKT
  OSQPInt    m, n;      // Dimensions of A
  OSQPInt    n_plus_m;  // n_plus_m dimension
  OSQPFloat* rhov;      // used for direct access to rho_vec data when polishing=false

  OSQPFloat sigma = settings->sigma;

  // Allocate private structure to store KKT factorization
  pardiso_solver *s = c_calloc(1, sizeof(pardiso_solver));
  *sp = s;

  // Size of KKT
  n = P->csc->n;
  m = A->csc->m;
  s->n = n;
  s->m = m;
  n_plus_m = n + m;
  s->nKKT = n_plus_m;

  // Sigma parameter
  s->sigma = sigma;

  // Polishing flag
  s->polishing = polishing;

  // Link Functions
  s->name            = &name_pardiso;
  s->solve           = &solve_linsys_pardiso;
  s->free            = &free_linsys_solver_pardiso;
  s->warm_start      = &warm_start_linsys_solver_pardiso;
  s->update_matrices = &update_linsys_solver_matrices_pardiso;
  s->update_rho_vec  = &update_linsys_solver_rho_vec_pardiso;
  s->update_settings = &update_settings_linsys_solver_pardiso;

  // Assign type
  s->type = OSQP_DIRECT_SOLVER;

  // Working vector
  s->bp = (OSQPFloat *)c_malloc(sizeof(OSQPFloat) * n_plus_m);

  // Solution vector
  s->sol = (OSQPFloat *)c_malloc(sizeof(OSQPFloat) * n_plus_m);

  // Parameter vector
  if (rho_vec) {
    s->rho_inv_vec = (OSQPFloat *)c_malloc(sizeof(OSQPFloat) * n_plus_m);
  }
  // else it is NULL

  // Form KKT matrix
  if (polishing){ // Called from polish()
    s->KKT = form_KKT(P->csc,A->csc,
                      1,  //format = 1 means CSR
                      sigma, s->rho_inv_vec, sigma,
                      OSQP_NULL, OSQP_NULL, OSQP_NULL);
  }
  else { // Called from ADMM algorithm

    // Allocate vectors of indices
    s->PtoKKT   = c_malloc(P->csc->p[n] * sizeof(OSQPInt));
    s->AtoKKT   = c_malloc(A->csc->p[n] * sizeof(OSQPInt));
    s->rhotoKKT = c_malloc(m * sizeof(OSQPInt));

    // Use s->rho_inv_vec for storing param2 = rho_inv_vec
    if (rho_vec) {
        rhov = rho_vec->values;
        for (i = 0; i < m; i++){
            s->rho_inv_vec[i] = 1. / rhov[i];
        }
    }
    else {
      s->rho_inv = 1. / settings->rho;
    }

    s->KKT = form_KKT(P->csc,A->csc,
                      1,  //format = 1 means CSR
                      sigma, s->rho_inv_vec, s->rho_inv,
                      s->PtoKKT, s->AtoKKT,s->rhotoKKT);
  }

  // Check if matrix has been created
  if (!(s->KKT)) {
	  c_eprint("Error in forming KKT matrix");
    free_linsys_solver_pardiso(s);
    return OSQP_LINSYS_SOLVER_INIT_ERROR;
  } else {
    // Adjust indexing for Pardiso
    nnzKKT = s->KKT->p[n_plus_m];
    s->KKT_i = c_malloc((nnzKKT) * sizeof(OSQPInt));
    s->KKT_p = c_malloc((n_plus_m + 1) * sizeof(OSQPInt));

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

#ifdef OSQP_USE_FLOAT
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
    c_eprint("Error during symbolic factorization: %d", (int)s->error);
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
    c_eprint("Error during numerical factorization: %d", (int)s->error);
    free_linsys_solver_pardiso(s);
    *sp = OSQP_NULL;
    return OSQP_LINSYS_SOLVER_INIT_ERROR;
  }
  if ( s->iparm[21] < n ) {
    // Error: Number of positive eigenvalues of KKT should be the same as dimension of P
    c_eprint("KKT matrix has fewer positive eigenvalues than it should. The problem seems to be non-convex.");
    return OSQP_NONCVX_ERROR;
  }

  return 0;
}

const char* name_pardiso(pardiso_solver* s) {
  return "Pardiso";
}

// Returns solution to linear system  Ax = b with solution stored in b
OSQPInt solve_linsys_pardiso(pardiso_solver* s,
                             OSQPVectorf*    b,
                             OSQPInt         admm_iter) {

  OSQPInt    j;
  OSQPInt    n  = s->n;
  OSQPInt    m  = s->m;
  OSQPFloat* bv = b->values;

  // Back substitution and iterative refinement
  s->phase = PARDISO_SOLVE;
  PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
            &(s->nKKT), s->KKT->x, s->KKT_p, s->KKT_i, &(s->idum), &(s->nrhs),
            s->iparm, &(s->msglvl), bv, s->sol, &(s->error));
  if ( s->error != 0 ){
    c_eprint("Error during linear system solution: %d", (int)s->error);
    return 1;
  }

  if (!(s->polishing)) {
    /* copy x_tilde from s->sol */
    for (j = 0 ; j < n ; j++) {
      bv[j] = s->sol[j];
    }

    /* compute z_tilde from b and s->sol */
    if (s->rho_inv_vec) {
      for (j = 0 ; j < m ; j++) {
        bv[j + n] += s->rho_inv_vec[j] * s->sol[j + n];
      }
    }
    else {
      for (j = 0 ; j < m ; j++) {
        bv[j + n] += s->rho_inv * s->sol[j + n];
      }
    }
  }
  return 0;
}

// Update solver structure with new P and A
OSQPInt update_linsys_solver_matrices_pardiso(pardiso_solver*   s,
                                              const OSQPMatrix* P,
                                              const OSQPInt*    Px_new_idx,
                                              OSQPInt           P_new_n,
                                              const OSQPMatrix* A,
                                              const OSQPInt*    Ax_new_idx,
                                              OSQPInt           A_new_n) {

  // Update KKT matrix with new P
  update_KKT_P(s->KKT, P->csc, Px_new_idx, P_new_n, s->PtoKKT, s->sigma, 1);

  // Update KKT matrix with new A
  update_KKT_A(s->KKT, A->csc, Ax_new_idx, A_new_n, s->AtoKKT);

  // Perform numerical factorization
  s->phase = PARDISO_NUMERIC;
  PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
           &(s->nKKT), s->KKT->x, s->KKT_p, s->KKT_i, &(s->idum), &(s->nrhs),
           s->iparm, &(s->msglvl), &(s->fdum), &(s->fdum), &(s->error));

  // Return exit flag
  return s->error;
}


OSQPInt update_linsys_solver_rho_vec_pardiso(pardiso_solver*    s,
                                             const OSQPVectorf* rho_vec,
                                             OSQPFloat          rho_sc) {

  OSQPInt    i;
  OSQPInt    m = s->m;
  OSQPFloat* rhov;

  // Update internal rho_inv_vec
  if (s->rho_inv_vec != OSQP_NULL) {
    rhov = rho_vec->values;
    for (i = 0; i < m; i++){
      s->rho_inv_vec[i] = 1. / rhov[i];
    }
  }
  else {
    s->rho_inv = 1. / rho_sc;
  }

  // Update KKT matrix with new rho_vec
  update_KKT_param2(s->KKT, s->rho_inv_vec, s->rho_inv, s->rhotoKKT, m);

  // Perform numerical factorization
  s->phase = PARDISO_NUMERIC;
  PARDISO (s->pt, &(s->maxfct), &(s->mnum), &(s->mtype), &(s->phase),
           &(s->nKKT), s->KKT->x, s->KKT_p, s->KKT_i, &(s->idum), &(s->nrhs),
           s->iparm, &(s->msglvl), &(s->fdum), &(s->fdum), &(s->error));

  // Return exit flag
  return s->error;
}
