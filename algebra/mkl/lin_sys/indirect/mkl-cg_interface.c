#include "algebra_impl.h"
#include "algebra_vector.h"
#include "reduced_kkt.h"
#include "mkl-cg_interface.h"
#include <mkl_rci.h>

OSQPFloat cg_compute_tolerance(OSQPInt    admm_iter,
                               OSQPFloat  rhs_norm,
                               OSQPFloat  scaled_prim_res,
                               OSQPFloat  scaled_dual_res,
                               OSQPFloat  reduction_factor,
                               OSQPFloat* eps_prev) {

  OSQPFloat eps = 1.0;

  if (admm_iter == 1) {
    // In case rhs = 0.0 we don't want to set eps_prev to 0.0
    if (rhs_norm < OSQP_CG_TOL_MIN)
      *eps_prev = 1.0;
    else
      *eps_prev = rhs_norm * reduction_factor;

    // Return early since scaled_prim_res and scaled_dual_res are meaningless before the first ADMM iteration
    return *eps_prev;
  }

  eps = reduction_factor * sqrt(scaled_prim_res * scaled_dual_res);
  eps = c_max(c_min(eps, (*eps_prev)), OSQP_CG_TOL_MIN);
  *eps_prev = eps;

  return eps;
}


MKL_INT cg_solver_init(mklcg_solver* s) {

  MKL_INT mkln = s->n;
  MKL_INT rci_request = 1;

  //initialise the parameters
  dcg_init(&mkln, NULL, NULL, &rci_request,
           s->iparm, s->dparm, OSQPVectorf_data(s->tmp));

  // Set MKL control parameters
  s->iparm[4] = s->max_iter; // Maximum number of iterations
  s->iparm[7] = 1;           // Enable maximum iteration stopping test
  s->iparm[8] = 0;           // Disable built-in residual stopping test
  s->iparm[9] = 1;           // Enable user defined stopping test

  // Enable the preconditioner if requested
  s->iparm[10] = (s->precond_type == OSQP_NO_PRECONDITIONER) ? 0 : 1;

  // Relative tolerance (default)
  s->dparm[0] = 1.E-5;

  //returns -1 for dcg failure, 0 otherwise
  dcg_check(&mkln, NULL, NULL, &rci_request,
            s->iparm, s->dparm, OSQPVectorf_data(s->tmp));

  return rci_request;
}


void cg_update_precond(mklcg_solver* s) {

  switch(s->precond_type) {
  /* No preconditioner, just initialize the inverse vector to all 1s */
  case OSQP_NO_PRECONDITIONER:
    OSQPVectorf_set_scalar(s->precond, 1.0);
    break;

  /* Diagonal preconditioner computation */
  case OSQP_DIAGONAL_PRECONDITIONER:
    reduced_kkt_diagonal(s->P, s->A, s->rho_vec, s->sigma, s->precond, s->precond_inv);
    break;
  }
}


OSQPInt init_linsys_mklcg(mklcg_solver**     sp,
                          const OSQPMatrix*  P,
                          const OSQPMatrix*  A,
                          const OSQPVectorf* rho_vec,
                          const OSQPSettings*settings,
                                OSQPFloat*   scaled_prim_res,
                                OSQPFloat*   scaled_dual_res,
                                OSQPInt      polish) {

  OSQPInt m = A->csc->m;
  OSQPInt n = P->csc->n;
  MKL_INT mkln = n;
  MKL_INT status;
  mklcg_solver* s = (mklcg_solver *)c_malloc(sizeof(mklcg_solver));
  *sp = s;

  //Just hold on to pointers to the problem
  //data, no copies or processing required
  s->P       = *(OSQPMatrix**)(&P);
  s->A       = *(OSQPMatrix**)(&A);
  s->sigma   = settings->sigma;
  s->polish  = polish;
  s->m       = m;
  s->n       = n;

  s->scaled_prim_res = scaled_prim_res;
  s->scaled_dual_res = scaled_dual_res;

  //if polish is false, use the rho_vec we get.
  //Otherwise, use rho_vec = ones.*(1/sigma)
  s->rho_vec = OSQPVectorf_malloc(m);
  if (!polish) {
      s->rho_vec = OSQPVectorf_copy_new(rho_vec);
  } else {
      OSQPVectorf_set_scalar(s->rho_vec, 1/settings->sigma);
  }

  //Link functions
  s->name            = &name_mklcg;
  s->solve           = &solve_linsys_mklcg;
  s->warm_start      = &warm_start_linys_mklcg;
  s->free            = &free_linsys_mklcg;
  s->update_matrices = &update_matrices_linsys_mklcg;
  s->update_rho_vec  = &update_rho_linsys_mklcg;
  s->update_settings = &update_settings_linsys_solver_mklcg;

  // Assign type
  s->type = OSQP_INDIRECT_SOLVER;

  // Assign preconditioner
  s->precond_type = settings->cg_precond;

  // Assign iteration limit
  s->max_iter = settings->cg_max_iter;

  // Assign tolerance-related settings
  s->reduction_interval = settings->cg_tol_reduction;
  s->tol_fraction       = settings->cg_tol_fraction;
  s->reduction_factor   = settings->cg_tol_fraction;
  s->cg_zero_iters      = 0;

  //Don't know the thread count.  Just use
  //the same thing as the pardiso solver
  s->nthreads = mkl_get_max_threads();

  //Initialise solver state to zero since it provides
  //cold start condition for the CG inner solver
  s->x = OSQPVectorf_calloc(n);

  //Workspace for CG products and polishing
  s->ywork = OSQPVectorf_malloc(m);

  //make subviews for the rhs.   OSQP passes
  //a different RHS pointer at every iteration,
  //so we will need to update these views every
  //time we solve. Just point them at x for now.
  s->r1 = OSQPVectorf_view(s->x, 0, 0);
  s->r2 = OSQPVectorf_view(s->x, 0, 0);

  //Allocate a 4*n vector for the MKL workspace
  // 1:n     = Vector to multiply by the matrix
  // n+1:2n  = Vector after multiplying by the matrix
  // 2n+1:3n = Vector to apply the preconditioner to
  // 3n+1:4n = Vector after application of the preconditioner
  s->tmp = OSQPVectorf_malloc(4*n);

  // Create subviews to tmp to aid the matrix-vector multiplication
  s->mvm_pre  = OSQPVectorf_view(s->tmp, 0, n);
  s->mvm_post = OSQPVectorf_view(s->tmp, n, n);

  // Subviews to tmp to aid the preconditioner application
  s->precond_pre  = OSQPVectorf_view(s->tmp, 2*n, n);
  s->precond_post = OSQPVectorf_view(s->tmp, 3*n, n);

  status = cg_solver_init(s);

  // Compute the preconditioner
  s->precond     = OSQPVectorf_malloc(n);
  s->precond_inv = OSQPVectorf_malloc(n);
  cg_update_precond(s);

  return status;
}


const char* name_mklcg(mklcg_solver* s) {
    switch(s->precond_type) {
  case OSQP_NO_PRECONDITIONER:
    return "MKL RCI Conjugate Gradient - No preconditioner";
  case OSQP_DIAGONAL_PRECONDITIONER:
    return "MKL RCI Conjugate Gradient - Diagonal preconditioner";
  }

  return "MKL RCI Conjugate Gradient - Unknown preconditioner";
}


OSQPInt solve_linsys_mklcg(mklcg_solver* s,
                           OSQPVectorf*  b,
                           OSQPInt       admm_iter) {

  MKL_INT   rci_request = 1;
  MKL_INT   mkln        = s->n;
  OSQPFloat rhs_norm    = 0.0;
  OSQPFloat res_norm    = 0.0;
  OSQPFloat eps         = 1.0;

  //Point our subviews at the OSQP RHS
  OSQPVectorf_view_update(s->r1, b,    0, s->n);
  OSQPVectorf_view_update(s->r2, b, s->n, s->m);

  // Compute the RHS for the CG solve and its norm
  reduced_kkt_compute_rhs(s->A, s->rho_vec, s->r1, s->r2, s->ywork);
  rhs_norm = OSQPVectorf_norm_inf(s->r1);

  // Compute the desired solution precision
  if (s->polish) {
    eps = c_max(rhs_norm * OSQP_CG_POLISH_TOL, OSQP_CG_TOL_MIN);
  } else {
    if (admm_iter == 1) {
      // On the first iteration, set reduction_factor to its default value
      s->reduction_factor = s->tol_fraction;
    } else if (s->cg_zero_iters >= s->reduction_interval) {
      // Otherwise. check to see if the tolerance reduction factor should be adapted.
      // This is done if CG is consistently never having to actually run.
      s->reduction_factor /= 2;
      s->cg_zero_iters = 0;
    }

    // Compute the new tolerance
    eps = cg_compute_tolerance(admm_iter, rhs_norm,
                               *(s->scaled_prim_res), *(s->scaled_dual_res),
                               s->reduction_factor, &(s->eps_prev));
  }

  // Solve the CG system
  // -------------------
  //resets internal work and counters,
  //but we still be warmstarting from s->x
  cg_solver_init(s);

  while (1) {
    //Call dcg to get the search direction
    dcg (&mkln, OSQPVectorf_data(s->x), OSQPVectorf_data(s->r1),
         &rci_request, s->iparm, s->dparm, OSQPVectorf_data(s->tmp));

    if (rci_request == 1) {
      // Multiply for reduced system.
      // mvm_pre and mvm_post are subviews of the cg workspace variable s->tmp.
      reduced_kkt_mv_times(s->P, s->A, s->rho_vec, s->sigma, s->mvm_pre, s->mvm_post, s->ywork );
    } else if (rci_request == 2) {
      // Check the stopping criteria, precond_pre contains the residual vector
      res_norm = OSQPVectorf_norm_inf(s->precond_pre);

      // It is our responsibility to break out of the loop once tolerance is reached
      if (res_norm < eps)
        break;
    } else if (rci_request == 3) {
      // Apply the preconditioner as (precond_post = precond.*precond_pre)
      OSQPVectorf_ew_prod(s->precond_post, s->precond_inv, s->precond_pre);
    } else {
      break;
    }
  }

  if ( (rci_request == 0) || (rci_request == 2) ) {  //solution was found for x.

    OSQPVectorf_copy(s->r1, s->x);

    if (!s->polish) {
      //OSQP wants us to return (x,Ax) in place
      OSQPMatrix_Axpy(s->A, s->x, s->r2, 1.0, 0.0);
    } else {
      //OSQP wants us to return (x,\nu) in place,
      // where r2 = \nu = rho.*(Ax - r2)
      OSQPMatrix_Axpy(s->A, s->x, s->r2, 1.0, -1.0);
      OSQPVectorf_ew_prod(s->r2, s->r2, s->rho_vec);
    }

    // Record if no CG iterations were performed
    if (s->iparm[3] == 0)
      s->cg_zero_iters++;
    else
      s->cg_zero_iters = 0;
  }

  return rci_request; //0 on succcess, otherwise MKL CG error code
}


void update_settings_linsys_solver_mklcg(struct mklcg_solver_* s,
                                         const OSQPSettings*   settings) {

  MKL_INT mkln = s->n;
  MKL_INT rci_request = 1;

  // New precoditioner type requested
  if (s->precond_type != settings->cg_precond) {
    s->precond_type = settings->cg_precond;

    // Enable the preconditioner if requested
    s->iparm[10] = (s->precond_type == OSQP_NO_PRECONDITIONER) ? 0 : 1;

    // Compute the new preconditioner
    cg_update_precond(s);
  }

  // Maximum number of iterations
  s->max_iter = settings->cg_max_iter;
  s->iparm[4] = settings->cg_max_iter;

  // Update adaptive tolerance parameters
  s->reduction_interval = settings->cg_tol_reduction;
  s->tol_fraction       = settings->cg_tol_fraction;

  //returns -1 for dcg failure, 0 otherwise
  dcg_check(&mkln, NULL, NULL, &rci_request,
            s->iparm, s->dparm, OSQPVectorf_data(s->tmp));

  return;
}


void warm_start_linys_mklcg(struct mklcg_solver_* self,
                            const OSQPVectorf*    x) {
  // TODO: Warm starting!
  return;
}


OSQPInt update_matrices_linsys_mklcg(mklcg_solver*     s,
                                     const OSQPMatrix* P,
                                     const OSQPInt*    Px_new_idx,
                                     OSQPInt           P_new_n,
                                     const OSQPMatrix* A,
                                     const OSQPInt*    Ax_new_idx,
                                     OSQPInt           A_new_n) {
  s->P = *(OSQPMatrix**)(&P);
  s->A = *(OSQPMatrix**)(&A);

  // Update the preconditioner (matrix-only update)
  cg_update_precond(s);

  return 0;
}


OSQPInt update_rho_linsys_mklcg(mklcg_solver*    s,
                              const OSQPVectorf* rho_vec,
                              OSQPFloat          rho_sc) {
  OSQPVectorf_copy(s->rho_vec, rho_vec);

  // Update the preconditioner (rho-only update)
  cg_update_precond(s);

  return 0;
}


void free_linsys_mklcg(mklcg_solver* s) {

  if (s->tmp) {
    OSQPVectorf_free(s->tmp);
    OSQPVectorf_free(s->rho_vec);
    OSQPVectorf_free(s->x);
    OSQPVectorf_free(s->ywork);
    OSQPVectorf_free(s->precond);
    OSQPVectorf_free(s->precond_inv);
    OSQPVectorf_view_free(s->r1);
    OSQPVectorf_view_free(s->r2);
    OSQPVectorf_view_free(s->mvm_pre);
    OSQPVectorf_view_free(s->mvm_post);
    OSQPVectorf_view_free(s->precond_pre);
    OSQPVectorf_view_free(s->precond_post);
  }
  c_free(s);
}
