#include "algebra_impl.h"
#include "mkl-cg_interface.h"
#include <mkl_rci.h>

MKL_INT cg_solver_init(mklcg_solver* s) {

  MKL_INT mkln = s->n;
  MKL_INT rci_request = 1;

  //initialise the parameters
  dcg_init(&mkln, NULL, NULL, &rci_request,
           s->iparm, s->dparm, OSQPVectorf_data(s->tmp));

  //Set MKL control parameters
  s->iparm[7] = 1;        // maximum iteration stopping test
  s->iparm[8] = 0;        // residual stopping test
  s->iparm[9] = 0;        // user defined stopping test
  s->dparm[0] = 1.E-5;    // relative tolerance (default)

  //returns -1 for dcg failure, 0 otherwise
  dcg_check(&mkln, NULL, NULL, &rci_request,
            s->iparm, s->dparm, OSQPVectorf_data(s->tmp));

  return rci_request;
}

//Compute v2 = (P+sigma I + A'*(rho)*A)v1,
//where v1 and v2 are successive columns of tmp
//unclear if I can overwrite v1, so avoid it

void cg_times(OSQPMatrix*  P,
              OSQPMatrix*  A,
              OSQPVectorf* v1,
              OSQPVectorf* v2,
              OSQPVectorf* rho_vec,
              OSQPFloat    sigma,
              OSQPVectorf* ywork) {

  OSQPMatrix_Axpy(A, v1, ywork, 1.0, 0.0); //scratch space for (rho)*A*v1
  OSQPVectorf_ew_prod(ywork, ywork, rho_vec);
  OSQPVectorf_copy(v2,v1);
  OSQPMatrix_Axpy(P, v1, v2, 1.0, sigma); //v2 = (P+sigma I) v1
  OSQPMatrix_Atxpy(A, ywork, v2, 1.0, 1.0);
}


OSQPInt init_linsys_mklcg(mklcg_solver**      sp,
                          const OSQPMatrix*   P,
                          const OSQPMatrix*   A,
                          const OSQPVectorf*  rho_vec,
                          const OSQPSettings* settings,
                          OSQPInt             polish) {

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

  //Don't know the thread count.  Just use
  //the same thing as the pardiso solver
  s->nthreads = mkl_get_max_threads();

  //allocate a vector 3*(m+n) for MKL workspace
  //NB: documentation says 3*n needed, not 4*n,
  //if we don't use a preconditioner
  s->tmp = OSQPVectorf_malloc(3*n);

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

  //subviews to tmp when computing M v1 = v2, where
  //M is the condensed matrix used in the CG iterations
  s->v1 = OSQPVectorf_view(s->tmp, 0, n);
  s->v2 = OSQPVectorf_view(s->tmp, n, n);

  status = cg_solver_init(s);
  return status;
}


const char* name_mklcg(mklcg_solver* s) {
  return "MKL RCI Conjugate Gradient";
}


OSQPInt solve_linsys_mklcg(mklcg_solver* s,
                           OSQPVectorf*  b,
                           OSQPInt       admm_iter) {

  MKL_INT  rci_request = 1;
  MKL_INT  mkln        = s->n;

  //Point our subviews at the OSQP RHS
  OSQPVectorf_view_update(s->r1, b,    0, s->n);
  OSQPVectorf_view_update(s->r2, b, s->n, s->m);

  //Set ywork = rho . *r_2
  OSQPVectorf_ew_prod(s->ywork, s->r2, s->rho_vec);

  //Compute r_1 = r_1 + A' (rho.*r_2)
  //This is the RHS for our CG solve
  OSQPMatrix_Atxpy(s->A, s->ywork, s->r1, 1.0, 1.0);

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
        //multiply for condensed system. v1 and v2 are subviews of
        //the cg workspace variable s->tmp.
        cg_times(s->P, s->A, s->v1, s->v2, s->rho_vec, s->sigma, s->ywork);
    } else {
      break;
    }
  }

  if (rci_request == 0) {  //solution was found for x.

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
  }

  return rci_request; //0 on succcess, otherwise MKL CG error code
}


void update_settings_linsys_solver_mklcg(struct mklcg_solver_* s,
                                         const OSQPSettings*   settings) {
    // TODO: Update settings!
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
  return 0;
}


OSQPInt update_rho_linsys_mklcg(mklcg_solver*    s,
                              const OSQPVectorf* rho_vec,
                              OSQPFloat          rho_sc) {
  OSQPVectorf_copy(s->rho_vec, rho_vec);
  return 0;
}


void free_linsys_mklcg(mklcg_solver* s) {

  if (s->tmp) {
    OSQPVectorf_free(s->tmp);
    OSQPVectorf_free(s->rho_vec);
    OSQPVectorf_free(s->x);
    OSQPVectorf_free(s->ywork);
    OSQPVectorf_view_free(s->r1);
    OSQPVectorf_view_free(s->r2);
    OSQPVectorf_view_free(s->v1);
    OSQPVectorf_view_free(s->v2);
  }
  c_free(s);
}
