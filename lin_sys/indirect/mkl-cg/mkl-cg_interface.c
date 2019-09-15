#include "mkl-cg_interface.h"
#include "mkl_rci.h"
#include "PG_debug.h"

MKL_INT cg_solver_init(mklcg_solver * s){

  MKL_INT mkln = s->n;
  MKL_INT rci_request = 1;
  OSQPVectorf_set_scalar(s->tmp,0.);

  //intialise the parameters
  dcg_init(&mkln, NULL, NULL, &rci_request,
           s->iparm, s->dparm, OSQPVectorf_data(s->tmp));

  //Set MKL control paramters
  s->iparm[7] = 1;        // maximum iteration stopping test
  s->iparm[8] = 0;        // residual stopping test
  s->iparm[9] = 0;        // user defined stopping test
  s->dparm[0] = 1.E-10;    // relative tolerance (default)

  //returns -1 for dcg failure, 0 otherwise
  dcg_check(&mkln, NULL, NULL, &rci_request,
            s->iparm, s->dparm, OSQPVectorf_data(s->tmp));

  return rci_request;

}

//Compute v2 = (P+sigma I + A'*(rho)*A)v1,
//where v1 and v2 are successive columns of tmp
//unclear if I can overwrite v1, so avoid it

void cg_times(OSQPMatrix* P,
              OSQPMatrix* A,
              OSQPVectorf* v1,
              OSQPVectorf* v2,
              OSQPVectorf* rho_vec,
              c_float sigma,
              OSQPVectorf* ywork){

  OSQPMatrix_Axpy(A, v1, ywork, 1.0, 0.0); //scratch space for (rho)*A*v1
  OSQPVectorf_ew_prod(ywork, ywork, rho_vec);
  OSQPVectorf_copy(v2,v1);
  OSQPMatrix_Axpy(P, v1, v2, 1.0, sigma); //v2 = (P+sigma I) v1
  OSQPMatrix_Atxpy(A, ywork, v2, 1.0, 1.0);
}



c_int init_linsys_mklcg(mklcg_solver ** sp,
                        const OSQPMatrix * P,
                        const OSQPMatrix * A,
                        c_float sigma,
                        const OSQPVectorf* rho_vec,
                        c_int polish){


  c_int   m = OSQPMatrix_get_m(A);
  c_int   n = OSQPMatrix_get_m(P);
  MKL_INT mkln = n;
  MKL_INT status;
  mklcg_solver* s = (mklcg_solver *)c_malloc(sizeof(mklcg_solver));
  *sp = s;

  //Just hold on to pointers to the problem
  //data, no copies or processing required
  s->P       = *(OSQPMatrix**)(&P);
  s->A       = *(OSQPMatrix**)(&A);
  s->rho_vec = *(OSQPVectorf**)(&rho_vec);
  s->sigma   = sigma;
  s->polish  = polish;
  s->m       = m;
  s->n       = n;

  //Link functions
  s->solve           = &solve_linsys_mklcg;
  s->free            = &free_linsys_mklcg;
  s->update_matrices = &update_matrices_linsys_mklcg;
  s->update_rho_vec  = &update_rho_linsys_mklcg;

  // Assign type
  s->type = MKL_INDIRECT_SOLVER;

  //Don't know the thread count.  Make it 999
  //as a marker to come back and fix this
  s->nthreads = mkl_get_max_threads();

  //allocate a vector 3*(m+n) for MKL workspace
  //NB: documentation says 3*n needed, not 4*n,
  //if we don't use a preconditioner
  s->tmp = OSQPVectorf_malloc(3*n);

  //RHS and LHS of the full KKT system to solve
  //Initialise lhs to zero since it provides the
  //cold start condition for the CG inner solver
  s->x = OSQPVectorf_calloc(n);

  //make subviews for the rhs.   OSQP passes
  //a different RHS pointer at every iteration,
  //so we will need to update these views every
  //pass.   Just point them at x for now.
  s->r1 = OSQPVectorf_view(s->x, 0, 0);
  s->r2 = OSQPVectorf_view(s->x, 0, 0);

  //subviews to tmp when computing M v1 = v2, where
  //M is the condensed matrix used in the CG iterations
  s->v1 = OSQPVectorf_view(s->tmp, 0, n);
  s->v2 = OSQPVectorf_view(s->tmp, n, n);

  status = cg_solver_init(s);
  return status;
}



c_int solve_linsys_mklcg(mklcg_solver * s,
                  OSQPVectorf* b){

  MKL_INT  rci_request  = 1;
  MKL_INT  mkln         = s->n;

  //initialise the parameters
  OSQPVectorf_set_scalar(s->tmp,0.);

  //Point our subviews at the OSQP RHS
  OSQPVectorf_view_update(s->r1, b,    0, s->n);
  OSQPVectorf_view_update(s->r2, b, s->n, s->m);

  //Set r_2 = rho . *r_2
  OSQPVectorf_ew_prod(s->r2, s->r2, s->rho_vec);

  //Compute r_1 = r_1 + A' (rho.*r_2)
  //This is the RHS for our CG solve
  OSQPMatrix_Atxpy(s->A, s->r2, s->r1, 1.0, 1.0);

  // Solve the CG system
  // -------------------
  //resets internal work and counters,
  //but we still be warmstarting from s->x
  cg_solver_init(s);

  while(1){
    //Call dcg to get the search direction
    dcg (&mkln, OSQPVectorf_data(s->x), OSQPVectorf_data(s->r1),
         &rci_request, s->iparm, s->dparm, OSQPVectorf_data(s->tmp));
    if(rci_request == 1){
      //multiply for condensed system.  We can use s->r2 as
      //work now since we already have the condensed rhs
      cg_times(s->P, s->A, s->v1, s->v2, s->rho_vec, s->sigma, s->r2);
    } else {
      break;
    }
  }

  if(rci_request == 0){  //solution was found for x.
    //OSQP wants us to return (x,Ax) in place
    OSQPVectorf_copy(s->r1, s->x);
    OSQPMatrix_Axpy(s->A, s->x, s->r2, 1.0, 0.0);
  }

  return rci_request; //0 on succcess, otherwise MKL CG error code

}



c_int update_matrices_linsys_mklcg(
                  mklcg_solver * s,
                  const OSQPMatrix *P,
                  const OSQPMatrix *A){
  s->P = *(OSQPMatrix**)(&P);
  s->A = *(OSQPMatrix**)(&A);
  return 0;
}



c_int update_rho_linsys_mklcg(
                   mklcg_solver * s,
                   const OSQPVectorf* rho_vec){
  s->rho_vec = *(OSQPVectorf**)(&rho_vec);
  return 0;
}



void free_linsys_mklcg(mklcg_solver * s){

  if(s->tmp){
    OSQPVectorf_free(s->tmp);
    OSQPVectorf_free(s->x);
    OSQPVectorf_view_free(s->r1);
    OSQPVectorf_view_free(s->r2);
    OSQPVectorf_view_free(s->v1);
    OSQPVectorf_view_free(s->v2);
  }
  c_free(s);
}
